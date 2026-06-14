import sys
from pathlib import Path

# Setup paths to ensure we can import moshi, moshi_mlx, and local modules
project_root = Path("/Users/ghchinoy/projects/misotts")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

original_sources_path = str(project_root / "sources" / "MisoTTS")
if original_sources_path not in sys.path:
    sys.path.append(original_sources_path)

# Disable PyTorch compilation to run in pure eager mode on CPU
import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

# Apply bitsandbytes import patch for unquantized layers on macOS
try:
    from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers
    patch_bitsandbytes_import_for_unquantized_layers()
except ImportError as e:
    print(f"Warning: Could not import moshi_compat: {e}")

import numpy as np
from huggingface_hub import hf_hub_download
from moshi.models import loaders
import mlx.core as mx
from moshi_mlx.models.mimi import mimi_202407, Mimi as MLXMimi

# Set reproducible seeds
torch.manual_seed(42)
np.random.seed(42)

mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)

# 1. Load PyTorch Mimi
pt_mimi = loaders.get_mimi(mimi_weight, device="cpu")
pt_mimi.set_num_codebooks(32)
pt_mimi.eval()

# 2. Load MLX Mimi
mlx_cfg = mimi_202407(32)
mlx_mimi = MLXMimi(mlx_cfg)
mlx_mimi.load_pytorch_weights(mimi_weight)
mlx_mimi.eval()

# 3. Create sample Mimi tokens
B, K, T = 1, 32, 20
codes_np = np.random.randint(0, 2048, size=(B, K, T)).astype(np.int64)

codes_pt = torch.from_numpy(codes_np)
codes_mx = mx.array(codes_np)

print("--- Inspecting Transformer config parameters ---")
if hasattr(pt_mimi, "decoder_transformer"):
    pt_trans = pt_mimi.decoder_transformer
    print("PyTorch Decoder Transformer Type:", type(pt_trans))
    if hasattr(pt_trans, "transformer"):
        pt_inner = pt_trans.transformer
        print("PyTorch Inner Transformer Layers Count:", len(pt_inner.layers) if hasattr(pt_inner, "layers") else "N/A")
        if hasattr(pt_inner, "layers") and len(pt_inner.layers) > 0:
            first_layer = pt_inner.layers[0]
            print("PyTorch LayerNorm Epsilon (norm1):", first_layer.norm1.eps if hasattr(first_layer, "norm1") else "N/A")
            print("PyTorch LayerNorm Epsilon (norm2):", first_layer.norm2.eps if hasattr(first_layer, "norm2") else "N/A")
            print("PyTorch Layer Scale Init:", first_layer.layer_scale_1 if hasattr(first_layer, "layer_scale_1") else "N/A")

if hasattr(mlx_mimi, "decoder_transformer"):
    mlx_trans = mlx_mimi.decoder_transformer
    print("MLX Decoder Transformer Type:", type(mlx_trans))
    if hasattr(mlx_trans, "transformer"):
        mlx_inner = mlx_trans.transformer
        print("MLX Inner Transformer Layers Count:", len(mlx_inner.layers) if hasattr(mlx_inner, "layers") else "N/A")
        if hasattr(mlx_inner, "layers") and len(mlx_inner.layers) > 0:
            first_layer = mlx_inner.layers[0]
            print("MLX LayerNorm Epsilon (norm1):", first_layer.norm1.eps if hasattr(first_layer, "norm1") else "N/A")
            print("MLX LayerNorm Epsilon (norm2):", first_layer.norm2.eps if hasattr(first_layer, "norm2") else "N/A")
            print("MLX Layer Scale (gamma1):", first_layer.gamma1 if hasattr(first_layer, "gamma1") else "N/A")

def compare_tensors(name, pt_tensor, mx_array):
    if isinstance(pt_tensor, torch.Tensor):
        pt_np = pt_tensor.detach().cpu().numpy().astype(np.float32)
    else:
        pt_np = np.array(pt_tensor, dtype=np.float32)
        
    mx_np = np.array(mx_array, dtype=np.float32)
    
    if pt_np.shape != mx_np.shape:
        print(f"Shape Mismatch for {name}: PyTorch {pt_np.shape} vs MLX {mx_np.shape}")
        return
        
    mae = np.mean(np.abs(pt_np - mx_np))
    max_err = np.max(np.abs(pt_np - mx_np))
    
    pt_flat = pt_np.flatten()
    mx_flat = mx_np.flatten()
    norm_pt = np.linalg.norm(pt_flat)
    norm_mx = np.linalg.norm(mx_flat)
    if norm_pt == 0 or norm_mx == 0:
        cos_sim = 1.0 if norm_pt == norm_mx else 0.0
    else:
        cos_sim = np.dot(pt_flat, mx_flat) / (norm_pt * norm_mx)
        
    print(f"{name}:")
    print(f"  Shape: {pt_np.shape}")
    print(f"  MAE:   {mae:.8e}")
    print(f"  Max:   {max_err:.8e}")
    print(f"  Cos:   {cos_sim:.8f}")
    return mae, max_err, cos_sim

print("\n=== Running Step-by-Step Forward Pass Comparison ===")

# --- Step 1: decode_latent ---
print("\n--- Step 1: Decode Latent (Quantizer Decode) ---")
with torch.no_grad():
    emb_pt = pt_mimi.decode_latent(codes_pt)

mlx_mimi.reset_all()
emb_mx = mlx_mimi.quantizer.decode(codes_mx)

compare_tensors("quantizer.decode", emb_pt, emb_mx)

# --- Step 2: upsample ---
print("\n--- Step 2: Upsample (ConvTrUpsample1d) ---")
with torch.no_grad():
    upsampled_pt = pt_mimi.upsample(emb_pt)
    
upsampled_mx = mlx_mimi.upsample(emb_mx)

compare_tensors("upsample", upsampled_pt, upsampled_mx)

# --- Step 3: decoder_transformer ---
print("\n--- Step 3: Decoder Transformer ---")
with torch.no_grad():
    transformer_pt = pt_mimi.decoder_transformer(upsampled_pt)
    if isinstance(transformer_pt, (list, tuple)):
        transformer_pt_tensor = transformer_pt[0]
        if isinstance(transformer_pt_tensor, (list, tuple)):
            transformer_pt_tensor = transformer_pt_tensor[0]
    else:
        transformer_pt_tensor = transformer_pt

# MLX
transformer_mx = mlx_mimi.decoder_transformer(upsampled_mx, cache=mlx_mimi.decoder_cache)
if isinstance(transformer_mx, (list, tuple)):
    transformer_mx_tensor = transformer_mx[0]
    if isinstance(transformer_mx_tensor, (list, tuple)):
         transformer_mx_tensor = transformer_mx_tensor[0]
else:
    transformer_mx_tensor = transformer_mx

compare_tensors("decoder_transformer", transformer_pt_tensor, transformer_mx_tensor)

# --- Step 4: decoder ---
print("\n--- Step 4: Decoder (SeanetDecoder) ---")
with torch.no_grad():
    out_pt = pt_mimi.decoder(transformer_pt_tensor)
    
out_mx = mlx_mimi.decoder(transformer_mx_tensor)

compare_tensors("decoder", out_pt, out_mx)

# --- Full decode method ---
print("\n--- Full Decode Method ---")
with torch.no_grad():
    full_out_pt = pt_mimi.decode(codes_pt)
    
mlx_mimi.reset_all()
full_out_mx = mlx_mimi.decode(codes_mx)

compare_tensors("full_decode_output", full_out_pt, full_out_mx)
