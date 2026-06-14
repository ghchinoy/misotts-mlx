import sys
from pathlib import Path

project_root = Path("/Users/ghchinoy/projects/misotts")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

original_sources_path = str(project_root / "sources" / "MisoTTS")
if original_sources_path not in sys.path:
    sys.path.append(original_sources_path)

import torch
torch._dynamo.config.disable = True

# Apply bitsandbytes import patch for unquantized layers on macOS
try:
    from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers
    patch_bitsandbytes_import_for_unquantized_layers()
except ImportError as e:
    print(f"Warning: Could not import moshi_compat: {e}")

import numpy as np
import mlx.core as mx
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from moshi_mlx.models.mimi import mimi_202407, Mimi as MLXMimi

mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)

pt_mimi = loaders.get_mimi(mimi_weight, device="cpu")
pt_mimi.set_num_codebooks(32)
pt_mimi.eval()

mlx_cfg = mimi_202407(32)
mlx_mimi = MLXMimi(mlx_cfg)
mlx_mimi.load_pytorch_weights(mimi_weight)
mlx_mimi.eval()

# Check attention layers
B, T, D = 1, 40, 512
x_np = np.random.randn(B, T, D).astype(np.float32)

x_pt = torch.from_numpy(x_np)
x_mx = mx.array(x_np)

print("--- Comparing ProjectedTransformer step-by-step ---")

pt_trans = pt_mimi.decoder_transformer.transformer
mlx_trans = mlx_mimi.decoder_transformer.transformer

# Reset cache
mlx_mimi.reset_all()
mlx_cache = mlx_mimi.decoder_transformer.make_cache()

pt_x = x_pt
mlx_x = x_mx

def compare_tensors(name, pt_t, mx_a):
    pt_np = pt_t.detach().cpu().numpy().astype(np.float32)
    mx_np = np.array(mx_a, dtype=np.float32)
    mae = np.mean(np.abs(pt_np - mx_np))
    cos = np.dot(pt_np.flatten(), mx_np.flatten()) / (np.linalg.norm(pt_np) * np.linalg.norm(mx_np))
    print(f"  {name:25s} | MAE: {mae:.8e} | Cos: {cos:.8f}")

# First, compare the input projections if any
if pt_mimi.decoder_transformer.input_proj is not None:
    with torch.no_grad():
        proj_pt = pt_mimi.decoder_transformer.input_proj(x_pt)
    proj_mx = mlx_mimi.decoder_transformer.input_proj(x_mx)
    compare_tensors("input_proj", proj_pt, proj_mx)
    pt_x = proj_pt
    mlx_x = proj_mx

# Let's compare layer by layer
for i in range(len(pt_trans.layers)):
    print(f"\n--- Layer {i} ---")
    pt_layer = pt_trans.layers[i]
    mlx_layer = mlx_trans.layers[i]
    
    # 1. Compare Norm1
    with torch.no_grad():
        n1_pt = pt_layer.norm1(pt_x)
    n1_mx = mlx_layer.norm1(mlx_x)
    compare_tensors(f"norm1", n1_pt, n1_mx)
    
    # 2. Compare Self-Attention
    with torch.no_grad():
        attn_pt = pt_layer.self_attn(n1_pt, n1_pt, n1_pt)
    attn_mx = mlx_layer.self_attn(n1_mx, cache=mlx_cache[i].self_attn)
    compare_tensors(f"self_attn", attn_pt, attn_mx)
    
    # 3. Compare layer_scale_1
    with torch.no_grad():
        scaled1_pt = pt_layer.layer_scale_1(attn_pt)
    scaled1_mx = mlx_layer.layer_scale_1(attn_mx)
    compare_tensors(f"layer_scale_1", scaled1_pt, scaled1_mx)
    
    # Update state post self_attn block
    with torch.no_grad():
        pt_x = pt_x + scaled1_pt
    mlx_x = mlx_x + scaled1_mx
    compare_tensors(f"post_attn_add", pt_x, mlx_x)
    
    # 4. Compare Norm2
    with torch.no_grad():
        n2_pt = pt_layer.norm2(pt_x)
    n2_mx = mlx_layer.norm2(mlx_x)
    compare_tensors(f"norm2", n2_pt, n2_mx)
    
    # 5. Compare gating / MLP
    with torch.no_grad():
        if pt_layer.gating is None:
            # gelu activation
            gate_pt = pt_layer.linear2(torch.nn.functional.gelu(pt_layer.linear1(n2_pt)))
        else:
            gate_pt = pt_layer.gating(n2_pt)
            
    gate_mx = mlx_layer.gating(n2_mx)
    compare_tensors(f"gating/MLP", gate_pt, gate_mx)
    
    # 6. Compare layer_scale_2
    with torch.no_grad():
        scaled2_pt = pt_layer.layer_scale_2(gate_pt)
    scaled2_mx = mlx_layer.layer_scale_2(gate_mx)
    compare_tensors(f"layer_scale_2", scaled2_pt, scaled2_mx)
    
    # Update state post MLP
    with torch.no_grad():
        pt_x = pt_x + scaled2_pt
    mlx_x = mlx_x + scaled2_mx
    compare_tensors(f"post_mlp_add", pt_x, mlx_x)
