import os
import sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

project_root = Path("/Users/ghchinoy/projects/misotts")
sys.path.append(str(project_root / "miso_mlx"))

from mlx_model import ModelArgs, MisoTTSModel

print("=== Loading MLX Model ===")
mlx_args = ModelArgs()
mlx_model = MisoTTSModel(mlx_args)
mlx_weights_file = project_root / "miso_mlx" / "mlx_weights" / "model.safetensors"
mlx_model.load_weights(str(mlx_weights_file), strict=False)

trace_dir = project_root / "miso_mlx" / "pt_trace"
print(f"Loading PyTorch trace arrays from {trace_dir}...")

pt_tokens = np.load(trace_dir / "pt_tokens.npy")
pt_masks = np.load(trace_dir / "pt_masks.npy")
pt_last_h = np.load(trace_dir / "pt_last_h.npy")
pt_c0_logits = np.load(trace_dir / "pt_c0_logits.npy")
pt_curr_sample = np.load(trace_dir / "pt_curr_sample.npy")
pt_next_token = np.load(trace_dir / "pt_next_token.npy")
pt_next_mask = np.load(trace_dir / "pt_next_mask.npy")
pt_pos2 = np.load(trace_dir / "pt_pos2.npy")
pt_last_h2 = np.load(trace_dir / "pt_last_h2.npy")
pt_c0_logits2 = np.load(trace_dir / "pt_c0_logits2.npy")

print("\n=== Warmup (Step 1) Comparison ===")
# Convert numpy inputs to MLX arrays
tokens_mx = mx.array(pt_tokens)
masks_mx = mx.array(pt_masks)

# 1. Embedding & Sum
embeds_mx = mlx_model._embed_tokens(mx.expand_dims(tokens_mx, axis=0))
masked_embeds_mx = embeds_mx * mx.expand_dims(mx.expand_dims(masks_mx, axis=0), axis=-1)
h_mx = masked_embeds_mx.sum(axis=-2)

# 2. Backbone pass
seq_len = tokens_mx.shape[0]
pos_mx = mx.arange(seq_len)
mask_mx = mx.array(nn.MultiHeadAttention.create_additive_causal_mask(seq_len))
backbone_caches = [None] * mlx_args.backbone_layers

h_out_mx, backbone_caches = mlx_model.backbone(h_mx, mask=mask_mx, caches=backbone_caches, input_pos=pos_mx)
last_h_mx = h_out_mx[:, -1, :]

# 3. Codebook0 head logits
c0_logits_mx = mlx_model.codebook0_head(last_h_mx)

# Compute Differences
last_h_mx_np = np.array(last_h_mx.astype(mx.float32))
c0_logits_mx_np = np.array(c0_logits_mx.astype(mx.float32))

diff_last_h = np.abs(pt_last_h - last_h_mx_np)
diff_c0_logits = np.abs(pt_c0_logits - c0_logits_mx_np)

print(f"Step 1 - Backbone Output (last_h) Diff: Max = {diff_last_h.max():.6f}, Mean = {diff_last_h.mean():.6f}")
print(f"Step 1 - Codebook0 Logits Diff: Max = {diff_c0_logits.max():.6f}, Mean = {diff_c0_logits.mean():.6f}")

pt_c0_sample_val = pt_curr_sample[0, 0]
mlx_c0_sample_val = c0_logits_mx_np.argmax(axis=-1)[0]
print(f"PyTorch C0 Token Sampled: {pt_c0_sample_val}")
print(f"MLX C0 Token Sampled:     {mlx_c0_sample_val}")

print("\n=== Step 2 Comparison ===")
# In Step 2, we feed the previous sampled tokens back
next_token_mx = mx.array(pt_next_token)
next_mask_mx = mx.array(pt_next_mask)

embeds_mx2 = mlx_model._embed_tokens(next_token_mx)
masked_embeds_mx2 = embeds_mx2 * mx.expand_dims(next_mask_mx, axis=-1)
h_mx2 = masked_embeds_mx2.sum(axis=-2)

pos_mx2 = mx.array([seq_len])
mask_mx2 = mx.array(nn.MultiHeadAttention.create_additive_causal_mask(1))

h_out_mx2, backbone_caches = mlx_model.backbone(h_mx2, mask=mask_mx2, caches=backbone_caches, input_pos=pos_mx2)
last_h_mx2 = h_out_mx2[:, -1, :]
c0_logits_mx2 = mlx_model.codebook0_head(last_h_mx2)

# Compute Differences for Step 2
last_h_mx2_np = np.array(last_h_mx2.astype(mx.float32))
c0_logits_mx2_np = np.array(c0_logits_mx2.astype(mx.float32))

diff_last_h2 = np.abs(pt_last_h2 - last_h_mx2_np)
diff_c0_logits2 = np.abs(pt_c0_logits2 - c0_logits_mx2_np)

print(f"Step 2 - Backbone Output (last_h) Diff: Max = {diff_last_h2.max():.6f}, Mean = {diff_last_h2.mean():.6f}")
print(f"Step 2 - Codebook0 Logits Diff: Max = {diff_c0_logits2.max():.6f}, Mean = {diff_c0_logits2.mean():.6f}")

pt_c0_sample2_val = 1484 # from saved trace output
mlx_c0_sample2_val = c0_logits_mx2_np.argmax(axis=-1)[0]
print(f"PyTorch C0 Token Sampled (Step 2): {pt_c0_sample2_val}")
print(f"MLX C0 Token Sampled (Step 2):     {mlx_c0_sample2_val}")

print("\n--- Step 1 Top 5 Logits ---")
pt_top5_idx = np.argsort(pt_c0_logits[0])[-5:][::-1]
pt_top5_val = pt_c0_logits[0][pt_top5_idx]
print("PyTorch Top 5:", list(zip(pt_top5_idx, pt_top5_val)))

mlx_top5_idx = np.argsort(c0_logits_mx_np[0])[-5:][::-1]
mlx_top5_val = c0_logits_mx_np[0][mlx_top5_idx]
print("MLX Top 5:    ", list(zip(mlx_top5_idx, mlx_top5_val)))

print("\n--- Step 2 Top 5 Logits ---")
pt_top5_idx2 = np.argsort(pt_c0_logits2[0])[-5:][::-1]
pt_top5_val2 = pt_c0_logits2[0][pt_top5_idx2]
print("PyTorch Top 5:", list(zip(pt_top5_idx2, pt_top5_val2)))

mlx_top5_idx2 = np.argsort(c0_logits_mx2_np[0])[-5:][::-1]
mlx_top5_val2 = c0_logits_mx2_np[0][mlx_top5_idx2]
print("MLX Top 5:    ", list(zip(mlx_top5_idx2, mlx_top5_val2)))

