import os
import sys
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path

project_root = Path("/Users/ghchinoy/projects/misotts")
sys.path.append(str(project_root))
sys.path.append(str(project_root / "miso_mlx"))

from mlx_model import ModelArgs, MisoTTSModel
from mlx_generator import MLXGenerator, sample_topk

print("Loading MLX model...")
mlx_args = ModelArgs()
mlx_model = MisoTTSModel(mlx_args)
mlx_weights_file = project_root / "miso_mlx" / "mlx_weights" / "model.safetensors"
mlx_model.load_weights(str(mlx_weights_file), strict=False)

generator = MLXGenerator(mlx_model)

text = "Hello"
speaker = 0

# Tokenize text segment
target_tokens, target_masks = generator._tokenize_text_segment(text, speaker)

# Warmup backbone
backbone_caches = [None] * mlx_args.backbone_layers
curr_tokens = mx.expand_dims(target_tokens, axis=0)
curr_masks = mx.expand_dims(target_masks, axis=0)
input_pos = mx.arange(target_tokens.shape[0])

c0_logits, last_h, backbone_caches, _ = mlx_model(
    tokens=curr_tokens,
    tokens_mask=curr_masks,
    input_pos=input_pos,
    backbone_caches=backbone_caches
)

print("\n--- STEP 1 ---")
# Sample first codebook (codebook 0)
c0_sample = sample_topk(c0_logits, 50, 0.9)
print("c0_sample value:", int(c0_sample[0, 0]))

c0_embed = mlx_model._embed_audio(0, c0_sample)
curr_h = mx.concatenate([mx.expand_dims(last_h, axis=1), c0_embed], axis=1)
curr_sample = c0_sample

decoder_caches = [None] * mlx_args.decoder_layers
curr_pos = mx.arange(curr_h.shape[1])[None, :]

for cb in range(1, mlx_args.audio_num_codebooks):
    decoder_input = mlx_model.projection(curr_h)
    dec_mask = mx.array(nn.MultiHeadAttention.create_additive_causal_mask(decoder_input.shape[1]))
    
    decoder_h, decoder_caches = mlx_model.decoder(
        decoder_input,
        mask=dec_mask,
        caches=decoder_caches,
        input_pos=curr_pos
    )
    
    ci_logits = mx.matmul(decoder_h[:, -1, :], mlx_model.audio_head[cb - 1])
    ci_sample = sample_topk(ci_logits, 50, 0.9)
    ci_embed = mlx_model._embed_audio(cb, ci_sample)
    
    curr_sample = mx.concatenate([curr_sample, ci_sample], axis=-1)
    curr_h = ci_embed
    curr_pos = curr_pos[:, -1:] + 1

print("Generated frame (curr_sample) at Step 1:")
print(curr_sample.tolist()[0])

print("\n--- Preparing Step 2 ---")
# Feed current sample back as input for next backbone step
next_step_token = mx.concatenate([curr_sample, mx.zeros((1, 1), dtype=mx.int32)], axis=-1)
next_step_mask = mx.ones((1, mlx_args.audio_num_codebooks + 1), dtype=mx.bool_)
next_step_mask[0, -1] = False

# Forward pass through backbone for next step logits
c0_logits2, last_h2, backbone_caches, _ = mlx_model(
    tokens=mx.expand_dims(next_step_token, axis=0),
    tokens_mask=mx.expand_dims(next_step_mask, axis=0),
    input_pos=mx.array([target_tokens.shape[0]]),
    backbone_caches=backbone_caches
)

print("c0_logits2 shape:", c0_logits2.shape)
print("c0_logits2 min/max/mean:", float(c0_logits2.min()), float(c0_logits2.max()), float(c0_logits2.mean()))

c0_sample2 = sample_topk(c0_logits2, 50, 0.9)
print("c0_sample2 value:", int(c0_sample2[0, 0]))
