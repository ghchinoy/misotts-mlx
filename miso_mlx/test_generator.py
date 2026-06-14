import os
import sys
import mlx.core as mx
import numpy as np
from pathlib import Path

project_root = Path("/Users/ghchinoy/projects/misotts")
sys.path.append(str(project_root))
sys.path.append(str(project_root / "miso_mlx"))

from mlx_model import ModelArgs, MisoTTSModel
from mlx_generator import MLXGenerator

print("Loading MLX model...")
mlx_args = ModelArgs()
mlx_model = MisoTTSModel(mlx_args)
mlx_weights_file = project_root / "miso_mlx" / "mlx_weights" / "model.safetensors"
mlx_model.load_weights(str(mlx_weights_file), strict=False)

generator = MLXGenerator(mlx_model)

text = "Hello"
speaker = 0

print(f"\n--- Running MLX generator for text: '{text}' ---")
# We will intercept/run the generate function step-by-step or just run it with printed steps
# Let's override the print inside generate loop or write a mini step runner here.

# Let's tokenize first
target_tokens, target_masks = generator._tokenize_text_segment(text, speaker)
print("Prompt tokens shape:", target_tokens.shape)
print("Prompt tokens:", target_tokens[:, -1].tolist())

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

print("Warmup done.")
print("c0_logits shape:", c0_logits.shape)
print("c0_logits min/max/mean:", float(c0_logits.min()), float(c0_logits.max()), float(c0_logits.mean()))

# Let's sample cb0
c0_sample = generator.generate(text, speaker, [], max_audio_length_ms=200)
print("Resulting array shape:", c0_sample.shape)
