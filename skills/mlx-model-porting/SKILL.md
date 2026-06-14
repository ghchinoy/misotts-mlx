---
name: mlx-model-porting
description: Port, migrate, and validate PyTorch neural network modules to Apple Silicon MLX format. Use when translating attention layers, customizing Llama RoPE scale factors, setting up step-by-step autoregressive KV caches, or mapping PyTorch checkpoint weights to MLX safetensors.
compatibility: Requires macOS, Apple Silicon, and Python with mlx installed
---

# MLX Model Porting Skill

This skill provides step-by-step instructions, architectural blueprints, and best practices for porting and optimizing large PyTorch transformer models to Apple Silicon using the MLX framework.

---

## Technical Workflow

### 1. Structure MLX Transformer Layers
In MLX, define modules using `mlx.nn.Module`. Re-implement RMSNorm, Linear projection layers, FeedForward, and attention blocks:

```python
import mlx.core as mx
import mlx.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, num_kv_heads: int, intermediate_dim: int, rope_base: float, eps: float):
        super().__init__()
        self.attention_norm = nn.RMSNorm(dims, eps=eps)
        self.attention = Attention(dims, num_heads, num_kv_heads, rope_base)
        self.ffn_norm = nn.RMSNorm(dims, eps=eps)
        self.feed_forward = FeedForward(dims, intermediate_dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Tuple[mx.array, mx.array]] = None):
        r, new_cache = self.attention(self.attention_norm(x), mask=mask, cache=cache)
        h = x + r
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, new_cache
```

### 2. Manage KV Cache Dynamically
Unlike PyTorch, which typically requires pre-allocating static tensors for Key-Value caches, MLX handles KV caching dynamically. Leverage Apple Silicon's Unified Memory Architecture by growing the cache array frame-by-frame:

```python
# Initialize or grow the attention cache step-by-step
offset = cache[0].shape[1] if cache is not None else 0

# Apply RoPE positional embeddings using the offset
queries = self.rope(queries, offset=offset)
keys = self.rope(keys, offset=offset)

# Update Attention KV Cache
if cache is not None:
    k_cache, v_cache = cache
    keys = mx.concatenate([k_cache, keys], axis=1)
    values = mx.concatenate([v_cache, values], axis=1)
new_cache = (keys, values)
```

### 3. Translate Weight Checkpoints
Write a weight translation utility to map, transpose, and serialize PyTorch checkpoint `.bin` or `.safetensors` files into MLX-compliant `.safetensors`:

1.  **Iterate State Dict**: Read PyTorch weight keys.
2.  **Transposing**: Make sure to transpose weight matrices for Linear projections since PyTorch linear layers store weights transposed relative to MLX’s linear layers:
    ```python
    # PyTorch weights for linear projection must be transposed for MLX:
    mlx_weight = pytorch_weight.T
    ```
3.  **Group Query Attention (GQA)**: Verify key/value projection weights conform to GQA group structures.
4.  **Serialization**: Save weight mappings to a translation blueprint file, then export arrays using `mx.save_safetensors()`.

---

## Verification & Dry-Runs (AX Mode)

Always perform lightweight validation checks to verify tokenization shapes, file pathways, and layer structures without loading full 16 GB weights:

```bash
# Execute a validation dry-run with json outputs
uv run python miso_mlx/miso_mlx_cli.py --json speak --text "Dry-run verification" --mlx --dry-run
```

Ensure validation logs redirect to `stderr` in JSON mode so `stdout` remains fully parseable by automated orchestrators.
