# MisoTTS MLX Developer Guide

This developer guide provides an in-depth walkthrough of our GPU-accelerated **Apple Silicon MLX implementation** of MisoTTS. It outlines how each component operates, how weights are converted, and how to execute inference locally at real-time speeds.

---

## 🏗️ Technical Architecture & Directory Layout

Our MLX-optimized environment is contained entirely in the [miso_mlx/](file:///Users/ghchinoy/projects/misotts/miso_mlx) workspace directory:

```
miso_mlx/
├── miso_mlx_cli.py     # Interactive CLI (optimize, download, speak, clone)
├── mlx_model.py        # Re-implemented Llama Backbone & AR Decoder in MLX
├── mlx_generator.py    # Autoregressive generation loop & Top-K sampling
└── mlx_converter.py    # PyTorch-to-MLX Safetensors translation tool
```

---

## 1. Deep Dive: Model Structure (`mlx_model.py`)

`mlx_model.py` mirrors the original PyTorch model definitions in `sources/MisoTTS/models.py`, but translates every weight and tensor operation into native **MLX (`mlx.core` & `mlx.nn`)** arrays to achieve full GPU acceleration on Apple Silicon.

### Reusable Llama Transformer Block
Since both the **8B Backbone** and the **300M Decoder** are structurally Llama-style transformers (decoder-only architectures), we implement a highly configurable `LlamaTransformer` container that handles standard layer blocks:

```python
class TransformerBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, num_kv_heads: int, intermediate_dim: int, rope_base: float, eps: float):
        super().__init__()
        self.attention_norm = RMSNorm(dims, eps=eps)
        self.attention = Attention(dims, num_heads, num_kv_heads, rope_base)
        self.ffn_norm = RMSNorm(dims, eps=eps)
        self.feed_forward = FeedForward(dims, intermediate_dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Tuple[mx.array, mx.array]] = None):
        r, new_cache = self.attention(self.attention_norm(x), mask=mask, cache=cache)
        h = x + r
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, new_cache
```

### Key-Value Cache Mechanics in MLX
Unlike PyTorch, which typically requires pre-allocated, static tensors for Key-Value caches, MLX handles KV caching dynamically. Because of macOS's **Unified Memory Architecture**, we don't have to perform expensive copy operations between CPU and GPU memory. 

The attention cache is initialized as `None` and grows frame-by-frame:

```python
# Extract current cache if present
offset = cache[0].shape[1] if cache is not None else 0

# Apply RoPE positional embeddings
queries = self.rope(queries, offset=offset)
keys = self.rope(keys, offset=offset)

# Update Attention KV Cache
if cache is not None:
    k_cache, v_cache = cache
    keys = mx.concatenate([k_cache, keys], axis=1)
    values = mx.concatenate([v_cache, values], axis=1)
new_cache = (keys, values)
```

---

## 2. Autoregressive Codebook Generation (`mlx_generator.py`)

The main synthesis pipeline inside `mlx_generator.py` operates at a **12.5 Hz temporal frequency**. This means for every 80ms of synthesized speech, the generator executes a multi-stage autoregressive cycle.

### Step-by-Step Frame Generation Loop

```
1. Get dense frame representation from Backbone (8B Transformer)
                  │
                  ▼
2. Sample Codebook 0 (c0) using Top-K filter
                  │
                  ▼
3. Reset Decoder Cache (Caches are localized per-frame for 300M Transformer)
                  │
                  ▼
4. For cb = 1 to 31:
   ├── Run 300M Decoder on current codebooks
   ├── Project to Audio Vocab
   └── Sample Codebook (cb) using Top-K filter
                  │
                  ▼
5. Stack all 32 codebooks into a complete audio frame
                  │
                  ▼
6. Feed completed frame back into Backbone context for the next 80ms step
```

### Sampling with Top-K Filtering
To ensure natural speech variation and prevent mechanical repetitions, we implement a GPU-accelerated Top-K filter on MLX arrays:

```python
def sample_topk(logits: mx.array, topk: int, temperature: float) -> mx.array:
    logits = logits / temperature
    
    # Extract the K-th largest value across the vocabulary dimension
    val, _ = mx.topk(logits, topk, axis=-1)
    threshold = val[:, -1:]
    
    # Mask out all values below the K-th threshold
    mask = logits < threshold
    filtered_logits = mx.where(mask, -float("inf"), logits)
    
    # Sample from the probability distribution
    return mx.random.categorical(filtered_logits, num_samples=1)
```

---

## 3. Weight Key-Mapping & Transposition (`mlx_converter.py`)

The `mlx_converter.py` utility is used to bridge the gap between standard Hugging Face PyTorch weights and our clean MLX layer nomenclature.

### Linear Layer Transposition
PyTorch stores `Linear` weights as matrices of shape `[out_features, in_features]`. However, MLX expects `Linear` weights of shape `[in_features, out_features]` to optimize matrix multiplication under Metal.

When writing or loading converted weights, the converter automatically handles transposing these weight matrices (`.T`):

```python
# Concept used inside MLX loader
if "weight" in key and "embeddings" not in key and "norm" not in key:
    weight = weight.T  # Transpose PyTorch linear weight to MLX format
```

### Weight Naming Key Translation Table
The converter reads the original safetensors and outputs a precise mapping report at `miso_mlx/mlx_weights/pytorch_to_mlx_mapping.txt`. Example mapping transformations:

| PyTorch Source Key | MLX Destination Key | Description |
|---|---|---|
| `backbone.layers.X.sa_norm.scale` | `backbone.layers.X.attention_norm.weight` | Attention Layer RMSNorm |
| `backbone.layers.X.mlp_norm.scale` | `backbone.layers.X.ffn_norm.weight` | MLP Layer RMSNorm |
| `backbone.layers.X.attn.q_proj.weight` | `backbone.layers.X.attention.wq.weight` | Query projection matrix |
| `backbone.layers.X.attn.output_proj.weight` | `backbone.layers.X.attention.wo.weight` | Output projection matrix |
| `backbone.layers.X.mlp.w1.weight` | `backbone.layers.X.feed_forward.w1.weight` | Gate Projection (SiLU) |
| `backbone.layers.X.mlp.w2.weight` | `backbone.layers.X.feed_forward.w2.weight` | Down Projection |

---

## 4. Environment Verification & Testing Guide

Follow these steps to run, verify, and test your installation on your Mac.

### Step 1: Install Dependencies
Install MLX, Hugging Face Hub, and the Rust-backed Mimi codec dependencies:
```bash
pip install mlx moshi_mlx rustymimi safetensors huggingface_hub
```

### Step 2: Validate System Hardware & Directories
Run the optimizer command to scan your hardware, verify unified memory allocations, and initialize directories:
```bash
uv run python miso_mlx/miso_mlx_cli.py optimize
```

### Step 3: Extract and Translate Weight Headers
Run the weight mapping script to inspect the safetensors file structure and output the mapping template:
```bash
uv run python miso_mlx/mlx_converter.py
```
This reads the local checkpoint cache, prints a structural breakdown of the Llama sub-networks (8B Backbone vs 300M Decoder), and saves a complete map blueprint to `miso_mlx/mlx_weights/pytorch_to_mlx_mapping.txt` for inspection.
