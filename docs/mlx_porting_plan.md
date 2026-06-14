# MLX Porting Plan: Optimizing MisoTTS for Apple Silicon (Option A)

This document outlines the step-by-step engineering plan for **Option A: Python + MLX**. This approach ports the core transformer networks to Apple's **MLX** framework to leverage the GPU and Unified Memory on Apple Silicon while keeping the outer scripts in Python.

---

## Technical Revelation: Mimi on MLX is Pre-Built!

The most complex part of porting MisoTTS is the **Mimi neural audio codec**. 
Fortunately, Kyutai Labs provides an officially optimized, Rust-backed Python implementation for Apple Silicon called **`moshi_mlx`** and **`rustymimi`**. 
This means:
1. We **do not** need to rewrite the Mimi encoder or decoder from scratch.
2. We can load and decode Mimi audio tokens natively on Mac at maximum performance using `moshi_mlx` and `rustymimi`!
3. We only need to focus on porting the custom **Llama 8B backbone** and the **300M autoregressive codebook decoder** to MLX.

---

## Step-by-Step Porting & Conversion Plan

### Phase 1: Environment & Requirements
Set up a clean Apple Silicon-optimized Python environment (recommended Python 3.12, as `moshi_mlx` and `rustymimi` compile natively here).

1. Install `mlx`, `moshi_mlx`, and `rustymimi`.
2. Add these to our local package requirements.

### Phase 2: Weight Extraction & Translation
MisoTTS weights are hosted on Hugging Face as standard PyTorch weights (`model.safetensors`). Before running in MLX, these weights need to be mapped to MLX-compatible key names and saved.

1. Create a weight conversion script (`mlx_converter.py`) that:
   * Downloads `MisoLabs/MisoTTS/model.safetensors` via `huggingface_hub`.
   * Maps PyTorch state-dict keys to standard MLX key formats (e.g., mapping PyTorch linear layers, attention projections, and embedding layers to MLX equivalents).
   * Saves the weights as a serialized `.npz` or MLX-native `.safetensors` file.

### Phase 3: Writing the MLX Model Definition
Re-implement the network classes in `models.py` using `mlx.core` and `mlx.nn`:

1. **`RMSNorm` / `LayerNorm`:** Implement MLX equivalents.
2. **`Attention` / `CausalAttention`:** Implement with MLX Rotary Position Embeddings (RoPE) and cached attention states for fast token-by-token generation.
3. **`Model` & `ModelArgs`:** Combine the Backbone and the Autoregressive Codebook Decoder into an unified MLX model structure.
4. **KV Cache Setup:** Implement efficient, dynamic KV-caching in MLX to speed up generation loops.

### Phase 4: Constructing the MLX Generator
Create an Apple Silicon-specific generator (`mlx_generator.py`) to replace `generator.py`:

1. Instantiate the MLX model and load the converted weights.
2. Load the Llama-3.2 text tokenizer.
3. Instantiate the Mimi codec using `moshi_mlx` / `rustymimi` on Apple Silicon.
4. Replace the generation loop (`generate()`) to run on the Mac GPU via MLX arrays.
5. Bypass the PyTorch `SilentCipher` watermarker if it causes performance degradation, or implement a lightweight MLX wrapper.

### Phase 5: Verification & Benchmark
Create comparison tests to verify that the MLX model outputs the exact same audio codebooks as the PyTorch model for identical inputs, and benchmark the generation latency (aiming for $\ge 20$ tokens/sec on your M4 GPU!).
