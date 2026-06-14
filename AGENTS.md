# MisoTTS Apple Silicon MLX Port: AI Developer / Agent Instructions

Welcome, fellow AI agent! This repository contains a Metal-accelerated MLX implementation of the **MisoTTS 8B** model for Apple Silicon. Since this codebase is designed to be fully **Agent-Aware**, please read and follow these standards.

---

## 📂 Repository Topology

- `miso_mlx/` - Local MLX workspace files.
  - `miso_mlx_cli.py` - Main CLI utility supporting dual-mode (Human/Agent) interactions.
  - `mlx_model.py` - Core PyTorch-to-MLX neural network structures.
  - `mlx_generator.py` - Step-by-step autoregressive speech generation and codec decoding.
  - `mlx_converter.py` - Safetensors weight translation utility.
  - `compare_audio.py` - Mathematical cross-backend audio comparison engine.
- `sources/MisoTTS/` - Unmodified PyTorch reference implementation (do not edit).
- `meta-llama/Llama-3.2-1B/` - Local cached copy of Llama 3.2 tokenizer assets to bypass gated HF authentication blocks.

---

## 🚀 Standard Workflows

### 1. Verification & Dry-Runs (AX Mode)
To validate pipeline inputs, file existences, and tokenization sequence shapes in **milliseconds** without loading heavy weight files or spinning up GPU compiler routines, always invoke the `--dry-run` flag with `--json` for machine-readable parsing:
```bash
uv run python miso_mlx/miso_mlx_cli.py --json speak --text "Dry-run validation" --mlx --dry-run
```

Expected output stdout (stderr contains debug logs):
```json
{
  "status": "dry_run_success",
  "text": "Dry-run validation",
  "speaker": 0,
  "token_count": 5,
  "backend": "mlx",
  "mlx_weights_found": true
}
```

### 2. High-Speed Speech Synthesis
To synthesize high-fidelity speech locally on Apple Silicon GPUs (Metal-accelerated):
```bash
uv run python miso_mlx/miso_mlx_cli.py speak --text "Hello from local GPU!" --mlx --output output_gpu.wav
```

### 3. PyTorch Reference Fallback
To generate a baseline PyTorch CPU reference file:
```bash
uv run python miso_mlx/miso_mlx_cli.py speak --text "Hello from PyTorch CPU!" --output output_cpu.wav
```

### 4. Cross-Backend Parity Evaluation
To mathematically verify that your MLX code changes have not introduced acoustic or phonetic alignment drift compared to the PyTorch reference:
```bash
uv run python miso_mlx/compare_audio.py --ref output_cpu.wav --target output_gpu.wav
```

---

## 🛠️ Code Standards for Modifying Model Layers

1. **RoPE Attention Alignment**:
   - MisoTTS uses a custom `Llama3ScaledRoPE` layer. In the decoder layers, make sure to pass the active `input_pos=curr_pos` step-by-step.
   - Attention KV Cache is handled dynamically in MLX by passing `caches` arrays which grow step-by-step.
2. **Double-Mode Compatibility**:
   - Do not print directly to `stdout` inside CLI utilities without checking `is_json_mode`. Always direct normal progress/warning logs to `stderr` if in JSON mode so output parsing remains deterministic.
   - Honor `NO_COLOR` and `MISO_NO_TUI` environment variables to auto-disable ANSI terminal escape formatting.
