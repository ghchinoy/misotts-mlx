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
- `miso_studio/` - Premium SwiftUI desktop macOS studio client.
  - `Sources/ContentView.swift` - Responsive dashboard UI with parameter dials, logging console, and audio player.
  - `Sources/MisoSynthesisWorker.swift` - Thread-safe background subprocess runner that parses machine-readable telemetry.
- `miso_swift/` - Native Swift package for layer shape and model state parity verification.
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

### 5. Running the SwiftUI Desktop Studio
To compile and launch the SwiftUI Desktop Studio application natively on macOS:
```bash
make run-studio
```
Or to compile a standalone `.app` bundle:
```bash
make bundle-studio
```
All studio-generated WAV outputs are saved uniquely using date-time parameters and text slugs under `outputs/` while copying the latest audio to `outputs/studio_output.wav` for compatibility.

---

## 🛠️ Code Standards for Modifying Model Layers

1. **RoPE Attention Alignment**:
   - MisoTTS uses a custom `Llama3ScaledRoPE` layer. In the decoder layers, make sure to pass the active `input_pos=curr_pos` step-by-step.
   - Attention KV Cache is handled dynamically in MLX by passing `caches` arrays which grow step-by-step.
2. **Double-Mode Compatibility**:
   - Do not print directly to `stdout` inside CLI utilities without checking `is_json_mode`. Always direct normal progress/warning logs to `stderr` if in JSON mode so output parsing remains deterministic.
   - Honor `NO_COLOR` and `MISO_NO_TUI` environment variables to auto-disable ANSI terminal escape formatting.

---

## 🎛️ Core Autoregressive Parameter Intuitions

When generating speech or configuring agent defaults, use these guidelines:
1. **Temperature Decay (Start Temp / Min Temp / Decay Steps)**:
   - **Start Temp** (e.g., `1.0`): Higher temperatures increase phonetic and vocal diversity early in the sentence.
   - **Min Temp** (e.g., `0.2` or `0.3`): Lowering temperature as generation progresses stabilizes the autoregressive loop, preventing phonetic drift, slurring, or babbling at the end of long sentences.
   - **Decay Steps** (e.g., `200` steps): Sets how quickly the model transitions from creative exploration to strict structural stability.
2. **Classifier-Free Guidance (CFG)** (e.g., `1.5 - 3.0`):
   - Higher CFG values align the synthesis strongly with the target speaker's voice embedding (or cloned audio prompt traits), but extremely high values (>3.5) may cause acoustic saturation, clipping, or digital distortions. Moderate values balance voice fidelity and high-quality rendering.
3. **Autoregressive Duration Budgets (`max_length_ms`)**:
   - MisoTTS synthesizes speech step-by-step. It **does not** automatically stretch or compress speech to fit a designated time frame.
   - Instead, `max_length_ms` acts as an absolute budget safety-valve. If the budget is set too low (e.g., `10000ms` for a long text), synthesis will cut off mid-sentence.
   - **Heuristic Rule of Thumb**: Budget **100ms - 150ms per text character** (plus ~3000ms buffer if performing voice cloning to account for prompt context processing overhead).

---

## 🧹 Git Housekeeping & Giant Binary Avoidance

- **Strict Binary Exclusion**: Do not commit native Swift compilation folders (`.build`, `.swiftpm`, `DerivedData`) or Python virtual environments (`.venv`). These are ignored in `.gitignore`, but double-check using `git status` prior to committing.
- **Git Filter Repo**: In the event that a large binary or debug tool gets accidentally committed, immediately use `git-filter-repo` to purge it from all historical branches, ensuring the repository's clone speed and metadata size remain optimal.
