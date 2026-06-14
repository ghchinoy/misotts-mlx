---
name: misotts-speech-generation
description: Synthesize high-fidelity English speech and perform zero-shot voice cloning locally on macOS GPU via MLX. Use when a user requests text-to-speech, speaker voice cloning with an audio reference, custom speaker indexing, adjusting sampling parameters, or bypassing watermarks.
compatibility: Requires macOS, Apple Silicon, and Python with mlx and dependencies synced via uv
---

# MisoTTS Speech Generation & Zero-Shot Cloning Skill

This skill provides comprehensive instructions for running local GPU-accelerated text-to-speech (TTS) synthesis and zero-shot voice cloning using the MLX port of MisoTTS on Apple Silicon.

---

## 🚀 Step-by-Step Workflows

Ensure dependencies are fully synchronized:
```bash
uv sync
```

### 1. Standard Speech Synthesis (bfloat16 MLX GPU)
Synthesize audio from raw text using the unquantized `bfloat16` model on your Mac's Metal GPU. Weight streaming utilizes Apple Silicon unified memory to keep CPU overhead near zero:

```bash
uv run python miso_mlx/miso_mlx_cli.py speak \
  --text "Hello! This is synthesized locally on my Mac using our unified GPU workspace." \
  --speaker 0 \
  --mlx \
  --output outputs/hello_unquantized.wav
```

### 2. High-Speed 4-bit Quantized Speech Synthesis
To save RAM (weights drop from 16.38 GB to 5.52 GB) and speed up synthesis (JIT compile times drop by up to 11x), add the `--quant` flag:

```bash
uv run python miso_mlx/miso_mlx_cli.py speak \
  --text "Hello! This is synthesized locally on my Mac using our unified GPU workspace." \
  --speaker 0 \
  --mlx \
  --quant \
  --output outputs/hello_4bit_quant.wav
```

### 3. Zero-Shot Voice Cloning
Clone any target voice by supplying a short (3–10s) audio reference file and its text transcription:

```bash
uv run python miso_mlx/miso_mlx_cli.py clone \
  --text "This sentence is spoken in my newly cloned voice profile!" \
  --prompt-audio "outputs/hello_unquantized.wav" \
  --prompt-text "Hello! This is synthesized locally on my Mac using our unified GPU workspace." \
  --output outputs/cloned_output.wav
```

### 4. Multi-Speaker Contextual Dialogue Generation
MisoTTS's Llama backbone supports chaining conversation turns contextually. By feeding prior segment objects (which store the speaker, text transcript, and raw waveform arrays) back into the prompt context list, multiple speakers can converse contextually while preserving distinct speaker timbres.

Run the pre-configured conversational demo:
```bash
uv run python miso_mlx/dialogue_generator.py
```

The script sequentially synthesizes:
*   **Turn 1 (Speaker 0):** *"Hello there! I am Speaker Zero. Welcome to our local dialogue demo."*
*   **Turn 2 (Speaker 1):** *"Hi! I am Speaker One. By chaining our model states, we can have a natural, multi-turn conversation."*
*   ...and contextually chains subsequent responses, saving the combined and normalized conversational track to `outputs/conversation_demo.wav`.

---


## 🛠️ Parameter Tuning & Optimization

Quantized models can sometimes drift, leading to premature cut-offs or sibilant feedback. Apply dynamic temperature decay, Classifier-Free Guidance (CFG), and bypass the SilentCipher watermark to resolve these artifacts:

```bash
uv run python miso_mlx/miso_mlx_cli.py speak \
  --text "Hello from local GPU! this is highly variable speech." \
  --mlx \
  --quant \
  --temp-start 0.7 \
  --temp-min 0.4 \
  --temp-decay-steps 30 \
  --cfg-scale 2.0 \
  --no-watermark \
  --output outputs/test_dynamic_opt.wav
```

### Key Parameters:
*   `--no-watermark`: Bypasses the SilentCipher 44.1 kHz acoustic watermarking pass to eliminate trailing whirring/background hum.
*   `--temp-start 0.7 --temp-min 0.4 --temp-decay-steps 30`: Smoothly decays entropy over 30 steps to prevent the accumulation of sibilant noise towards the tail of the sentence.
*   `--cfg-scale 2.0`: Increases text-guidance conditioning scale to prevent the model from generating silence loops.
*   `--max_length_ms <duration>`: Controls the hard autoregressive frame generation ceiling (defaults to `10000` / 10 seconds). For long text scripts, this parameter must be increased to prevent mid-word cutoff.

---

## ⏱️ Speech Duration & Safety Ceiling Heuristic

Because MisoTTS is an autoregressive transformer, it generates audio frame-by-frame. To prevent runaway generation, both the CLI and generator enforce a default safety limit of **10 seconds** (`--max_length_ms 10000`).

To avoid premature speech truncation, follow the human conversational speech approximation heuristic before choosing your `--max_length_ms` limit:

$$\text{Estimated Speaking Duration (seconds)} \approx \frac{\text{Word Count}}{2.2} + 2.0\text{ (padding for punctuation/JIT)}$$

### Quick Lookup & Selection Table:
| Word Count | Est. Speaking Time | Recommended `--max_length_ms` |
|:---|:---|:---|
| 1 - 15 words | ~3.0s - 8.8s | `10000` (Default 10s) |
| 16 - 30 words | ~8.9s - 15.6s | `20000` (20s) |
| 31 - 50 words | ~15.7s - 24.7s | `30000` (30s) |
| 51 - 100 words | ~24.8s - 47.4s | `50000` to `60000` (50s - 60s) |

*Always align your safety ceiling parameter (`--max_length_ms`) generously with the expected speaking duration to prevent frustrating mid-phrase cutoffs.*
