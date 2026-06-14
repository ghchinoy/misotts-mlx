---
name: audio-quality-evaluation
description: Evaluate, transcribe, and mathematically verify synthesized audio and speech outputs. Use when measuring Log-Mel Spectrogram MAE, Temporal Envelope Correlation, or when running Vertex AI Gemini or offline Gemma 4 multimodal quality evaluations.
compatibility: Requires macOS, Python with librosa and soundfile, and optional Vertex AI or local Gemma 4 environment
---

# Audio Quality & Parity Evaluation Skill

This skill provides step-by-step instructions and technical guidance for executing mathematical parity verification and AI-driven quality validation (via Gemini and Gemma 4) on synthesized speech outputs.

---

## 📊 1. Mathematical Parity Verification

To verify that your MLX code modifications or quantized layers have not introduced acoustic or phonetic alignment drift, run a spectral comparison against a PyTorch CPU baseline:

```bash
uv run python miso_mlx/compare_audio.py \
  --ref outputs/hello_unquantized.wav \
  --target outputs/hello_4bit_quant.wav
```

### Expected Core Metrics:
*   **Log-Mel Spectrogram MAE (Mean Absolute Error):** Measures distance in the frequency domain. Low is better.
*   **Spectral Cosine Similarity:** Measures timbre alignment. Parity is achieved when cosine similarity is **>0.80**.
*   **Temporal Envelope Correlation:** Measures phonetic and syllable timing alignment. Parity is achieved when correlation is **>0.60**.

---

## 🤖 2. Cloud-Based Auditing (Gemini 3.1 Flash Lite)

Run a programmatic cloud-based transcription, clarity, and alignment audit using Google GenAI SDK on Vertex AI. Ensure your environment has a configured GCP Project and run:

```bash
uv run python miso_mlx/audio_evaluator.py \
  --audio outputs/test_dynamic_opt.wav \
  --text "Hello from local GPU! this is highly variable speech." \
  --model gemini-3.1-flash-lite \
  --location global
```

The evaluator returns a structured report with:
*   **Acoustic Clarity & Noise Rating**
*   **Prosody & Naturalness**
*   **Completeness (Loop/cut-off detection)**
*   **Speech Quality and Alignment Scores**

---

## 💻 3. Offline Joint Auditing (Local Gemma 4 MLX GPU)

For a fully private, offline, and zero-cost quality evaluation, execute a joint vision-speech audit on your local GPU using **Gemma 4** via `mlx-vlm`:

```bash
# Runs local vision + speech joint validation using gemma 4
/Users/ghchinoy/projects/gemmma/.venv/bin/python miso_mlx/test_multimodal_validation.py
```

*   **Setup Pre-requisites**: Requires downloading weights for `google-gemma-4-E2B-it-qat-q4_0-unquantized` to `/Users/ghchinoy/projects/gemma/` and configuring a dedicated virtual environment with `mlx-vlm` and `transformers`.
*   **Prompt Formulation**: Interleave text instructions with explicit `<|image|>` and `<|audio|>` tokens separated by newline delimiters to avoid modal attention overlap collisions.
