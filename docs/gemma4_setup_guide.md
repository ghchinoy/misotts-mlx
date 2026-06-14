# Local Gemma 4 Multimodal Validation Setup Guide

This guide describes how to configure, run, and utilize the unquantized **Gemma 4** model locally on macOS for **100% offline, private, and zero-cost speech and visual validation** of our MisoTTS MLX outputs.

Using local Gemma 4 via Apple's native **MLX GPU** framework provides a high-speed, secure alternative to cloud APIs (such as Vertex AI Gemini) with zero network latency or API fee overhead.

---

## 🛠️ Architecture & Requirements

Local validation utilizes `mlx-vlm` (MLX Vision-Language Model framework) to stream model weights directly to Apple Silicon's Unified Memory, compiling optimal Metal GPU shaders for blazing-fast inference.

### 📋 Prerequisites
1. **Apple Silicon Mac:** (M1, M2, M3, or M4 series chip with unified memory)
2. **Dedicated Virtual Environment:** Isolate your MLX-VLM dependencies from your standard Python workspace.
3. **Core Dependencies:**
   - `mlx-vlm` (>= 0.4.4)
   - `transformers`
   - `Pillow` (for test-image synthesis)

---

## ⚙️ Step-by-Step Installation

### 1️⃣ Step 1: Environment Setup
We recommend creating a specialized virtual environment (e.g., `gemmma-env` or similar) to manage the vision-language libraries cleanly:

```bash
# Create a dedicated virtual environment
uv venv .venv_gemma

# Activate the environment
source .venv_gemma/bin/activate

# Install the required MLX multimodal packages
uv pip install mlx-vlm transformers Pillow
```

---

### 2️⃣ Step 2: Download or Locate Gemma 4 Weights
Place your unquantized local Gemma 4 weights in a local model directory. 

For our pipeline, the unquantized local weights are expected in:
`/Users/ghchinoy/projects/gemma/google-gemma-4-E2B-it-qat-q4_0-unquantized`

If you are pulling weights directly from Hugging Face for the first time, you can let `mlx-vlm` download them automatically by specifying the Hugging Face repo ID in your script:
```python
model_path = "google/gemma-4-2b-it"
```

---

### 3️⃣ Step 3: Run the Local Gemma 4 Validation Pipeline

Our codebase includes a fully-configured validation script: [miso_mlx/test_multimodal_validation.py](file:///Users/ghchinoy/projects/misotts/miso_mlx/test_multimodal_validation.py). 

Run the validation script using your local Gemma 4 Python environment:

```bash
# Execute local vision-speech validation
/Users/ghchinoy/projects/gemmma/.venv/bin/python miso_mlx/test_multimodal_validation.py
```

This script will:
1. Synthesize a light-gray validation image displaying a red circle with white `"Gemma 4"` text using Python's Pillow library at `outputs/validation_test_image.png`.
2. Execute **Part 1 (Interleaved Multimodal Run):** Interleave the test image and a target Elvish pronunciation recording, prompt the model to describe the image and transcribe the audio, and print the joint response.
3. Execute **Part 2 (Audio-only Transcription Run):** Stream the synthesized MisoTTS WAV file `outputs/hello_miso_gpu.wav` and output its offline transcription.

---

## 💡 Best Practices for Interleaved Prompts

When running multimodal vision-language-audio models like Gemma 4, placing consecutive placeholder tokens (e.g., `[<image>, <audio>]`) directly in contact can trigger **cross-sensory attention collisions**, causing the model to seek audio data inside the image pixels (resulting in errors like *"There is no audio in the image"*).

To guarantee robust multi-sensor processing, always wrap your placeholder tokens in **explicit textual buffers**:

> [!TIP]
> **Recommended Interleaved Format:**
> ```python
> messages_interleaved = [
>     {
>         "role": "user",
>         "content": [
>             {"type": "text", "text": "Below is our visual canvas:\n"},
>             {"type": "image"},
>             {"type": "text", "text": "\nAnd here is the corresponding speech recording:\n"},
>             {"type": "audio"},
>             {"type": "text", "text": "\nPlease describe the image and transcribe the audio."}
>         ]
>     }
> ]
> ```
> Adding these simple textual breaks lets the model's self-attention layers cleanly partition and decode separate modality token registers.

---

## 📊 Gemma 4 Offline Metrics & Telemetry

Programmatic profiling on an Apple Silicon GPU yields the following offline benchmarks:

```
=== Local Gemma 4 Generation Telemetry ===
ℹ   Model Load Time:                   4.79 seconds
ℹ   Audio Transcription Duration:      0.85 seconds
ℹ   Generation Speed (Audio-only):     24.84 tokens/second
ℹ   Peak Memory Consumption (RSS):     10.80 GB
```

* **Zero-Cost Iterations:** Bypasses cloud endpoints completely, facilitating continuous regression testing of vocoder adjustments.
* **Low Memory Footprint:** Fits comfortably within a standard Mac unified memory layout, requiring only **10.8 GB** of resident memory during active processing.
