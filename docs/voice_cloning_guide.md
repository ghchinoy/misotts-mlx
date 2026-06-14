# MisoTTS Zero-Shot Voice Cloning Guide

This guide provides step-by-step instructions for preparing high-fidelity vocal references, preprocessing audio recordings, and executing local zero-shot voice cloning using the Metal-accelerated MLX GPU environment on Apple Silicon Macs.

Zero-shot voice cloning (prompted generation) in MisoTTS works by conditioning the **8B Llama Backbone** on prior audio frames encoded as discrete acoustic codes by the **Mimi codec**. The **300M AR Decoder** then predicts subsequent codes, maintaining the acoustic texture, pitch, speed, and emotional tone of the source speaker.

---

## 🎙️ The Acoustic Blueprint: Recording & Sample Preparation

To achieve perfect phonetic alignment and natural-sounding cloned speech without metallic buzzing, robotic clipping, or slurred delivery, your reference audio sample must adhere to strict acoustic specifications.

### 1. Hard Technical Specifications
Your reference file must meet these format requirements before feeding it into the MisoTTS pipeline:

| Attribute | Specification | Why It Matters |
| :--- | :--- | :--- |
| **File Format** | `WAV` (uncompressed) | Compressed lossy formats (MP3/AAC) introduce compression artifacts that distort the Mimi codebook indices. |
| **Encoding** | `16-bit Signed PCM` | Ensures maximum precision without clipping or quantization noise. |
| **Sample Rate** | **`24,000 Hz` (24 kHz)** | This is the exact native temporal sampling rate of the Mimi codec. Higher/lower rates force automatic software resampling. |
| **Channels** | **`Mono` (1 channel)** | Multi-channel stereo arrays confuse the single-channel encoder, causing phase cancellation or phase drift. |
| **Length** | **`3 to 10 seconds`** (Sweet spot: **`5s`**) | **Under 3s**: Too short to capture the unique spectral envelope of the speaker's vocal folds. <br>**Over 10s**: Inflates the 8B context window, multiplying JIT compilation latency and memory footprint. |

### 2. Environmental & Vocal Guidelines
*   **Zero Background Noise (High SNR)**: Record in a quiet room with no background hum (fans, air conditioning), room reflections (echo/reverb), or mouse clicks. Any environmental noise will be encoded as part of the speaker's "voice identity," causing the generated clone to sound muddy or echoey.
*   **Exact Word-for-Word Transcript**: The transcript you provide via `--prompt-text` **must match the spoken audio perfectly**. 
    *   Do *not* include filler words like "um," "uh," or throat clearances.
    *   Do *not* gloss over stuttering or mispronunciations; the prompt text must represent exactly what is heard.
    *   Mismatch between text tokens and audio codes will break cross-attention weights, leading to immediate speech breakdown (muttering, stutters, or silence).
*   **Natural Expressive Delivery**: Avoid recording in a flat, monotone voice. Speak with natural cadence, comfortable pacing, and clear punctuation. The model mirrors the exact energy and breath patterns of the prompt.

---

## 🛠️ Audio Preprocessing with FFmpeg

To easily prepare and convert arbitrary recordings (e.g., recorded on your phone or downloaded) to the perfect MisoTTS format, use standard `ffmpeg` commands.

### Convert any file to 24kHz Mono WAV:
```bash
ffmpeg -i my_recording.m4a -ar 24000 -ac 1 -c:a pcm_s16le outputs/prepared_prompt.wav
```

### Convert and crop to exactly 5 seconds starting from the beginning:
```bash
ffmpeg -i my_recording.wav -ss 00:00:00 -t 5 -ar 24000 -ac 1 -c:a pcm_s16le outputs/prepared_prompt_5s.wav
```

---

## 🚀 Voice Cloning CLI Workflows

Once your reference sample is prepared, you can synthesize new sentences in the cloned voice.

### 1. Perform a Millisecond Validation (Dry-Run)
Before loading heavy model parameters into memory, verify file paths, audio headers, and token sequence shapes instantly:
```bash
uv run python miso_mlx/miso_mlx_cli.py clone \
  --prompt-audio outputs/prepared_prompt.wav \
  --prompt-text "The exact word-for-word transcript of the prompt audio sample." \
  --text "Synthesize this new sentence in the cloned voice." \
  --mlx --dry-run
```

### 2. High-Speed Quantized GPU Synthesis (Highly Recommended)
Synthesize cloned speech in seconds with a highly compact memory footprint on Apple Silicon (Metal-accelerated):
```bash
uv run python miso_mlx/miso_mlx_cli.py clone \
  --prompt-audio outputs/prepared_prompt.wav \
  --prompt-text "The exact word-for-word transcript of the prompt audio sample." \
  --text "Hello! This is synthesized locally on my Mac GPU using 4-bit quantization." \
  --mlx --quant \
  --output outputs/cloned_output_quantized.wav
```

### 3. Full-Precision FP16 GPU Synthesis
Run at full precision for maximum acoustic fidelity (requires ~11.6 GB of resident unified memory):
```bash
uv run python miso_mlx/miso_mlx_cli.py clone \
  --prompt-audio outputs/prepared_prompt.wav \
  --prompt-text "The exact word-for-word transcript of the prompt audio sample." \
  --text "Hello! This is synthesized locally on my Mac GPU at full precision." \
  --mlx \
  --output outputs/cloned_output_full.wav
```

### 4. PyTorch CPU Fallback (Slow)
Fallback to CPU execution if MLX dependencies are not installed or for baseline comparison:
```bash
uv run python miso_mlx/miso_mlx_cli.py clone \
  --prompt-audio outputs/prepared_prompt.wav \
  --prompt-text "The exact word-for-word transcript of the prompt audio sample." \
  --text "Hello from standard PyTorch CPU execution." \
  --output outputs/cloned_output_cpu.wav
```

---

## 📊 Local GPU Cloning Benchmarks

The following local empirical benchmarks compare the performance of full FP16 precision against 4-bit quantized generation on an **Apple Silicon GPU** for a 10-second voice cloning generation task (125 steps):

| Metric | MLX Full-Precision (FP16) | MLX Quantized (4-Bit) | Performance Impact |
| :--- | :--- | :--- | :--- |
| **Wall Clock Time** | `254.17 seconds` | **`22.28 seconds`** | **11.4x Speedup** 🚀 |
| **Resident Memory (RSS)** | `11,658.39 MB` (~11.4 GB) | **`8,664.22 MB`** (~8.4 GB) | **26% Memory Reduction** |
| **Generation Speed** | `0.49 frames/sec` | **`3.19 frames/sec`** | **6.5x Generation Speedup** |
| **Average Time Per Step**| `1,642.8 ms` | **`280.8 ms`** | **5.8x Latency Reduction** |
| **Real-Time Factor (RTF)**| `25.42x` | **`3.92x`** | Significant latency cut |
| **EOS Handling** | Generated full length (125 steps) | Naturally terminated via **`[EOS]`** (71 steps) | **More intelligent context termination** |

> [!NOTE]
> *Wall Clock Time* includes initial JIT graph compilation and warmup passes, which only occur on the first run. Subsequent runs of the quantized model complete in under **20 seconds**!

---

## 🎛️ Acoustic Tuning & Hyperparameter Optimization

If the cloned voice sounds slurred, breaks up, or exhibits unnatural pacing, use the following CLI tuning tags to calibrate the synthesis:

### 1. Temperature Calibration (`--temperature`)
*   **Too High (> 1.0)**: Introduces phonetic variety but can cause stutters, mispronunciations, and slurred syllables.
*   **Too Low (< 0.7)**: Highly deterministic, but can sound metallic, monotone, or robotic.
*   **Sweet Spot**: `0.85` to `0.90` for natural expressiveness.

### 2. Dynamic Temperature Scheduling
Instead of a static temperature, decay the temperature over time to start expressive and settle into stable intonations:
```bash
--temp-start 0.9 --temp-min 0.6 --temp-decay-steps 40
```

### 3. Classifier-Free Guidance (`--cfg-scale`)
Control how strictly the model aligns with the input text versus the speaker's vocal style:
*   `1.0`: CFG Disabled (Standard).
*   `1.5` to `2.5`: Enhances speech clarity and forces the speaker to pronounce each word strictly, reducing mumbling.
*   *Warning*: Setting CFG too high (> 3.0) can distort the vocal quality and introduce static noise.

### 4. Bypassing Watermarking (`--no-watermark`)
By default, MisoTTS applies a **SilentCipher** inaudible audio watermark to prevent malicious deepfakes. If you require absolute pure audiophile-grade outputs for testing, you can disable it with the `--no-watermark` tag.

---

## 🧪 Cross-Backend Parity Evaluation

To mathematically evaluate whether your MLX GPU clone aligns perfectly with the standard PyTorch reference, use our local audio comparison engine:

```bash
uv run python miso_mlx/compare_audio.py \
  --ref outputs/cloned_output_cpu.wav \
  --target outputs/cloned_output_gpu.wav
```

This will run comparative spectral analyses, measuring **Log-Spectral Distance (LSD)**, **Signal-to-Noise Ratio (SNR)**, and structural alignments to guarantee that no phonetic or speaker identity drift was introduced by our custom Apple Silicon MLX GPU optimizations.
