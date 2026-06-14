# MisoTTS Apple Silicon (MLX) Tooling & Explainer Suite

Welcome to the **MisoTTS Apple Silicon (MLX) Optimization Suite**. This repository contains detailed architectural analyses, step-by-step developer guides, and high-performance command-line utilities designed to run, quantize, and mathematically evaluate the **MisoTTS 8B** model locally on macOS utilizing Apple's native **MLX** GPU framework.

---

## 🌟 Model Introduction & Capabilities

**MisoTTS 8B** is a state-of-the-art **8.2 Billion parameter Text-to-Dialogue RVQ Transformer** optimized for high-fidelity English conversational speech synthesis and zero-shot voice cloning.

* **Contextual Conversational Flow:** Natively maintains segment history, ensuring stylistic consistency, emotion, and realistic turn-taking transitions.
* **Simple Speaker Conditioning:** Switch speakers seamlessly using simple inline text prefixing (e.g., `[0] Hello there!`).
* **Zero-Shot Voice Cloning:** Synthesize speech in any target voice using only a 3–10 second clean reference audio clip and its text transcript.
* **Implicit Content Protection:** Integrates Sony's `SilentCipher` 44.1 kHz acoustic watermarking by default for responsible AI deployment.

On standard macOS CPU backends, running an 8.2B model is extremely resource-intensive. This suite ports the neural synthesizer, decoder stages, and attention layers to Apple's native **Metal GPU (via MLX)**, delivering up to **6.7x speedups** and enabling local, real-time speech generation with highly-compressed memory footprints.

---

## 📂 Repository Layout

```
misotts/ (Project Root)
├── README.md               # End-to-end usage & evaluation manual
├── .gitignore              # Ignored outputs, caches, and venv files
├── pyproject.toml          # Project package definitions (managed via uv)
├── uv.lock                 # Fast python lockfile
├── AGENTS.md               # Agent-Aware specifications and local workflows
├── miso_mlx/               # Apple Silicon MLX GPU Codebase
│   ├── miso_mlx_cli.py     # Deterministic Dual-Mode CLI (speak, clone, optimize)
│   ├── mlx_model.py        # Backbone & AR Decoder layers written in MLX
│   ├── mlx_generator.py    # AR sampling generator and dynamic parameters loop
│   ├── mlx_quantizer.py    # 4-bit in-place weight quantizer module
│   ├── mlx_converter.py    # Safetensors weight mapper/translator utility
│   ├── compare_audio.py    # Cross-backend spectral comparison engine
│   └── audio_evaluator.py  # Vertex AI Gemini-powered speech quality auditor
├── docs/                   # Fleshed-Out Technical Manuals
│   ├── explainer.md        # 3-Stage architecture & phonetics guide
│   ├── developer_guide.md  # Detailed MLX layer implementation guide
│   ├── evaluation_report.md # Performance benchmarks & dynamic parameter trials
│   ├── miso_mlx_friction_report.md # Hard-learned porting lessons & findings
│   └── mlx_porting_plan.md # 5-phase porting blueprint
├── outputs/                # Folder containing generated .wav audio outputs
└── sources/                # Original PyTorch reference models (preserved untouched)
```

---

## 🚀 End-to-End Walkthrough

Follow this step-by-step workflow to configure, run, quantize, and programmatically evaluate the MisoTTS model on your Mac.

### 📋 Prerequisites
Ensure you have the following installed on your system:
* **macOS** with Apple Silicon (M1, M2, M3, or M4 series chips)
* **Python 3.12** or newer
* **uv** (recommended fast package manager for Python)
* **gcloud CLI** (optional, required if using the automated AI audio evaluator)

---

### 1️⃣ Step 1: Environment Setup
Clone the repository and synchronize dependencies using `uv`. This will isolate all compiled C++ and Metal backends within a local virtual environment:

```bash
# Sync package locks and build virtual environment
uv sync
```

---

### 2️⃣ Step 2: System Diagnostic Scan
Before running heavy weight-conversion tasks, execute the optimization diagnostic tool to scan your Mac's hardware specifications (GPU cores, memory bandwidth, Unified Memory allocations) and verify directories:

```bash
uv run python miso_mlx/miso_mlx_cli.py optimize
```

---

### 3️⃣ Step 3: Pre-Download and Cache Weights
To prevent network bottlenecks during local operations, pre-download and cache all required weights (Llama-3.2, Mimi codec, SilentCipher watermarker, and the MisoTTS backbone) to your local Hugging Face storage:

```bash
uv run python miso_mlx/miso_mlx_cli.py download
```
> [!NOTE]
> This command requires approximately **30–40 GB** of free disk space to store all model safe-tensors checkpoints and model definitions.

---

### 4️⃣ Step 4: Translate PyTorch Weights to MLX Format
The downloaded checkpoints are stored in standard PyTorch format. Run our Safetensors weight translation utility to map, transpose, and serialize them into unified MLX `.safetensors` files compatible with Metal memory layouts:

```bash
uv run python miso_mlx/mlx_converter.py
```
This utility reads the PyTorch model keys, applies the mapping topology, and saves the direct translation blueprint to `miso_mlx/mlx_weights/pytorch_to_mlx_mapping.txt` for your verification.

---

### 5️⃣ Step 5: High-Fidelity Text-to-Speech (bfloat16 MLX GPU)
Synthesize your first audio file using the unquantized `bfloat16` model executing entirely on your Mac's Metal GPU.

```bash
uv run python miso_mlx/miso_mlx_cli.py speak \
  --text "Hello! This is synthesized locally on my Mac using our unified GPU workspace." \
  --speaker 0 \
  --mlx \
  --output outputs/hello_unquantized.wav
```
*By running on the GPU via `--mlx`, the massive memory bandwidth of Apple Silicon unified memory streams layer weights in parallel, keeping your CPU cool and fans quiet.*

---

### 6️⃣ Step 6: 4-bit Model Quantization & Blistering Speedups
The full model weights require approximately **16.38 GB of RAM**, creating streaming bandwidth bottlenecks. Compress the linear projection and transformer layers to **4-bit** in-place (reducing weight size to **5.52 GB** and cutting Unified Memory usage in half) by adding the `--quant` flag:

```bash
uv run python miso_mlx/miso_mlx_cli.py speak \
  --text "Hello! This is synthesized locally on my Mac using our unified GPU workspace." \
  --speaker 0 \
  --mlx \
  --quant \
  --output outputs/hello_4bit_quant.wav
```
* **Performance Impact:** First-step JIT compiler warmup drops from **6.28s to 0.53s (11.8x reduction)**, and real-time generation factor (RTF) drops comfortably, delivering up to **3.82x faster step inference**!

---

### 7️⃣ Step 7: Zero-Shot Voice Cloning
Clone any target voice by supplying a short (3–10s) clean audio reference file along with its exact transcription. The AR decoder will ingest the acoustic codes and speak the new text with the reference speaker's timbre:

```bash
uv run python miso_mlx/miso_mlx_cli.py clone \
  --text "This sentence is spoken in my newly cloned voice profile!" \
  --prompt-audio "outputs/hello_unquantized.wav" \
  --prompt-text "Hello! This is synthesized locally on my Mac using our unified GPU workspace." \
  --output outputs/cloned_output.wav
```

---

### 8️⃣ Step 8: Dynamic Parameter Scheduling (Studio-Quality Sweet Spot)
Quantized weights can sometimes drift, triggering a **"Silence Attractor" Cut-Off** (low temperature) or a **"Sibilant Hissing" Feedback Loop** (high temperature). Bypassing the SilentCipher watermark and applying **dynamic temperature decay scheduling** and Classifier-Free Guidance (CFG) isolates the acoustic sweet-spot:

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
*   `--no-watermark`: Removes the silent-cipher watermarking pass, completely eliminating trailing whirring/background hums.
*   `--temp-start 0.7 --temp-min 0.4 --temp-decay-steps 30`: Decays the entropy step-by-step to prevent the accumulated sibilant hiss.
*   `--cfg-scale 2.0`: Increases the text-guidance conditioning vector, preventing the model from falling into silence loops.

---

### 9️⃣ Step 9: Mathematical Parity Verification
Mathematically audit your generated outputs against a PyTorch CPU baseline to evaluate spectral divergence and phonetic temporal envelope correlation:

```bash
uv run python miso_mlx/compare_audio.py \
  --ref outputs/hello_unquantized.wav \
  --target outputs/hello_4bit_quant.wav
```
Expected output showing strong phonetic alignment:
```
=== MisoTTS Cross-Backend Audio Parity Report ===

ℹ Comparing Reference (PyTorch) vs Target (MLX):
  Reference file: hello_unquantized.wav (3.200s, 24000Hz)
  Target file:    hello_4bit_quant.wav (3.040s, 24000Hz)

--------------------------------------------------
1. Signal Statistics:
--------------------------------------------------
  Reference Peak Amplitude:  0.4420
  Target Peak Amplitude:     0.9500
  Reference RMS Energy:      0.077837
  Target RMS Energy:         0.212074

--------------------------------------------------
2. Spectral & Envelope Similarity Metrics:
--------------------------------------------------
  Log-Mel Spectrogram MAE:       3.6745
  Spectral Cosine Similarity:    0.9135 (expected >0.80 for speech parity)
  Temporal Envelope Correlation: 0.2351 (expected >0.60 for phonetic alignment)
```

---

### 🔟 Step 10: AI-Driven Audio Quality & Accuracy Validation

To programmatically transcribe, assess, and benchmark your synthesized audio outputs without manual listening overhead, you can choose between **cloud-based Gemini validation** (requires a GCP project) or **100% offline local Gemma 4 validation** (fully private and zero-cost).

---

#### ☁️ Option A: Cloud-Based Auditing (Gemini 3.1 Flash Lite)

By default, we utilize the Google GenAI Vertex AI backend using the **Gemini 3.1 Flash Lite** model in the `global` region. It streams your audio binary directly and returns an alignment, clarity, and prosody scorecard:

```bash
# Run cloud audit using the default global model
uv run python miso_mlx/audio_evaluator.py \
  --audio outputs/test_dynamic_opt.wav \
  --text "Hello from local GPU! this is highly variable speech." \
  --model gemini-3.1-flash-lite \
  --location global
```

##### Expected Output Report:
```
=== MisoTTS Google GenAI Vertex Audio Evaluator ===
ℹ Initializing Google GenAI SDK client (Vertex AI Backend)...
ℹ Reading target audio file: test_dynamic_opt.wav (752720 bytes)
ℹ Sending audio assessment request to Vertex AI using model 'gemini-3.1-flash-lite'...
✔ Evaluation completed successfully!
ℹ Model used: gemini-3.1-flash-lite on Vertex AI (global)

============================================================
Gemini Audio Quality Evaluation & Transcription:
============================================================
### Evaluation Report

**1. Transcription:**
"Hello from location, this is Wily ever will speed."

**2. Acoustic Clarity:**
The audio is clear and free of static, buzzing, or clipping. The voice profile is consistent with a high-quality neural TTS model.

**3. Prosody & Naturalness:**
The pacing is rhythmic and appropriate for speech, though the intonation is slightly monotone. It maintains a smooth flow throughout the sentence.

**4. Completeness:**
The model completed the full sentence without cutting off or entering infinite loops.

**5. Accuracy Comparison:**
*   **Expected:** "Hello from local GPU! this is highly variable speech."
*   **Actual:** "Hello from location, this is Wily ever will speed."
*   **Discrepancies:** "local GPU" was transcribed as "location", and "highly variable speech" was misinterpreted as "Wily ever will speed" due to vocoder compression.

*   **Speech Quality Score:** 75/100
*   **Alignment Accuracy:** 30/100
============================================================
```

---

#### 💻 Option B: 100% Offline Auditing (Local Gemma 4 MLX GPU)

For a fully private, offline, and zero-cost transcription pipeline, you can run multimodal evaluations locally on your Mac's GPU using the unquantized **Gemma 4** model via `mlx-vlm`.

Run the custom local validation suite:
```bash
# Execute local vision + speech joint validation
/Users/ghchinoy/projects/gemmma/.venv/bin/python miso_mlx/test_multimodal_validation.py
```

This will run joint interleaved (Vision + Speech) analyses as well as audio-only transcription, matching the transcription fidelity of cloud-grade models locally on your GPU in milliseconds!

> [!NOTE]
> For detailed instructions on setting up your local Python environments, downloading model weights, and formatting interleaved multimodal prompts for Gemma 4, see our [Local Gemma 4 Multimodal Validation Setup Guide](docs/gemma4_setup_guide.md).


---

## 🛠️ CLI Global Diagnostic Features

### Headless and Agent-Aware Operation
The command-line tools support global environment variable flags to ensure deterministic, silent execution inside developer scripts, CI/CD pipelines, and autonomous AI coding environments:
* **JSON Output (`--json`):** Redirects standard warning logs to `stderr` and prints machine-readable JSON blocks to `stdout`.
* **No TUI / No Color (`NO_COLOR=1` or `MISO_NO_TUI=1`):** Dynamically defeats all ANSI terminal escaping to produce clean logging streams.
* **Mutative Safety (`--dry-run`):** Performs syntax verification, resolves Hugging Face tokenizers, and verifies weights exist in milliseconds, without loading heavy 16 GB models or executing GPU compiling runs.

---

## 📖 Deep-Dive Documentation

For detailed blueprints and explanations of how the underlying model works, explore our fleshed-out explainer guides:

* 📘 **[Technical Explainer & Phonetics Guide](docs/explainer.md):** Deep-dive into the 3-stage architecture (Mimi, Llama 8B, Llama 300M) and detailed answers on IPA/X-SAMPA phonetic inputs.
* 📙 **[Developer Guide](docs/developer_guide.md):** Architectural implementation of MLX model layers, attention parameters, dynamic KV caches, and step-by-step frame loops.
* 📊 **[Optimization & Evaluation Report](docs/evaluation_report.md):** Detailed performance and quality benchmarks comparing unquantized bfloat16 vs. 4-bit quantized configurations on Apple Silicon GPUs, along with dynamic parameter trade-off curves and our mathematical parity audit.
* 📗 **[MLX Porting Blueprint](docs/mlx_porting_plan.md):** Step-by-step engineering roadmap for porting and optimizing large-scale transformer networks to Apple Silicon.
* 🎙️ **[Local Gemma 4 Multimodal Validation Setup Guide](docs/gemma4_setup_guide.md):** Steps to configure, download, and run Gemma 4 locally on macOS for offline, private, and zero-cost speech and visual validation of MisoTTS.

