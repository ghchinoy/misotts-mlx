# MisoTTS Apple Silicon (MLX) Tooling & Explainer Suite

This repository contains tools, developer guides, and command-line utilities for running, quantizing, and evaluating the **MisoTTS 8B** model locally on macOS using Apple's native **MLX** GPU framework.

---

## Model Introduction & Capabilities

**MisoTTS 8B** is an **8.2 Billion parameter Text-to-Dialogue RVQ Transformer** optimized for high-fidelity English conversational speech synthesis and zero-shot voice cloning.

* **Contextual Conversational Flow:** Natively maintains segment history, ensuring stylistic consistency, emotion, and realistic turn-taking transitions.
* **Simple Speaker Conditioning:** Switch speakers using inline text prefixing (e.g., `[0] Hello there!`).
* **Zero-Shot Voice Cloning:** Synthesize speech in a target voice using a 3–10 second clean reference audio clip and its text transcript.
* **Implicit Content Protection:** Integrates Sony's `SilentCipher` 44.1 kHz acoustic watermarking by default.

Running an 8.2B parameter model on standard macOS CPU backends is resource-intensive. This suite ports the model components (the neural synthesizer, decoder stages, and attention layers) to Apple's native **Metal GPU (via MLX)**, delivering up to a **6.7x speedup** and enabling real-time local speech generation with reduced memory usage.


## Repository Layout

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
├── docs/                   # Technical Manuals
│   ├── explainer.md        # 3-Stage architecture & phonetics guide
│   ├── developer_guide.md  # Detailed MLX layer implementation guide
│   ├── evaluation_report.md # Performance benchmarks & dynamic parameter trials
│   ├── miso_mlx_friction_report.md # Hard-learned porting lessons & findings
│   └── mlx_porting_plan.md # 5-phase porting blueprint
├── outputs/                # Folder containing generated .wav audio outputs
└── sources/                # Original PyTorch reference models (preserved untouched)
```

---

## 🚀 Unified Workspace Dashboard (Makefile)

To streamline development across Python and Swift, this repository includes a top-level self-documenting `Makefile`. Instead of typing long command-line strings, you can use simple aliases to manage the entire workspace:

| Target | Description | Key Variable Overrides |
| :--- | :--- | :--- |
| `make setup` | Sync Python virtual environment (`uv sync`) | N/A |
| `make dry-run` | Rapid model path and token shape validation | N/A |
| `make speak` | Synthesize high-fidelity speech locally on GPU (MLX) | `TEXT="..." SPEAKER=0` |
| `make speak-ref` | Synthesize baseline speech on CPU (PyTorch) | `TEXT="..." SPEAKER=0` |
| `make compare-audio` | Mathematically compare target and reference WAVs | `REF_WAV="..." TARGET_WAV="..."` |
| `make build-swift` | Compile Swift-MLX load-verification CLI | N/A |
| `make run-swift` | Run Swift-MLX weight load validation | N/A |
| `make build-studio` | Build the premium macOS desktop SwiftUI app | N/A |
| `make run-studio` | Build and run the macOS desktop SwiftUI app | N/A |
| `make clean` | Remove python caches and clean both Swift packages | N/A |

### Example Usage:
```bash
# Display the interactive help menu of targets
make help

# Run a rapid validation in milliseconds
make dry-run

# Synthesize a custom sentence on GPU
make speak TEXT="Hello from my unified Makefile!" SPEAKER=2

# Compile and launch the native macOS SwiftUI Studio App
make run-studio
```

---

## End-to-End Walkthrough

Follow this workflow to configure, run, quantize, and evaluate the MisoTTS model on macOS.

### Prerequisites
Ensure you have the following installed on your system:
* **macOS** with Apple Silicon (M1, M2, M3, or M4 series chips)
* **Python 3.12** or newer
* **uv** (fast package manager for Python)
* **gcloud CLI** (optional, required if using the automated Vertex AI audio evaluator)


### Step 1: Environment Setup
Clone the repository and synchronize dependencies using `uv` to isolate compiled C++ and Metal backends within a virtual environment:

```bash
# Sync package locks and build virtual environment
uv sync
```

### Step 2: System Diagnostics
Before running weight-conversion tasks, execute the diagnostics command to scan your hardware specifications (GPU cores, memory bandwidth, unified memory allocation) and verify directories:

```bash
uv run python miso_mlx/miso_mlx_cli.py optimize
```

### Step 3: Download and Cache Weights
To avoid network bottlenecks, download and cache all required weights (Llama-3.2, Mimi codec, SilentCipher, and the MisoTTS backbone) to your local Hugging Face storage:

```bash
uv run python miso_mlx/miso_mlx_cli.py download
```
> [!NOTE]
> This command requires approximately **30–40 GB** of free disk space to store all checkpoints and model definitions.

### Step 4: Convert PyTorch Weights to MLX Format
The downloaded checkpoints are stored in PyTorch format. Run the weight translation utility to map, transpose, and serialize them into MLX `.safetensors` files compatible with Metal memory layouts:

```bash
uv run python miso_mlx/mlx_converter.py
```
This utility reads the PyTorch model keys, applies the mapping topology, and saves the translation blueprint to `miso_mlx/mlx_weights/pytorch_to_mlx_mapping.txt` for verification.

### Step 5: Text-to-Speech Synthesis (bfloat16 MLX GPU)
Synthesize your first audio file using the unquantized `bfloat16` model on your Mac's Metal GPU.

```bash
uv run python miso_mlx/miso_mlx_cli.py speak \
  --text "Hello! This is synthesized locally on my Mac using our unified GPU workspace." \
  --speaker 0 \
  --mlx \
  --output outputs/hello_unquantized.wav
```
*Running on the GPU via `--mlx` streams model weights using Apple Silicon unified memory, minimizing CPU overhead.*

### Step 6: 4-bit Model Quantization
The unquantized weights require 16.38 GB of RAM, which can limit streaming speed. You can compress the linear projection and transformer layers to 4-bit in-place (reducing weight size to 5.52 GB and halving unified memory usage) by adding the `--quant` flag:

```bash
uv run python miso_mlx/miso_mlx_cli.py speak \
  --text "Hello! This is synthesized locally on my Mac using our unified GPU workspace." \
  --speaker 0 \
  --mlx \
  --quant \
  --output outputs/hello_4bit_quant.wav
```
* **Performance Impact:** First-step JIT compilation drops from **6.28s to 0.53s (11.8x reduction)**, and the real-time generation factor (RTF) is reduced, resulting in up to **3.82x faster step inference**.

### Step 7: Zero-Shot Voice Cloning
Clone a target voice by supplying a short (3–10s) audio reference file and its transcription. The autoregressive decoder uses the reference acoustic codes to synthesize the new text with the target speaker's timbre:

```bash
uv run python miso_mlx/miso_mlx_cli.py clone \
  --text "This sentence is spoken in my newly cloned voice profile!" \
  --prompt-audio "outputs/hello_unquantized.wav" \
  --prompt-text "Hello! This is synthesized locally on my Mac using our unified GPU workspace." \
  --output outputs/cloned_output.wav
```

### Step 8: Dynamic Parameter Scheduling
Quantized models can sometimes drift, leading to premature cut-offs at low temperatures or sibilant feedback at high temperatures. You can avoid these issues and bypass the SilentCipher watermark by applying dynamic temperature decay and Classifier-Free Guidance (CFG):

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
*   `--no-watermark`: Removes the SilentCipher watermark, which can eliminate background noise or artifacts.
*   `--temp-start 0.7 --temp-min 0.4 --temp-decay-steps 30`: Decays the sampling temperature over 30 steps to prevent accumulation of sibilant noise.
*   `--cfg-scale 2.0`: Increases the text-guidance conditioning scale, preventing the model from generating silence loops.


### ⚠️ Model Generation Limits & Reading Speed Heuristic

MisoTTS is an autoregressive transformer that generates audio frame-by-frame (with a frame rate of 12.5 steps per second). To prevent runaway generation and manage GPU/memory overhead, the local Python CLI and generator enforce a default safety limit of **10,000 milliseconds (10 seconds)** via the `--max_length_ms` parameter.

* **How Truncation Happens**: The model generates speech until it naturally predicts a blank/zero **End of Sequence (EOS)** token indicating it has finished reading the text, or until it hits the hard `--max_length_ms` limit. If the model is still mid-word when it hits the limit, the loop is forcibly terminated, cutting off the speech abruptly.
* **The Reading Speed Heuristic**: Standard English conversational speech averages **130 to 150 words per minute** (roughly **2 to 2.5 words per second**).
  * Use this formula to estimate the minimum required limit: 
    $$\text{Estimated Duration (seconds)} \approx \frac{\text{Word Count}}{2.2} + 2.0 \text{ (padding)}$$
  * **Example (15 words)**: $\approx (15 / 2.2) + 2 = 8.8$ seconds. The default 10,000 ms (10s) limit is perfectly sufficient.
  * **Example (100 words)**: $\approx (100 / 2.2) + 2 = 47.4$ seconds. You must increase `--max_length_ms` to `50000` (50 seconds) or more to avoid cutoffs.


### Step 9: Mathematical Parity Verification
Compare your generated outputs against a PyTorch CPU baseline to measure spectral divergence and phonetic alignment:

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

### Step 10: Audio Quality and Accuracy Validation

You can programmatically transcribe and assess your synthesized audio outputs using either **cloud-based Gemini validation** or **offline local Gemma 4 validation**.


#### Option A: Cloud-Based Auditing (Gemini 3.1 Flash Lite)

This option uses the Google GenAI SDK on Vertex AI with the **Gemini 3.1 Flash Lite** model to transcribe the audio and evaluate alignment, clarity, and prosody:

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

#### Option B: Offline Auditing (Local Gemma 4 MLX GPU)

To run evaluations offline without external API dependencies, you can execute multimodal assessments locally on your Mac's GPU using the **Gemma 4** model via `mlx-vlm`.

Run the custom local validation suite:
```bash
# Execute local vision + speech joint validation
/Users/ghchinoy/projects/gemmma/.venv/bin/python miso_mlx/test_multimodal_validation.py
```

This script runs joint interleaved (vision and speech) analyses as well as audio transcription.

> [!NOTE]
> For instructions on setting up local Python environments, downloading model weights, and formatting interleaved multimodal prompts for Gemma 4, see the [Local Gemma 4 Multimodal Validation Setup Guide](docs/gemma4_setup_guide.md).

---

## Native Swift-MLX Integration & Validation

The `miso_swift/` directory contains a minimal, standalone Swift package (`MisoSwiftDemo`) demonstrating how to leverage `mlx-swift` to natively load model weights and execute computations on Apple Silicon GPUs (Metal).

### Key Technical Insights
* **Explicit Array Typing:** In Swift-MLX, `matmul` operations require floating-point matrices. Because Swift overloads can resolve untyped floating-point literals to integers or doubles, constructing arrays without explicit casts can trigger runtime `[matmul]` type mismatch exceptions. 
* **Type-Safe Solution:** We have explicitly cast arrays as `[Float]` in `miso_swift/Sources/main.swift` to guarantee `float32` compilation on the GPU backend:
  ```swift
  let a = MLXArray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] as [Float], [2, 4])
  ```

### Running the Swift Utility

#### Option A: Running via Xcode (Recommended)
1. Open the folder `miso_swift` or its `Package.swift` file directly in **Xcode**.
2. Xcode will automatically download and resolve the `mlx-swift` dependency (configured for version `0.10.0+`).
3. Select the `MisoSwiftDemo` executable target.
4. Click **Product -> Run** (`Cmd + R`) to execute the validation utility.

> [!WARNING]
> **Resolving Cached MatMul Type Crashes:** If you previously built the project and encounter the error `Fatal error: [matmul] Only inexact types are supported but int32 and int32 were provided`, Xcode is running an outdated, cached build. To resolve this, clean Xcode's cache by selecting **Product -> Clean Build Folder** (`Cmd + Shift + K`), and then re-run (`Cmd + R`).

#### Option B: Running via Swift PM Command Line
To build and run the utility natively via the terminal:
```bash
cd miso_swift
swift run
```
*(Note: Terminal execution may emit warnings if macOS sandboxing prevents loading the default `metallib` shader library outside of Xcode.)*

---



## CLI Global Diagnostic Features

### Headless and Automated Operation
The command-line tools support environment variables to ensure deterministic, silent execution inside scripts, CI/CD pipelines, and automated coding environments:
* **JSON Output (`--json`):** Redirects standard warning logs to `stderr` and prints machine-readable JSON blocks to `stdout`.
* **No TUI / No Color (`NO_COLOR=1` or `MISO_NO_TUI=1`):** Disables ANSI terminal escaping to produce clean logging streams.
* **Mutative Safety (`--dry-run`):** Performs syntax verification, resolves Hugging Face tokenizers, and verifies weights in milliseconds, without loading the full 16 GB model or starting GPU compilation.


## Agent Skills (agentskills.io)

This repository includes custom, single-purpose **Agent Skills** conforming to the [agentskills.io](https://agentskills.io/specification.md) format. These files enable autonomous AI agents to easily discover and execute model migration, audio generation, and evaluation workflows:

* **[mlx-model-porting](skills/mlx-model-porting/SKILL.md):** Port, migrate, and validate PyTorch transformer weights, attention projections, and dynamic KV caches to native MLX.
* **[misotts-speech-generation](skills/misotts-speech-generation/SKILL.md):** Run local bfloat16 or 4-bit quantized text-to-speech synthesis and zero-shot voice cloning.
* **[audio-quality-evaluation](skills/audio-quality-evaluation/SKILL.md):** Audit synthesized WAV files using mathematical spectral comparison or multimodal Gemini/Gemma 4 models.

---

## Deep-Dive Documentation

For detailed explanations of the model architecture, see the documentation below:

* **[Technical Explainer & Phonetics Guide](docs/explainer.md):** Analysis of the 3-stage architecture (Mimi, Llama 8B, Llama 300M) and documentation on IPA/X-SAMPA phonetic inputs.
* **[Developer Guide](docs/developer_guide.md):** Implementation details of MLX model layers, attention parameters, dynamic KV caches, and step-by-step frame generation loops.
* **[Optimization & Evaluation Report](docs/evaluation_report.md):** Performance and quality benchmarks comparing unquantized bfloat16 vs. 4-bit quantized configurations on Apple Silicon GPUs, along with parameter trade-off curves and the mathematical parity audit.
* **[MLX Porting Blueprint](docs/mlx_porting_plan.md):** Roadmap for porting and optimizing large-scale transformer networks to Apple Silicon.
* **[Local Gemma 4 Multimodal Validation Setup Guide](docs/gemma4_setup_guide.md):** Instructions to configure, download, and run Gemma 4 locally on macOS for offline speech and visual validation of MisoTTS.

