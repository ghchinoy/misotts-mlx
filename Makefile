# ==============================================================================
# MisoTTS Apple Silicon MLX Port - Unified Workspace Makefile
# ==============================================================================
# This Makefile unifies the Python virtual environment (via uv), PyTorch baseline
# reference engines, audio evaluation comparative tools, and Apple Swift/SwiftUI
# executables. Run 'make' or 'make help' for a list of available targets.

# Default Target
.DEFAULT_GOAL := help

# Configurable variables
TEXT ?= "Hello from local Apple Silicon! MisoTTS is accelerating speech generation on my GPU."
SPEAKER ?= 0
REF_WAV ?= outputs/studio_output.wav # Default reference wav path
TARGET_WAV ?= outputs/studio_output_gpu.wav # Default target wav path

# Self-documenting help system
.PHONY: help
help: ## Display this help message containing all available targets
	@echo "=============================================================================="
	@echo "                   MisoTTS Apple Silicon Workspace Makefile"
	@echo "=============================================================================="
	@echo "Usage: make <target> [VARIABLES...]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (configurable):"
	@echo "  \033[33mTEXT\033[0m       Speech prompt text (default: 'Hello from local Apple Silicon...')"
	@echo "  \033[33mSPEAKER\033[0m    Active preset speaker ID 0-30 (default: 0)"
	@echo "  \033[33mREF_WAV\033[0m    Reference audio path for evaluation comparison"
	@echo "  \033[33mTARGET_WAV\033[0m Target audio path for evaluation comparison"
	@echo "=============================================================================="

# ------------------------------------------------------------------------------
# 1. Onboarding & Environment Preparation
# ------------------------------------------------------------------------------

.PHONY: setup
setup: ## Initialize Python virtualenv and install project dependencies with uv
	@echo "⚙️ Setting up Python environment via uv sync..."
	uv sync

# ------------------------------------------------------------------------------
# 2. Python Inference & Verification
# ------------------------------------------------------------------------------

.PHONY: dry-run
dry-run: ## Validate token shapes, model state, and pipeline paths in milliseconds without GPU compilation
	@echo "🔍 Running high-speed dry-run validation (AX Mode)..."
	uv run python miso_mlx/miso_mlx_cli.py --json speak --text "Dry-run validation" --mlx --dry-run

.PHONY: speak
speak: ## Synthesize high-fidelity speech locally on Apple Silicon GPUs (Metal-accelerated)
	@echo "⚡ Synthesizing speech using local Metal-accelerated MLX GPU..."
	@mkdir -p outputs
	uv run python miso_mlx/miso_mlx_cli.py speak --text $(TEXT) --speaker $(SPEAKER) --mlx --output outputs/studio_output_gpu.wav

.PHONY: speak-ref
speak-ref: ## Synthesize speech using PyTorch CPU baseline reference (for comparison validation)
	@echo "🕒 Synthesizing baseline speech using PyTorch CPU reference..."
	@mkdir -p outputs
	uv run python miso_mlx/miso_mlx_cli.py speak --text $(TEXT) --speaker $(SPEAKER) --output outputs/studio_output_cpu.wav

# ------------------------------------------------------------------------------
# 3. Audio Parity & Evaluation
# ------------------------------------------------------------------------------

.PHONY: compare-audio
compare-audio: ## Mathematically verify acoustic & alignment parity between GPU and reference WAV files
	@echo "📊 Comparing generated audio files mathematically for parity..."
	uv run python miso_mlx/compare_audio.py --ref $(REF_WAV) --target $(TARGET_WAV)

# ------------------------------------------------------------------------------
# 4. Swift Command-Line Validation App (miso_swift)
# ------------------------------------------------------------------------------

.PHONY: build-swift
build-swift: ## Compile the Swift-MLX Command-Line Weight-loading Verification utility
	@echo "🛠️ Compiling Swift-MLX command-line verification utility..."
	cd miso_swift && swift build

.PHONY: run-swift
run-swift: ## Compile and execute the Swift-MLX Weight-loading Verification utility
	@echo "🚀 Executing Swift-MLX verification utility..."
	cd miso_swift && swift run

# ------------------------------------------------------------------------------
# 5. SwiftUI Desktop Studio Application (miso_studio)
# ------------------------------------------------------------------------------

.PHONY: build-studio
build-studio: ## Compile the premium macOS SwiftUI 'MisoTTS Studio' application package
	@echo "🛠️ Compiling MisoTTS Studio macOS desktop application..."
	cd miso_studio && swift build

.PHONY: run-studio
run-studio: ## Compile and run the premium macOS SwiftUI 'MisoTTS Studio' application
	@echo "🚀 Launching MisoTTS Studio Desktop app natively..."
	cd miso_studio && swift run

.PHONY: bundle-studio
bundle-studio: ## Compile and package MisoTTS Studio as a native, standalone macOS .app bundle
	@chmod +x miso_mlx/bundle_studio.sh
	@./miso_mlx/bundle_studio.sh


# ------------------------------------------------------------------------------
# 6. Cleaning & Housekeeping
# ------------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove compiled binary targets, build caches, and python pycaches
	@echo "🧹 Cleaning Python __pycache__ folders..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	@echo "🧹 Cleaning miso_swift Package build directories..."
	cd miso_swift && swift package clean
	@echo "🧹 Cleaning miso_studio Package build directories..."
	cd miso_studio && swift package clean
	@echo "✨ Clean-up complete!"
