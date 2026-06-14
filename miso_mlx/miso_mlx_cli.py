#!/usr/bin/env python
import argparse
import os
import sys
import json
from pathlib import Path

# Resolve pathing so we can load generator and models from original sources
original_sources_path = str(Path(__file__).parent.parent / "sources" / "MisoTTS")
if original_sources_path not in sys.path:
    sys.path.append(original_sources_path)

# Set HF hub timeouts
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["NO_TORCH_COMPILE"] = "1"

# AX/DX Override Checks
disable_color = "NO_COLOR" in os.environ or os.environ.get("MISO_NO_TUI") == "1"
is_json_mode = False  # Set dynamically in main()

# ANSI Terminal Styling Constants
BOLD = "" if disable_color else "\033[1m"
RESET = "" if disable_color else "\033[0m"

# Tufte-style Functional Semantic Color Tokens (Agent-Aware guidelines)
COLOR_ACCENT = "" if disable_color else "\033[34m"   # Blue for Landmarks & Headers
COLOR_PASS = "" if disable_color else "\033[32m"     # Green for Success State
COLOR_WARN = "" if disable_color else "\033[33m"     # Yellow/Orange for Pending/Warnings
COLOR_FAIL = "" if disable_color else "\033[31m"     # Red for Errors
COLOR_COMMAND = "" if disable_color else "\033[90m"  # Grey/Light Grey for Copy-Paste Command hints
COLOR_ID = "" if disable_color else "\033[36m"       # Teal/Mint for Unique Identifiers

def print_log(msg: str, style_prefix: str = "", is_error: bool = False):
    """
    Prints styled log messages. If running in JSON mode, routes logs to stderr
    so stdout remains clean for parsing.
    """
    if is_json_mode:
        print(f"{style_prefix}{msg}{RESET}", file=sys.stderr, flush=True)
    else:
        file = sys.stderr if is_error else sys.stdout
        print(f"{style_prefix}{msg}{RESET}", file=file, flush=True)

def print_header(title: str):
    print_log(f"=== {title} ===", f"\n{BOLD}{COLOR_ACCENT}")

def print_success(msg: str):
    print_log(f"✔ {msg}", f"{BOLD}{COLOR_PASS}")

def print_info(msg: str):
    print_log(f"ℹ {msg}", f"{COLOR_ACCENT}")

def print_warning(msg: str):
    print_log(f"⚠ {msg}", f"{COLOR_WARN}")

def print_error(msg: str):
    print_log(f"✘ {msg}", f"{BOLD}{COLOR_FAIL}", is_error=True)

# Check dependencies and import PyTorch backends
try:
    import torch
    import torchaudio
except ImportError:
    print_error("PyTorch or Torchaudio not found. Please run 'uv sync' first.")
    sys.exit(1)

def check_mlx_dependencies() -> bool:
    """Checks if MLX and moshi_mlx packages are installed."""
    try:
        import mlx.core as mx
        import moshi_mlx
        import rustymimi
        return True
    except ImportError:
        return False

def handle_optimize(args):
    print_header("MisoTTS MLX Optimizer Tool")
    print_info("Scanning system hardware and software capabilities...")

    # Hardware detection
    import platform
    is_mac = platform.system() == "Darwin"
    is_arm = platform.machine() == "arm64"
    apple_silicon = is_mac and is_arm

    if apple_silicon:
        print_success("Apple Silicon hardware detected (Mac ARM64).")
    else:
        print_warning(f"System is running on {platform.system()} ({platform.machine()}). MLX is only supported on Apple Silicon Macs.")

    print_info("Checking for MLX python dependencies...")
    has_mlx = check_mlx_dependencies()

    if has_mlx:
        print_success("All MLX & Apple Silicon dependencies are installed (mlx, moshi_mlx, rustymimi).")
    else:
        print_warning("MLX dependencies are missing.")
        print_log("\nTo install MLX dependencies, run:", COLOR_WARN)
        print_log("  pip install mlx moshi_mlx rustymimi", COLOR_COMMAND)

    print_info("Locating MisoTTS model checkpoint...")
    from generator import DEFAULT_MISO_TTS_REPO_ID
    model_source = os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID)
    print_info(f"Target model repo ID: {model_source}")

    print_info("Preparing MLX weights directory...")
    mlx_weights_dir = Path(__file__).parent / "mlx_weights"
    mlx_weights_dir.mkdir(exist_ok=True)
    print_success(f"MLX weight export directory initialized: {mlx_weights_dir.resolve()}")

    if is_json_mode:
        results = {
            "status": "success",
            "hardware": {
                "system": platform.system(),
                "machine": platform.machine(),
                "apple_silicon": apple_silicon
            },
            "dependencies": {
                "mlx": has_mlx,
                "moshi_mlx": has_mlx,
                "rustymimi": has_mlx
            },
            "model_source": model_source,
            "mlx_weights_dir": str(mlx_weights_dir.resolve())
        }
        print(json.dumps(results, indent=2))
        return

    print("\n" + "=" * 50)
    print_success("Optimization Preparation Completed!")
    print("=" * 50)
    print(f"1. Refer to {BOLD}docs/mlx_porting_plan.md{RESET} for weight mapping specs.")
    print("2. Once MLX translation is finalized, use the --mlx flag on 'speak' or 'clone' to run at GPU speeds.")
    print("=" * 50)

def handle_download(args):
    print_header("Downloading MisoTTS Weights & Assets")
    from generator import DEFAULT_MISO_TTS_REPO_ID, load_miso_8b
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    model_source = os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID)
    print_info(f"Downloading main MisoTTS weights from: https://huggingface.co/{model_source}")
    
    try:
        # Download tokenizer & models to cache
        print_info("Downloading Llama 3.2 tokenizer assets...")
        from transformers import AutoTokenizer
        try:
            AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
        except Exception:
            AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        
        print_info("Downloading Mimi codec model weights...")
        hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)

        print_info("Downloading SilentCipher watermarker weights...")
        from watermarking import load_watermarker
        # Try loading watermarker on CPU to trigger download
        load_watermarker(device="cpu")

        print_info("Downloading main MisoTTS checkpoint...")
        load_miso_8b(device="cpu", model_path_or_repo_id=model_source)

        print_success("All model weights and tokenizers downloaded and cached successfully!")
        
        if is_json_mode:
            print(json.dumps({
                "status": "success",
                "message": "All weights and assets cached successfully."
            }, indent=2))
    except Exception as e:
        print_error(f"Failed to complete download: {e}")
        if is_json_mode:
            print(json.dumps({"status": "error", "reason": str(e)}), file=sys.stderr)
        sys.exit(1)

def handle_speak(args):
    print_header("MisoTTS Speak: Text-to-Speech Generation")
    
    # Check for Converted and Quantized MLX weight files
    mlx_weights_file = Path(__file__).parent / "mlx_weights" / "model.safetensors"
    quantized_weights_file = Path(__file__).parent / "mlx_weights" / "quantized_model_4bit.safetensors"
    has_mlx_weights = mlx_weights_file.exists()
    has_quantized_weights = quantized_weights_file.exists()

    # Determine if we should use 4-bit quantization
    use_quant = False
    if args.mlx:
        if getattr(args, "quant", False):
            use_quant = True
        elif has_quantized_weights and not has_mlx_weights:
            use_quant = True

    if args.dry_run:
        print_info("[DRY-RUN] Initiating Text-to-Speech input validation...")
        print_info(f"  Target Text: \"{args.text}\"")
        print_info(f"  Speaker ID:  {args.speaker}")
        print_info(f"  Max Length:  {args.max_length_ms}ms")
        print_info(f"  Backend:     {'MLX (GPU)' if args.mlx else 'PyTorch (CPU)'}")
        print_info(f"  Temperature: {args.temperature}")
        print_info(f"  Top-K:       {args.topk}")
        print_info(f"  CFG Scale:   {args.cfg_scale}")
        if args.temp_start is not None:
            print_info(f"  Temp Start:  {args.temp_start}")
        if args.temp_min is not None:
            print_info(f"  Temp Min:    {args.temp_min}")
        if args.temp_decay_steps is not None:
            print_info(f"  Decay Steps: {args.temp_decay_steps}")
        if args.mlx:
            print_info(f"  Quantization:{' Enabled (4-bit)' if use_quant else ' Disabled'}")
        
        # Verify text tokenization length
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
        
        formatted_text = f"[{args.speaker}] {args.text.lstrip()}"
        text_tokens = tokenizer.encode(formatted_text)
        print_success(f"Text tokenized successfully (Sequence Length: {len(text_tokens)} tokens).")
        
        if args.mlx:
            if use_quant:
                print_info(f"MLX Quantized Weight File Presence: {has_quantized_weights} ({quantized_weights_file})")
            else:
                print_info(f"MLX Converted Weight File Presence: {has_mlx_weights} ({mlx_weights_file})")
            
        print_success("[DRY-RUN] Parameter verification complete! Pipeline is fully valid.")
        
        if is_json_mode:
            print(json.dumps({
                "status": "dry_run_success",
                "text": args.text,
                "speaker": args.speaker,
                "token_count": len(text_tokens),
                "backend": "mlx" if args.mlx else "pytorch",
                "mlx_weights_found": has_mlx_weights,
                "quantized_weights_found": has_quantized_weights,
                "quantization_enabled": use_quant,
                "temperature": args.temperature,
                "topk": args.topk,
                "temp_start": args.temp_start,
                "temp_min": args.temp_min,
                "temp_decay_steps": args.temp_decay_steps,
                "cfg_scale": args.cfg_scale
            }, indent=2))
        return

    if args.mlx:
        if not check_mlx_dependencies():
            print_warning("MLX backend requested, but MLX dependencies are not installed. Falling back to PyTorch CPU backend.")
            args.mlx = False
        else:
            print_info("Using MLX GPU Backend (Metal-accelerated)...")
            try:
                # Add current directory to path to ensure local imports succeed
                if str(Path(__file__).parent) not in sys.path:
                    sys.path.append(str(Path(__file__).parent))
                
                from mlx_model import ModelArgs, MisoTTSModel
                from mlx_generator import MLXGenerator
                import mlx.core as mx
                import mlx.nn as nn
                
                print_info("Initializing MLX model layers on GPU...")
                args_model = ModelArgs()
                model = MisoTTSModel(args_model)
                
                if use_quant:
                    print_info("Applying 4-bit quantization to model architecture...")
                    nn.quantize(
                        model,
                        group_size=64,
                        bits=4,
                        class_predicate=lambda k, m: isinstance(m, nn.Linear)
                    )
                    if has_quantized_weights:
                        print_info(f"Loading quantized 4-bit MLX weights from: {quantized_weights_file.resolve()}")
                        model.load_weights(str(quantized_weights_file), strict=False)
                        print_success("Quantized 4-bit weights loaded successfully!")
                    else:
                        print_warning(f"Quantized 4-bit MLX weights not found at: {quantized_weights_file.resolve()}")
                        print_log("  Please run 'uv run python miso_mlx/mlx_quantizer.py' to quantize base weights.", COLOR_COMMAND)
                        print_info("Initializing with random parameters for quantized MLX pipeline execution test...")
                else:
                    if has_mlx_weights:
                        print_info(f"Loading converted MLX weights from: {mlx_weights_file.resolve()}")
                        model.load_weights(str(mlx_weights_file), strict=False)
                        print_success("Converted weights loaded successfully!")
                    else:
                        print_warning(f"Converted MLX weights not found at: {mlx_weights_file.resolve()}")
                        print_log("  Please run 'uv run python miso_mlx/mlx_converter.py' to convert downloaded weights.", COLOR_COMMAND)
                        print_info("Initializing with random parameters for MLX pipeline execution test...")
                print_info("Initializing MLX Generator...")
                generator = MLXGenerator(model)
                
                print_info(f"Synthesizing text via MLX: \"{args.text}\" (Speaker: {args.speaker})")
                
                import time
                import resource
                
                def get_peak_memory_mb() -> float:
                    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    if sys.platform == "darwin":
                        return usage / (1024 * 1024)
                    else:
                        return usage / 1024

                start_time = time.perf_counter()
                mem_before = get_peak_memory_mb()
                
                audio_array = generator.generate(
                    text=args.text,
                    speaker=args.speaker,
                    context=[],
                    max_audio_length_ms=args.max_length_ms,
                    temperature=args.temperature,
                    topk=args.topk,
                    no_watermark=args.no_watermark,
                    temp_start=args.temp_start,
                    temp_min=args.temp_min,
                    temp_decay_steps=args.temp_decay_steps,
                    cfg_scale=args.cfg_scale,
                )

                
                end_time = time.perf_counter()
                mem_after = get_peak_memory_mb()
                
                duration_sec = end_time - start_time
                audio_len_samples = len(audio_array)
                sample_rate = generator.sample_rate
                audio_len_sec = audio_len_samples / sample_rate
                total_steps = int(audio_len_samples / 1920)
                rtf = duration_sec / audio_len_sec if audio_len_sec > 0 else 0.0
                steps_per_sec = total_steps / duration_sec if duration_sec > 0 else 0.0
                mem_increase = mem_after - mem_before
                
                # Convert MLX array to PyTorch tensor for saving with torchaudio
                import numpy as np
                audio_tensor = torch.from_numpy(np.array(audio_array)).unsqueeze(0)
                
                # Peak normalize the audio to [-0.95, 0.95] to prevent clipping/silencing on macOS
                max_val = torch.max(torch.abs(audio_tensor))
                if max_val > 0:
                    audio_tensor = (audio_tensor / max_val) * 0.95
                audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
                
                output_path = Path(args.output)
                torchaudio.save(str(output_path), audio_tensor, generator.sample_rate)
                print_success(f"Audio synthesized successfully on MLX GPU! Saved to: {output_path.resolve()}")
                
                stats = generator.last_run_stats
                print_header("Generation Statistics")
                print_info(f"  Wall Time (including compilation/warmup): {duration_sec:.2f} seconds")
                print_info(f"  Generated Audio Duration:                  {audio_len_sec:.2f} seconds")
                print_info(f"  Real-Time Factor (RTF):                   {rtf:.2f}x ({'Faster than real-time' if rtf < 1.0 else 'Slower than real-time'})")
                print_info(f"  Total Steps (Autoregressive Frames):     {total_steps}")
                print_info(f"  Generation Speed:                         {steps_per_sec:.2f} frames/sec")
                print_info(f"  Process Peak Memory (RSS):                {mem_after:.2f} MB")
                print_info(f"  Memory Allocated During Synthesis:        {max(0.0, mem_increase):.2f} MB")
                
                print_header("MLX Profiling Breakdown")
                print_info(f"  Prompt Tokenization Time:                  {stats.get('prompt_tokenization_time', 0.0):.2f} seconds")
                print_info(f"  Model Warmup Pass JIT Compilation Time:    {stats.get('warmup_time', 0.0):.2f} seconds")
                print_info(f"  Total Autoregressive Loop Time:            {stats.get('generation_loop_time', 0.0):.2f} seconds")
                print_info(f"    - First Step JIT Compilation Time:       {stats.get('first_step_time', 0.0):.2f} seconds")
                print_info(f"    - Subsequent Steps Running Time:         {stats.get('subsequent_steps_time', 0.0):.2f} seconds")
                print_info(f"    - Average Running Time per Step:         {stats.get('avg_subsequent_step_time', 0.0) * 1000.0:.1f} milliseconds")
                print_info(f"  Mimi Audio Codec Decoding Time:            {stats.get('mimi_decoding_time', 0.0):.2f} seconds")
                print_info(f"  SilentCipher Watermarking Time:            {stats.get('watermarking_time', 0.0):.2f} seconds")
                
                if is_json_mode:
                    print(json.dumps({
                        "status": "success",
                        "backend": "mlx",
                        "quantization_enabled": use_quant,
                        "text": args.text,
                        "speaker": args.speaker,
                        "output": str(output_path.resolve()),
                        "sample_rate": generator.sample_rate,
                        "stats": {
                            "generation_time_sec": duration_sec,
                            "audio_duration_sec": audio_len_sec,
                            "real_time_factor": rtf,
                            "total_steps": total_steps,
                            "steps_per_second": steps_per_sec,
                            "peak_memory_mb": mem_after,
                            "memory_increase_mb": max(0.0, mem_increase),
                            "profiling": stats
                        }
                    }, indent=2))
                return
            except Exception as e:
                print_error(f"MLX Generator failed: {e}")
                print_info("Falling back to PyTorch CPU backend...")
                args.mlx = False

    if not args.mlx:
        print_info("Using PyTorch Backend...")
        if torch.cuda.is_available():
            device = "cuda"
            print_success("Running on CUDA GPU.")
        else:
            device = "cpu"
            print_warning("Running on CPU. Since this is an 8B parameter model, generation will be SLOW.")
 
        print_info(f"Loading MisoTTS generator (Device: {device})...")
        from generator import load_miso_8b
        try:
            generator = load_miso_8b(device=device)
        except Exception as e:
            print_error(f"Failed to load model: {e}")
            sys.exit(1)
 
        print_info(f"Synthesizing text: \"{args.text}\" (Speaker: {args.speaker})")
        try:
            import time
            import resource
            
            def get_peak_memory_mb() -> float:
                usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform == "darwin":
                    return usage / (1024 * 1024)
                else:
                    return usage / 1024
            
            start_time = time.perf_counter()
            mem_before = get_peak_memory_mb()
            
            audio = generator.generate(
                text=args.text,
                speaker=args.speaker,
                context=[],
                max_audio_length_ms=args.max_length_ms,
            )
            
            end_time = time.perf_counter()
            mem_after = get_peak_memory_mb()
            
            duration_sec = end_time - start_time
            audio_len_samples = len(audio)
            sample_rate = generator.sample_rate
            audio_len_sec = audio_len_samples / sample_rate
            total_steps = int(audio_len_samples / 1920)
            rtf = duration_sec / audio_len_sec if audio_len_sec > 0 else 0.0
            steps_per_sec = total_steps / duration_sec if duration_sec > 0 else 0.0
            mem_increase = mem_after - mem_before
            
            output_path = Path(args.output)
            torchaudio.save(str(output_path), audio.unsqueeze(0).cpu(), generator.sample_rate)
            print_success(f"Audio synthesized successfully! Saved to: {output_path.resolve()}")
            
            print_header("Generation Statistics")
            print_info(f"  Wall Time (including compilation/warmup): {duration_sec:.2f} seconds")
            print_info(f"  Generated Audio Duration:                  {audio_len_sec:.2f} seconds")
            print_info(f"  Real-Time Factor (RTF):                   {rtf:.2f}x ({'Faster than real-time' if rtf < 1.0 else 'Slower than real-time'})")
            print_info(f"  Total Steps (Autoregressive Frames):     {total_steps}")
            print_info(f"  Generation Speed:                         {steps_per_sec:.2f} frames/sec")
            print_info(f"  Process Peak Memory (RSS):                {mem_after:.2f} MB")
            print_info(f"  Memory Allocated During Synthesis:        {max(0.0, mem_increase):.2f} MB")
            
            if is_json_mode:
                print(json.dumps({
                    "status": "success",
                    "backend": "pytorch",
                    "text": args.text,
                    "speaker": args.speaker,
                    "output": str(output_path.resolve()),
                    "sample_rate": generator.sample_rate,
                    "stats": {
                        "generation_time_sec": duration_sec,
                        "audio_duration_sec": audio_len_sec,
                        "real_time_factor": rtf,
                        "total_steps": total_steps,
                        "steps_per_second": steps_per_sec,
                        "peak_memory_mb": mem_after,
                        "memory_increase_mb": max(0.0, mem_increase)
                    }
                }, indent=2))
        except Exception as e:
            print_error(f"Synthesis failed: {e}")
            sys.exit(1)

def handle_clone(args):
    print_header("MisoTTS Clone: Voice Cloning / Prompted Generation")
    
    prompt_audio_path = Path(args.prompt_audio)
    
    # Check for Converted and Quantized MLX weight files
    mlx_weights_file = Path(__file__).parent / "mlx_weights" / "model.safetensors"
    quantized_weights_file = Path(__file__).parent / "mlx_weights" / "quantized_model_4bit.safetensors"
    has_mlx_weights = mlx_weights_file.exists()
    has_quantized_weights = quantized_weights_file.exists()

    # Determine if we should use 4-bit quantization
    use_quant = False
    if args.mlx:
        if getattr(args, "quant", False):
            use_quant = True
        elif has_quantized_weights and not has_mlx_weights:
            use_quant = True

    if args.dry_run:
        print_info("[DRY-RUN] Initiating Voice Cloning input validation...")
        print_info(f"  Target Text:      \"{args.text}\"")
        print_info(f"  Prompt Audio Path: {prompt_audio_path}")
        print_info(f"  Prompt Transcript: \"{args.prompt_text}\"")
        print_info(f"  Speaker ID:        {args.speaker}")
        print_info(f"  Max Length:        {args.max_length_ms}ms")
        print_info(f"  Backend:           {'MLX (GPU)' if args.mlx else 'PyTorch (CPU)'}")
        print_info(f"  Temperature:       {args.temperature}")
        print_info(f"  Top-K:             {args.topk}")
        print_info(f"  CFG Scale:         {args.cfg_scale}")
        if args.temp_start is not None:
            print_info(f"  Temp Start:        {args.temp_start}")
        if args.temp_min is not None:
            print_info(f"  Temp Min:          {args.temp_min}")
        if args.temp_decay_steps is not None:
            print_info(f"  Decay Steps:       {args.temp_decay_steps}")
        if args.mlx:
            print_info(f"  Quantization:{' Enabled (4-bit)' if use_quant else ' Disabled'}")
        
        # Validate reference audio presence
        if not prompt_audio_path.exists():
            print_error(f"Reference voice prompt audio not found: {prompt_audio_path}")
            sys.exit(1)
        
        # Load audio shapes
        metadata = torchaudio.info(str(prompt_audio_path))
        print_success(f"Prompt audio structure valid ({metadata.num_channels} channels, {metadata.num_frames} frames, {metadata.sample_rate}Hz).")
        
        # Verify text tokenization
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
        
        formatted_prompt = f"[{args.speaker}] {args.prompt_text.lstrip()}"
        formatted_text = f"[{args.speaker}] {args.text.lstrip()}"
        prompt_tokens_len = len(tokenizer.encode(formatted_prompt))
        text_tokens_len = len(tokenizer.encode(formatted_text))
        
        print_success(f"Prompt text tokenized ({prompt_tokens_len} tokens). Target text tokenized ({text_tokens_len} tokens).")
        
        if args.mlx:
            if use_quant:
                print_info(f"MLX Quantized Weight File Presence: {has_quantized_weights} ({quantized_weights_file})")
            else:
                print_info(f"MLX Converted Weight File Presence: {has_mlx_weights} ({mlx_weights_file})")
            
        print_success("[DRY-RUN] Parameter and files verification complete! Pipeline is fully valid.")
        
        if is_json_mode:
            print(json.dumps({
                "status": "dry_run_success",
                "text": args.text,
                "prompt_audio": str(prompt_audio_path.resolve()),
                "prompt_text": args.prompt_text,
                "speaker": args.speaker,
                "prompt_token_count": prompt_tokens_len,
                "target_token_count": text_tokens_len,
                "backend": "mlx" if args.mlx else "pytorch",
                "mlx_weights_found": has_mlx_weights,
                "quantized_weights_found": has_quantized_weights,
                "quantization_enabled": use_quant,
                "temperature": args.temperature,
                "topk": args.topk,
                "temp_start": args.temp_start,
                "temp_min": args.temp_min,
                "temp_decay_steps": args.temp_decay_steps,
                "cfg_scale": args.cfg_scale
            }, indent=2))
        return

    if args.mlx:
        if not check_mlx_dependencies():
            print_warning("MLX backend requested, but MLX dependencies are not installed. Falling back to PyTorch CPU backend.")
            args.mlx = False
        else:
            print_info("Using MLX GPU Backend (Metal-accelerated)...")
            try:
                if str(Path(__file__).parent) not in sys.path:
                    sys.path.append(str(Path(__file__).parent))
                
                from mlx_model import ModelArgs, MisoTTSModel
                from mlx_generator import MLXGenerator, Segment as MLXSegment
                import mlx.core as mx
                import mlx.nn as nn
                
                print_info("Initializing MLX model layers on GPU...")
                args_model = ModelArgs()
                model = MisoTTSModel(args_model)
                
                if use_quant:
                    print_info("Applying 4-bit quantization to model architecture...")
                    nn.quantize(
                        model,
                        group_size=64,
                        bits=4,
                        class_predicate=lambda k, m: isinstance(m, nn.Linear)
                    )
                    if has_quantized_weights:
                        print_info(f"Loading quantized 4-bit MLX weights from: {quantized_weights_file.resolve()}")
                        model.load_weights(str(quantized_weights_file), strict=False)
                        print_success("Quantized 4-bit weights loaded successfully!")
                    else:
                        print_warning(f"Quantized 4-bit MLX weights not found at: {quantized_weights_file.resolve()}")
                        print_log("  Please run 'uv run python miso_mlx/mlx_quantizer.py' to quantize base weights.", COLOR_COMMAND)
                        print_info("Initializing with random parameters for quantized MLX pipeline execution test...")
                else:
                    if has_mlx_weights:
                        print_info(f"Loading converted MLX weights from: {mlx_weights_file.resolve()}")
                        model.load_weights(str(mlx_weights_file), strict=False)
                        print_success("Converted weights loaded successfully!")
                    else:
                        print_warning(f"Converted MLX weights not found at: {mlx_weights_file.resolve()}")
                        print_log("  Please run 'uv run python miso_mlx/mlx_converter.py' to convert downloaded weights.", COLOR_COMMAND)
                        print_info("Initializing with random parameters for MLX pipeline execution test...")
                
                print_info("Initializing MLX Generator...")
                generator = MLXGenerator(model)
                
                if not prompt_audio_path.exists():
                    print_error(f"Prompt audio file not found: {prompt_audio_path}")
                    sys.exit(1)

                print_info(f"Loading reference voice prompt audio from: {prompt_audio_path}")
                prompt_audio_pt, sample_rate = torchaudio.load(str(prompt_audio_path))
                prompt_audio_pt = torchaudio.functional.resample(
                    prompt_audio_pt.squeeze(0),
                    orig_freq=sample_rate,
                    new_freq=generator.sample_rate,
                )
                
                # Convert PyTorch tensor to MLX array
                prompt_audio_mx = mx.array(prompt_audio_pt.numpy())
                
                print_info("Assembling voice-prompt conditioning context in MLX...")
                context = [
                    MLXSegment(
                        speaker=args.speaker,
                        text=args.prompt_text,
                        audio=prompt_audio_mx,
                    )
                ]
                
                import time
                import resource
                
                def get_peak_memory_mb() -> float:
                    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    if sys.platform == "darwin":
                        return usage / (1024 * 1024)
                    else:
                        return usage / 1024

                start_time = time.perf_counter()
                mem_before = get_peak_memory_mb()
                
                print_info(f"Synthesizing new text in cloned voice via MLX: \"{args.text}\"")
                audio_array = generator.generate(
                    text=args.text,
                    speaker=args.speaker,
                    context=context,
                    max_audio_length_ms=args.max_length_ms,
                    temperature=args.temperature,
                    topk=args.topk,
                    no_watermark=args.no_watermark,
                    temp_start=args.temp_start,
                    temp_min=args.temp_min,
                    temp_decay_steps=args.temp_decay_steps,
                    cfg_scale=args.cfg_scale,
                )

                
                end_time = time.perf_counter()
                mem_after = get_peak_memory_mb()
                
                duration_sec = end_time - start_time
                audio_len_samples = len(audio_array)
                sample_rate = generator.sample_rate
                audio_len_sec = audio_len_samples / sample_rate
                total_steps = int(audio_len_samples / 1920)
                rtf = duration_sec / audio_len_sec if audio_len_sec > 0 else 0.0
                steps_per_sec = total_steps / duration_sec if duration_sec > 0 else 0.0
                mem_increase = mem_after - mem_before
                
                # Convert MLX array to PyTorch tensor for saving with torchaudio
                import numpy as np
                audio_tensor = torch.from_numpy(np.array(audio_array)).unsqueeze(0)
                
                # Peak normalize the audio to [-0.95, 0.95] to prevent clipping/silencing on macOS
                max_val = torch.max(torch.abs(audio_tensor))
                if max_val > 0:
                    audio_tensor = (audio_tensor / max_val) * 0.95
                audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
                
                output_path = Path(args.output)
                torchaudio.save(str(output_path), audio_tensor, generator.sample_rate)
                print_success(f"Voice cloned and synthesized successfully on MLX GPU! Saved to: {output_path.resolve()}")
                
                stats = generator.last_run_stats
                print_header("Generation Statistics")
                print_info(f"  Wall Time (including compilation/warmup): {duration_sec:.2f} seconds")
                print_info(f"  Generated Audio Duration:                  {audio_len_sec:.2f} seconds")
                print_info(f"  Real-Time Factor (RTF):                   {rtf:.2f}x ({'Faster than real-time' if rtf < 1.0 else 'Slower than real-time'})")
                print_info(f"  Total Steps (Autoregressive Frames):     {total_steps}")
                print_info(f"  Generation Speed:                         {steps_per_sec:.2f} frames/sec")
                print_info(f"  Process Peak Memory (RSS):                {mem_after:.2f} MB")
                print_info(f"  Memory Allocated During Synthesis:        {max(0.0, mem_increase):.2f} MB")
                
                print_header("MLX Profiling Breakdown")
                print_info(f"  Prompt Tokenization Time:                  {stats.get('prompt_tokenization_time', 0.0):.2f} seconds")
                print_info(f"  Model Warmup Pass JIT Compilation Time:    {stats.get('warmup_time', 0.0):.2f} seconds")
                print_info(f"  Total Autoregressive Loop Time:            {stats.get('generation_loop_time', 0.0):.2f} seconds")
                print_info(f"    - First Step JIT Compilation Time:       {stats.get('first_step_time', 0.0):.2f} seconds")
                print_info(f"    - Subsequent Steps Running Time:         {stats.get('subsequent_steps_time', 0.0):.2f} seconds")
                print_info(f"    - Average Running Time per Step:         {stats.get('avg_subsequent_step_time', 0.0) * 1000.0:.1f} milliseconds")
                print_info(f"  Mimi Audio Codec Decoding Time:            {stats.get('mimi_decoding_time', 0.0):.2f} seconds")
                print_info(f"  SilentCipher Watermarking Time:            {stats.get('watermarking_time', 0.0):.2f} seconds")
                
                if is_json_mode:
                    print(json.dumps({
                        "status": "success",
                        "backend": "mlx",
                        "quantization_enabled": use_quant,
                        "text": args.text,
                        "prompt_audio": str(prompt_audio_path.resolve()),
                        "output": str(output_path.resolve()),
                        "sample_rate": generator.sample_rate,
                        "stats": {
                            "generation_time_sec": duration_sec,
                            "audio_duration_sec": audio_len_sec,
                            "real_time_factor": rtf,
                            "total_steps": total_steps,
                            "steps_per_second": steps_per_sec,
                            "peak_memory_mb": mem_after,
                            "memory_increase_mb": max(0.0, mem_increase),
                            "profiling": stats
                        }
                    }, indent=2))
                return
            except Exception as e:
                print_error(f"MLX Generator Cloning failed: {e}")
                print_info("Falling back to PyTorch CPU backend...")
                args.mlx = False

    if not args.mlx:
        print_info("Using PyTorch Backend...")
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            print_warning("Running on CPU. Since this is an 8B parameter model, generation will be SLOW.")

        print_info(f"Loading MisoTTS generator (Device: {device})...")
        from generator import load_miso_8b, Segment
        try:
            generator = load_miso_8b(device=device)
        except Exception as e:
            print_error(f"Failed to load model: {e}")
            sys.exit(1)

        if not prompt_audio_path.exists():
            print_error(f"Prompt audio file not found: {prompt_audio_path}")
            sys.exit(1)

        print_info(f"Loading reference voice prompt audio from: {prompt_audio_path}")
        try:
            prompt_audio, sample_rate = torchaudio.load(str(prompt_audio_path))
            prompt_audio = torchaudio.functional.resample(
                prompt_audio.squeeze(0),
                orig_freq=sample_rate,
                new_freq=generator.sample_rate,
            )
        except Exception as e:
            print_error(f"Failed to load prompt audio: {e}")
            sys.exit(1)

        print_info("Assembling voice-prompt conditioning context...")
        context = [
            Segment(
                speaker=args.speaker,
                text=args.prompt_text,
                audio=prompt_audio,
            )
        ]

        print_info(f"Synthesizing new text with cloned voice: \"{args.text}\"")
        try:
            audio = generator.generate(
                text=args.text,
                speaker=args.speaker,
                context=context,
                max_audio_length_ms=args.max_length_ms,
            )
            
            output_path = Path(args.output)
            torchaudio.save(str(output_path), audio.unsqueeze(0).cpu(), generator.sample_rate)
            print_success(f"Voice cloned and synthesized successfully! Saved to: {output_path.resolve()}")
            
            if is_json_mode:
                print(json.dumps({
                    "status": "success",
                    "backend": "pytorch",
                    "text": args.text,
                    "prompt_audio": str(prompt_audio_path.resolve()),
                    "output": str(output_path.resolve()),
                    "sample_rate": generator.sample_rate
                }, indent=2))
        except Exception as e:
            print_error(f"Cloning synthesis failed: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="MisoTTS 8B Apple Silicon Optimizer & Voice Synthesis CLI Tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Global Dual-Mode JSON Output override
    parser.add_argument("--json", action="store_true", help="Output results and status in machine-readable JSON format strictly on stdout.")
    
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    # Subcommand: optimize
    parser_optimize = subparsers.add_parser(
        "optimize",
        help="Check hardware suitability, install MLX modules, and prepare weight optimization.",
        epilog=f"""{BOLD}Examples:{RESET}
  {COLOR_COMMAND}# Check hardware and dependencies for local MLX inference{RESET}
  uv run python miso_mlx/miso_mlx_cli.py optimize

  {COLOR_COMMAND}# Get optimization report in machine-readable JSON format{RESET}
  uv run python miso_mlx/miso_mlx_cli.py optimize --json
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Subcommand: download
    parser_download = subparsers.add_parser(
        "download",
        help="Pre-download and cache all necessary model weights and tokenizer configurations.",
        epilog=f"""{BOLD}Examples:{RESET}
  {COLOR_COMMAND}# Download and cache standard weights and tokenizer assets{RESET}
  uv run python miso_mlx/miso_mlx_cli.py download

  {COLOR_COMMAND}# Download and get status in JSON format{RESET}
  uv run python miso_mlx/miso_mlx_cli.py download --json
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Subcommand: speak
    parser_speak = subparsers.add_parser(
        "speak",
        help="Synthesize text to speech using standard speaker presets.",
        epilog=f"""{BOLD}Examples:{RESET}
  {COLOR_COMMAND}# Synthesize locally on CPU (standard PyTorch fallback){RESET}
  uv run python miso_mlx/miso_mlx_cli.py speak --text "Hello from my Mac!" --output hello.wav

  {COLOR_COMMAND}# Synthesize at high-speed on local GPU using Apple Silicon MLX{RESET}
  uv run python miso_mlx/miso_mlx_cli.py speak --text "Hello from local GPU!" --mlx --output hello.wav

  {COLOR_COMMAND}# Perform a fast dry-run parameter validation without compiling the model{RESET}
  uv run python miso_mlx/miso_mlx_cli.py speak --text "Hello!" --mlx --dry-run
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_speak.add_argument("--text", "-t", type=str, required=True, help="Text sentence to synthesize.")
    parser_speak.add_argument("--speaker", "-s", type=int, default=0, help="Speaker index (e.g., 0 or 1).")
    parser_speak.add_argument("--output", "-o", type=str, default="output.wav", help="Path to save the output WAV file.")
    parser_speak.add_argument("--max_length_ms", type=float, default=10000, help="Maximum audio duration limit in milliseconds.")
    parser_speak.add_argument("--mlx", action="store_true", help="Enable Apple Silicon Metal-accelerated GPU backend.")
    parser_speak.add_argument("--quant", action="store_true", help="Enable 4-bit quantization for faster generation and lower memory footprint.")
    parser_speak.add_argument("--dry-run", action="store_true", help="Perform input validation and shapes testing without compiling/running models.")
    parser_speak.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for the codebook generation loops.")
    parser_speak.add_argument("--topk", type=int, default=50, help="Top-K vocabulary filtering constraint during sampling.")
    parser_speak.add_argument("--no-watermark", action="store_true", help="Bypass applying SilentCipher AI-generated voice watermarking.")
    parser_speak.add_argument("--temp-start", type=float, default=None, help="Starting temperature for dynamic temperature decay scheduling.")
    parser_speak.add_argument("--temp-min", type=float, default=None, help="Minimum/ending temperature for dynamic temperature decay scheduling.")
    parser_speak.add_argument("--temp-decay-steps", type=int, default=None, help="Number of steps over which temperature decays.")
    parser_speak.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-Free Guidance (CFG) scale (1.0 means disabled).")


    # Subcommand: clone
    parser_clone = subparsers.add_parser(
        "clone",
        help="Clone a voice from reference audio and synthesize new text.",
        epilog=f"""{BOLD}Examples:{RESET}
  {COLOR_COMMAND}# Clone voice on CPU (standard PyTorch fallback){RESET}
  uv run python miso_mlx/miso_mlx_cli.py clone \\
    --text "Synthesize this in the cloned voice" \\
    --prompt-audio my_reference.wav \\
    --prompt-text "The text read in my reference audio file" \\
    --output cloned.wav

  {COLOR_COMMAND}# Clone voice at high-speed on local GPU using Apple Silicon MLX{RESET}
  uv run python miso_mlx/miso_mlx_cli.py clone \\
    --text "Synthesize this in the cloned voice" \\
    --prompt-audio my_reference.wav \\
    --prompt-text "The text read in my reference audio file" \\
    --mlx --output cloned.wav

  {COLOR_COMMAND}# Perform a fast dry-run parameters and file validation of the cloning pipeline{RESET}
  uv run python miso_mlx/miso_mlx_cli.py clone \\
    --text "Validation test" \\
    --prompt-audio my_reference.wav \\
    --prompt-text "The text read in my reference audio file" \\
    --mlx --dry-run
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_clone.add_argument("--text", "-t", type=str, required=True, help="New text sentence to synthesize in the cloned voice.")
    parser_clone.add_argument("--prompt-audio", "-pa", type=str, required=True, help="Path to reference WAV file of the target voice.")
    parser_clone.add_argument("--prompt-text", "-pt", type=str, required=True, help="Exact text transcript of the reference audio.")
    parser_clone.add_argument("--speaker", "-s", type=int, default=0, help="Speaker index to override/apply voice profile onto.")
    parser_clone.add_argument("--output", "-o", type=str, default="cloned_output.wav", help="Path to save the output WAV file.")
    parser_clone.add_argument("--max_length_ms", type=float, default=10000, help="Maximum audio duration limit in milliseconds.")
    parser_clone.add_argument("--mlx", action="store_true", help="Enable Apple Silicon Metal-accelerated GPU backend.")
    parser_clone.add_argument("--quant", action="store_true", help="Enable 4-bit quantization for faster generation and lower memory footprint.")
    parser_clone.add_argument("--dry-run", action="store_true", help="Perform input validation and shapes testing without compiling/running models.")
    parser_clone.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for the codebook generation loops.")
    parser_clone.add_argument("--topk", type=int, default=50, help="Top-K vocabulary filtering constraint during sampling.")
    parser_clone.add_argument("--no-watermark", action="store_true", help="Bypass applying SilentCipher AI-generated voice watermarking.")
    parser_clone.add_argument("--temp-start", type=float, default=None, help="Starting temperature for dynamic temperature decay scheduling.")
    parser_clone.add_argument("--temp-min", type=float, default=None, help="Minimum/ending temperature for dynamic temperature decay scheduling.")
    parser_clone.add_argument("--temp-decay-steps", type=int, default=None, help="Number of steps over which temperature decays.")
    parser_clone.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-Free Guidance (CFG) scale (1.0 means disabled).")


    args = parser.parse_args()

    # Dynamic JSON override configuration
    global is_json_mode
    if args.json:
        is_json_mode = True

    if args.command == "optimize":
        handle_optimize(args)
    elif args.command == "download":
        handle_download(args)
    elif args.command == "speak":
        handle_speak(args)
    elif args.command == "clone":
        handle_clone(args)

if __name__ == "__main__":
    main()
