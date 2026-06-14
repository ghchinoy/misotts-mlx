#!/usr/bin/env python
import argparse
import os
import sys
import time
import json
from pathlib import Path

# Resolve pathing so we can load model parameters from local mlx_model
miso_mlx_dir = Path(__file__).parent.resolve()
if str(miso_mlx_dir) not in sys.path:
    sys.path.append(str(miso_mlx_dir))

original_sources_path = str(miso_mlx_dir.parent / "sources" / "MisoTTS")
if original_sources_path not in sys.path:
    sys.path.append(original_sources_path)

# Environment variables for terminal coloring
disable_color = "NO_COLOR" in os.environ or os.environ.get("MISO_NO_TUI") == "1"

# ANSI Terminal Styling Constants
BOLD = "" if disable_color else "\033[1m"
RESET = "" if disable_color else "\033[0m"
COLOR_ACCENT = "" if disable_color else "\033[34m"   # Blue
COLOR_PASS = "" if disable_color else "\033[32m"     # Green
COLOR_WARN = "" if disable_color else "\033[33m"     # Yellow
COLOR_FAIL = "" if disable_color else "\033[31m"     # Red
COLOR_CYAN = "" if disable_color else "\033[36m"     # Cyan

def print_log(msg: str, style_prefix: str = "", is_error: bool = False, is_json_mode: bool = False):
    """
    Prints styled log messages. If running in JSON mode, routes logs to stderr
    so stdout remains clean for parsing.
    """
    if is_json_mode:
        print(f"{style_prefix}{msg}{RESET}", file=sys.stderr, flush=True)
    else:
        file = sys.stderr if is_error else sys.stdout
        print(f"{style_prefix}{msg}{RESET}", file=file, flush=True)

def main():
    parser = argparse.ArgumentParser(
        description="MisoTTS MLX Weight Quantizer Utility",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--bits", "-b", type=int, default=4, help="Number of bits per quantized weight (default: 4).")
    parser.add_argument("--group-size", "-g", type=int, default=64, help="Quantization group size (default: 64).")
    parser.add_argument("--input", "-i", type=str, default=str(miso_mlx_dir / "mlx_weights" / "model.safetensors"),
                        help="Path to unquantized input weights in safetensors format.")
    parser.add_argument("--output", "-o", type=str, default=str(miso_mlx_dir / "mlx_weights" / "quantized_model_4bit.safetensors"),
                        help="Path to save the quantized output weights.")
    parser.add_argument("--json", action="store_true", help="Output only the final status JSON to stdout, forwarding logs to stderr.")

    args = parser.parse_args()
    is_json_mode = args.json

    print_log(f"=== MisoTTS MLX 4-bit Weight Quantizer ===", f"{BOLD}{COLOR_CYAN}", is_json_mode=is_json_mode)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print_log(f"Input weight file not found at: {input_path.resolve()}", f"{BOLD}{COLOR_FAIL}", is_error=True, is_json_mode=is_json_mode)
        if is_json_mode:
            print(json.dumps({"status": "error", "reason": f"Input file {input_path} does not exist"}))
        sys.exit(1)

    print_log(f"ℹ Loading MLX core and dependencies...", f"{COLOR_ACCENT}", is_json_mode=is_json_mode)
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_model import ModelArgs, MisoTTSModel
    except ImportError as e:
        print_log(f"MLX dependency missing: {e}", f"{BOLD}{COLOR_FAIL}", is_error=True, is_json_mode=is_json_mode)
        if is_json_mode:
            print(json.dumps({"status": "error", "reason": f"ImportError: {e}"}))
        sys.exit(1)

    print_log(f"ℹ Initializing clean MisoTTS float16 architecture shell...", f"{COLOR_ACCENT}", is_json_mode=is_json_mode)
    model_args = ModelArgs()
    model = MisoTTSModel(model_args)

    print_log(f"ℹ Loading bfloat16 base weights from: {input_path.resolve()}...", f"{COLOR_ACCENT}", is_json_mode=is_json_mode)
    t_load_start = time.perf_counter()
    try:
        # Load weights with strict=False in case of minor metadata discrepancies
        model.load_weights(str(input_path), strict=False)
        load_duration = time.perf_counter() - t_load_start
        print_log(f"✔ Base weights loaded successfully in {load_duration:.2f} seconds.", f"{BOLD}{COLOR_PASS}", is_json_mode=is_json_mode)
    except Exception as e:
        print_log(f"Failed to load weights: {e}", f"{BOLD}{COLOR_FAIL}", is_error=True, is_json_mode=is_json_mode)
        if is_json_mode:
            print(json.dumps({"status": "error", "reason": f"Failed to load weights: {e}"}))
        sys.exit(1)

    print_log(f"ℹ Performing in-place {args.bits}-bit quantization of Linear layers (Group Size: {args.group_size})...", f"{COLOR_ACCENT}", is_json_mode=is_json_mode)
    t_quant_start = time.perf_counter()
    try:
        # Only quantize nn.Linear layers, leaving nn.Embedding and other special tensors intact
        nn.quantize(
            model,
            group_size=args.group_size,
            bits=args.bits,
            class_predicate=lambda k, m: isinstance(m, nn.Linear)
        )
        quant_duration = time.perf_counter() - t_quant_start
        print_log(f"✔ Quantization completed in {quant_duration:.2f} seconds.", f"{BOLD}{COLOR_PASS}", is_json_mode=is_json_mode)
    except Exception as e:
        print_log(f"Quantization process failed: {e}", f"{BOLD}{COLOR_FAIL}", is_error=True, is_json_mode=is_json_mode)
        if is_json_mode:
            print(json.dumps({"status": "error", "reason": f"Quantization failed: {e}"}))
        sys.exit(1)

    print_log(f"ℹ Serializing and saving quantized weights to: {output_path.resolve()}...", f"{COLOR_ACCENT}", is_json_mode=is_json_mode)
    t_save_start = time.perf_counter()
    try:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        model.save_weights(str(output_path))
        save_duration = time.perf_counter() - t_save_start
        print_log(f"✔ Quantized weights saved successfully in {save_duration:.2f} seconds.", f"{BOLD}{COLOR_PASS}", is_json_mode=is_json_mode)
    except Exception as e:
        print_log(f"Saving quantized weights failed: {e}", f"{BOLD}{COLOR_FAIL}", is_error=True, is_json_mode=is_json_mode)
        if is_json_mode:
            print(json.dumps({"status": "error", "reason": f"Saving failed: {e}"}))
        sys.exit(1)

    # Calculate statistics
    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    compression_ratio = input_size / output_size if output_size > 0 else 0.0
    size_reduction_pct = (1.0 - (output_size / input_size)) * 100.0 if input_size > 0 else 0.0

    print_log(f"\n=== Quantization Summary ===", f"{BOLD}{COLOR_CYAN}", is_json_mode=is_json_mode)
    print_log(f"  Unquantized Size: {input_size / (1024**3):.2f} GB ({input_size:,} bytes)", f"{COLOR_ACCENT}", is_json_mode=is_json_mode)
    print_log(f"  Quantized Size:   {output_size / (1024**3):.2f} GB ({output_size:,} bytes)", f"{COLOR_ACCENT}", is_json_mode=is_json_mode)
    print_log(f"  Compression:      {compression_ratio:.2f}x reduction ({size_reduction_pct:.1f}% space saved)", f"{BOLD}{COLOR_PASS}", is_json_mode=is_json_mode)
    print_log(f"  Total Time Taken: {load_duration + quant_duration + save_duration:.2f} seconds", f"{COLOR_ACCENT}", is_json_mode=is_json_mode)

    if is_json_mode:
        results = {
            "status": "quantization_success",
            "bits": args.bits,
            "group_size": args.group_size,
            "input_file": str(input_path.resolve()),
            "output_file": str(output_path.resolve()),
            "input_size_bytes": input_size,
            "output_size_bytes": output_size,
            "compression_ratio": compression_ratio,
            "size_reduction_pct": size_reduction_pct,
            "load_time_sec": load_duration,
            "quant_time_sec": quant_duration,
            "save_time_sec": save_duration
        }
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
