#!/usr/bin/env python
import argparse
import os
import sys
import json
from pathlib import Path

# Resolve pathing so we can run from anywhere
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# AX/DX Environment Override Checks
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


def evaluate_audio_with_gemini(
    audio_path: Path,
    target_text: str = None,
    model_name: str = "gemini-3.1-flash-lite",
    project_id: str = None,
    location: str = "us-central1"
) -> dict:
    """
    Sends the audio file to Gemini via the Google GenAI SDK (Vertex AI backend)
    to transcribe and evaluate pronunciation quality, prosody, clarity, and correctness.
    """
    print_info(f"Initializing Google GenAI SDK client (Vertex AI Backend)...")
    
    # Try importing the new google-genai library
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "The modern 'google-genai' SDK is not installed. "
            "Please run: uv pip install google-genai"
        )

    # Automatically resolve Project ID if not provided
    if not project_id:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEX_PROJECT")
        if project_id:
            print_info(f"Resolved Cloud Project ID from environment: {project_id}")
        else:
            # Vertex backend requires a project ID. In some local environments, it falls back to default.
            print_warning("No Project ID specified. Attempting default Vertex initialization...")

    # Initialize client with Vertex AI backend
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location
    )

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Read audio bytes
    print_info(f"Reading target audio file: {audio_path.name} ({audio_path.stat().st_size} bytes)")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Formulate structured evaluation prompt
    prompt = (
        "You are an expert audio quality and speech assessment AI. You are evaluating a local Text-to-Speech (TTS) "
        "system port running on Apple Silicon. \n\n"
        "Please analyze the attached audio file and perform the following structured evaluation:\n"
        "1. **Transcription**: Transcribe the audio word-for-word.\n"
        "2. **Acoustic Clarity**: Assess if the speech is clear, or if there is static, buzzing, robotic distortion, or clipping.\n"
        "3. **Prosody & Naturalness**: Assess the pacing, flow, intonation, and whether it sounds like a natural human or a dry robot.\n"
        "4. **Completeness**: Did the model cut off early or run into infinite loops of silence/repetitions?\n"
    )

    if target_text:
        prompt += (
            f"5. **Accuracy Comparison**: Compare the transcribed text against the expected reference text: \"{target_text}\". "
            "Are there any missing words, wrong words, or spelling deviations?\n"
        )
    
    prompt += (
        "\nProvide your final assessment as a concise review with a 'Speech Quality Score' (0-100) and 'Alignment Accuracy' (0-100)."
    )

    print_info(f"Sending audio assessment request to Vertex AI using model '{model_name}'...")
    
    # Send contents with native multi-modal support (passing audio bytes directly)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/wav"
            ),
            prompt
        ]
    )

    return {
        "model": model_name,
        "backend": "vertex_ai",
        "project": project_id,
        "location": location,
        "audio_file": str(audio_path.resolve()),
        "assessment": response.text
    }


def main():
    global is_json_mode
    
    parser = argparse.ArgumentParser(
        description="Google GenAI SDK Audio Evaluator for MisoTTS Outputs via Vertex AI Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a synthesized MLX GPU audio file
  uv run python miso_mlx/audio_evaluator.py -a hello_miso_gpu.wav -t "Hello! This is synthesized locally on my Mac."

  # Evaluate using a custom Google Cloud Project ID and location
  uv run python miso_mlx/audio_evaluator.py -a output.wav -p my-gcp-project-123 -l us-east4
        """
    )
    
    parser.add_argument("--audio", "-a", type=str, required=True, help="Path to the synthesized WAV audio file.")
    parser.add_argument("--text", "-t", type=str, default=None, help="The target text used to generate the speech.")
    parser.add_argument("--model", "-m", type=str, default="gemini-3.1-flash-lite", help="Gemini model name on Vertex AI.")
    parser.add_argument("--project", "-p", type=str, default=None, help="Google Cloud Project ID for Vertex AI.")
    parser.add_argument("--location", "-l", type=str, default="us-central1", help="Google Cloud Location region.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON evaluation data to stdout.")
    
    args = parser.parse_args()
    
    is_json_mode = args.json
    
    if not is_json_mode:
        print_header("MisoTTS Google GenAI Vertex Audio Evaluator")
    
    audio_path = Path(args.audio)
    
    try:
        results = evaluate_audio_with_gemini(
            audio_path=audio_path,
            target_text=args.text,
            model_name=args.model,
            project_id=args.project,
            location=args.location
        )
        
        if is_json_mode:
            # Clean JSON output to stdout
            print(json.dumps(results, indent=2))
        else:
            print_success("Evaluation completed successfully!")
            print_info(f"Model used: {results['model']} on Vertex AI ({results['location']})")
            print("\n" + "=" * 60)
            print(f"{BOLD}Gemini Audio Quality Evaluation & Transcription:{RESET}")
            print("=" * 60)
            print(results["assessment"])
            print("=" * 60)
            
    except Exception as e:
        print_error(f"Audio evaluation failed: {e}")
        if is_json_mode:
            print(json.dumps({"status": "error", "reason": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
