import os
import sys
import time
from PIL import Image, ImageDraw, ImageFont
from mlx_vlm import load, generate

def generate_image_if_missing():
    path = "outputs/validation_test_image.png"
    if os.path.exists(path):
        print(f"[Image] Found existing test image at {path}")
        return path
    
    print("[Image] Generating new test image...")
    os.makedirs("outputs", exist_ok=True)
    img = Image.new("RGB", (400, 400), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 80, 320, 320], fill=(220, 50, 50), outline=(180, 0, 0), width=5)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((200, 200), "Gemma 4", fill=(255, 255, 255), anchor="mm")
    img.save(path)
    print(f"[Image] Saved test image to {path}")
    return path

def main():
    model_path = "/Users/ghchinoy/projects/gemma/google-gemma-4-E2B-it-qat-q4_0-unquantized"
    print(f"Loading Gemma 4 model from: {model_path}")
    start_time = time.time()
    model, processor = load(model_path)
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.2f} seconds!")

    # 1. Primary Multimodal Interleaved Validation
    print("\n--- Running Part 1: Interleaved Multimodal (Image + Audio) Validation ---")
    image_path = generate_image_if_missing()
    audio_path = "/Users/ghchinoy/projects/eldamo-group/pronouncing-elvish/govannen_pronounce.wav"
    
    if not os.path.exists(audio_path):
        print(f"Error: Reference audio file not found at {audio_path}")
        sys.exit(1)
        
    messages_interleaved = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Below is an image:\n"},
                {"type": "image"},
                {"type": "text", "text": "\nAnd here is an audio recording:\n"},
                {"type": "audio"},
                {"type": "text", "text": "\nPlease perform two tasks:\n1. Describe the contents of the image.\n2. Transcribe the audio clip word-for-word."}
            ]
        }
    ]
    
    prompt_interleaved = processor.apply_chat_template(messages_interleaved, add_generation_prompt=True)
    print("Generating interleaved response...")
    
    start_gen = time.time()
    response_interleaved = generate(
        model,
        processor,
        prompt=prompt_interleaved,
        image=image_path,
        audio=audio_path,
        max_tokens=256,
        verbose=False
    )
    gen_time_interleaved = time.time() - start_gen
    print(f"Interleaved generation completed in {gen_time_interleaved:.2f} seconds!")
    print("\n=== Interleaved Response ===")
    print(response_interleaved)
    print("============================\n")

    # 2. Audio-only Transcription for Cross-backend Comparison
    print("--- Running Part 2: Transcription of synthesized hello_miso_gpu.wav ---")
    gpu_audio_path = "outputs/hello_miso_gpu.wav"
    if not os.path.exists(gpu_audio_path):
        print(f"Warning: Synthesized GPU audio not found at {gpu_audio_path}")
    else:
        messages_audio_only = [
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": "Transcribe this audio file accurately. Output only the transcription, nothing else."}
                ]
            }
        ]
        prompt_audio_only = processor.apply_chat_template(messages_audio_only, add_generation_prompt=True)
        print("Generating audio-only transcription...")
        
        start_gen_audio = time.time()
        response_audio = generate(
            model,
            processor,
            prompt=prompt_audio_only,
            audio=gpu_audio_path,
            max_tokens=128,
            verbose=False
        )
        gen_time_audio = time.time() - start_gen_audio
        print(f"Audio transcription completed in {gen_time_audio:.2f} seconds!")
        print("\n=== Gemma 4 Transcription ===")
        print(response_audio)
        print("=============================\n")

if __name__ == "__main__":
    main()
