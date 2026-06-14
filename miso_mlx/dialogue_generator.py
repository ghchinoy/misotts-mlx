#!/usr/bin/env python3
"""
MisoTTS Multi-Speaker Contextual Dialogue Generator (MLX)
---------------------------------------------------------
This script demonstrates how to synthesize a multi-turn, multi-speaker conversation
locally on Apple Silicon GPUs using the MLX framework.

Instead of generating unrelated sentences, each turn is contextually conditioned on 
the history of the conversation (speaker, text, and audio waveforms) by leveraging 
MisoTTS's Llama-based autoregressive conditioning context.
"""

import sys
import os
import time
from pathlib import Path
import resource

# Add miso_mlx to system path to ensure local imports resolve
miso_mlx_dir = Path(__file__).resolve().parent
if str(miso_mlx_dir) not in sys.path:
    sys.path.append(str(miso_mlx_dir))

import mlx.core as mx
import mlx.nn as nn
import torch
import torchaudio

from mlx_model import ModelArgs, MisoTTSModel
from mlx_generator import MLXGenerator, Segment as MLXSegment


def get_peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    else:
        return usage / 1024


def main():
    print("=" * 60)
    print(" MisoTTS Contextual Multi-Speaker Dialogue Generator (MLX) ")
    print("=" * 60)

    # 1. Resolve model weights
    mlx_weights_file = miso_mlx_dir / "mlx_weights" / "model.safetensors"
    quantized_weights_file = miso_mlx_dir / "mlx_weights" / "quantized_model_4bit.safetensors"
    
    use_quant = False
    if quantized_weights_file.exists():
        use_quant = True
        weights_path = quantized_weights_file
    elif mlx_weights_file.exists():
        weights_path = mlx_weights_file
    else:
        print(f"❌ Error: No MLX weights found in {miso_mlx_dir}/mlx_weights/.")
        print("   Please run weight download and conversion steps first.")
        sys.exit(1)

    # 2. Initialize Model
    print(f"Initializing MLX model layers on GPU (Quantized: {use_quant})...")
    args_model = ModelArgs()
    model = MisoTTSModel(args_model)

    if use_quant:
        print("Applying 4-bit quantization to model architecture...")
        nn.quantize(
            model,
            group_size=64,
            bits=4,
            class_predicate=lambda k, m: isinstance(m, nn.Linear)
        )
    
    print(f"Loading weights from: {weights_path.name}")
    model.load_weights(str(weights_path), strict=False)
    print("✅ Model weights loaded successfully!")

    # 3. Create Generator
    print("Initializing MLX Generator...")
    generator = MLXGenerator(model)
    print("✅ Generator initialized.")

    # 4. Define the Dialogue
    dialogue_turns = [
        {
            "speaker": 0, 
            "text": "Hello there! I am Speaker Zero. Welcome to our local dialogue demo."
        },
        {
            "speaker": 1, 
            "text": "Hi! I am Speaker One. By chaining our model states, we can have a natural, multi-turn conversation."
        },
        {
            "speaker": 0, 
            "text": "That is incredible. It actually remembers the timbre of our voices and what we said."
        },
        {
            "speaker": 1, 
            "text": "Yes, the attention cache tracks everything, keeping our voices distinct and the flow natural."
        }
    ]

    print(f"\nStarting synthesis of a {len(dialogue_turns)}-turn conversation...")
    print("-" * 60)

    context = []
    generated_waveforms = []
    
    start_time = time.perf_counter()
    mem_before = get_peak_memory_mb()

    for idx, turn in enumerate(dialogue_turns):
        turn_speaker = turn["speaker"]
        turn_text = turn["text"]
        
        print(f"\n[Turn {idx + 1}/{len(dialogue_turns)}] Speaker {turn_speaker}: \"{turn_text}\"")
        
        t0 = time.perf_counter()
        
        # Generate speech for the current turn, passing the cumulative conversation context list
        audio_array = generator.generate(
            text=turn_text,
            speaker=turn_speaker,
            context=context,
            max_audio_length_ms=10000.0,
            temperature=0.8,
            topk=50,
            no_watermark=True,  # Bypass watermark for cleaner combined audio
            temp_start=0.7,
            temp_min=0.4,
            temp_decay_steps=30,
            cfg_scale=2.0
        )
        
        duration = time.perf_counter() - t0
        audio_duration = len(audio_array) / generator.sample_rate
        rtf = duration / audio_duration if audio_duration > 0 else 0.0
        
        print(f"  └─ Generated {audio_duration:.2f}s of audio in {duration:.2f}s (RTF: {rtf:.2f}x)")
        
        # Save generated audio as part of this turn's Segment object.
        # This segment is then passed as context for all subsequent turns, keeping attention states aligned!
        segment = MLXSegment(
            speaker=turn_speaker,
            text=turn_text,
            audio=audio_array
        )
        context.append(segment)
        generated_waveforms.append(audio_array)

    total_wall_time = time.perf_counter() - start_time
    mem_after = get_peak_memory_mb()

    # 5. Concatenate and Save output
    print("\n" + "-" * 60)
    print("Chaining conversation turns into a single unified audio file...")
    
    # Contextual concatenation of all waveforms
    full_conversation_audio = mx.concatenate(generated_waveforms, axis=0)
    
    # Convert MLX array to PyTorch CPU tensor for saving
    np_audio = np_audio = full_conversation_audio.tolist() # Or np.array()
    import numpy as np
    audio_tensor = torch.from_numpy(np.array(full_conversation_audio)).unsqueeze(0)
    
    # Normalize final audio output to avoid clipping
    max_val = torch.max(torch.abs(audio_tensor))
    if max_val > 0:
        audio_tensor = (audio_tensor / max_val) * 0.95
    audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)

    output_path = Path("outputs/conversation_demo.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torchaudio.save(str(output_path), audio_tensor, generator.sample_rate)
    
    total_audio_len = len(full_conversation_audio) / generator.sample_rate
    print(f"✅ Conversation saved successfully to: {output_path.resolve()}")
    print(f"  - Total Dialogue Duration: {total_audio_len:.2f} seconds")
    print(f"  - Total Synthesis Time:    {total_wall_time:.2f} seconds")
    print(f"  - Overall Real-Time Factor: {(total_wall_time / total_audio_len):.2f}x")
    print(f"  - Peak Memory Usage:       {mem_after:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
