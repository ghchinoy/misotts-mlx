from dataclasses import dataclass
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Append the original MisoTTS sources path so we can resolve its submodules if needed
original_sources_path = str(Path(__file__).parent.parent / "sources" / "MisoTTS")
if original_sources_path not in sys.path:
    sys.path.append(original_sources_path)

# Apply bitsandbytes import patch for unquantized layers on macOS before importing other modules
try:
    from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers
    patch_bitsandbytes_import_for_unquantized_layers()
except ImportError:
    pass

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

from mlx_model import ModelArgs, MisoTTSModel


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000, stored as a float32 numpy array or MLX array
    audio: mx.array


def sample_topk(logits: mx.array, topk: int, temperature: float) -> mx.array:
    """
    Performs Top-K filtered sampling on MLX arrays.
    """
    logits = logits / temperature
    
    # Extract the top-k values across the last dimension (returned in ascending order in MLX)
    val = mx.topk(logits, topk, axis=-1)
    threshold = val[:, 0:1] # Index 0 is the smallest of the top-k values
    
    # Mask values below the threshold
    mask = logits < threshold
    filtered_logits = mx.where(mask, -float("inf"), logits)
    
    # Sample from the probability distribution
    return mx.random.categorical(filtered_logits, num_samples=1)


class MLXGenerator:
    def __init__(
        self,
        model: MisoTTSModel,
        tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    ):
        self.model = model
        self.device = "metal" # MLX runs automatically on Apple Silicon Metal GPUs
        self.last_run_stats = {}

        # Load standard Llama 3.2 text tokenizer with fallback to unsloth if gated
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            fallback_name = "unsloth/Llama-3.2-1B" if tokenizer_name != "unsloth/Llama-3.2-1B" else "meta-llama/Llama-3.2-1B"
            self.text_tokenizer = AutoTokenizer.from_pretrained(fallback_name)
            
        # Configure special token formatting post-processor
        bos = self.text_tokenizer.bos_token
        eos = self.text_tokenizer.eos_token
        self.text_tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", self.text_tokenizer.bos_token_id), (f"{eos}", self.text_tokenizer.eos_token_id)],
        )
        
        # Load the real Mimi audio tokenizer (Mimi neural audio codec) via PyTorch/huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            from moshi.models import loaders
            mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            mimi = loaders.get_mimi(mimi_weight, device="cpu")
            mimi.set_num_codebooks(self.model.args.audio_num_codebooks)
            self.audio_tokenizer = mimi
            self.sample_rate = mimi.sample_rate
        except Exception as e:
            print(f"⚠ Failed to load real Mimi audio tokenizer: {e}")
            self.audio_tokenizer = None
            self.sample_rate = 24000

        # Load SilentCipher watermarker
        try:
            from watermarking import load_watermarker
            self.watermarker = load_watermarker(device="cpu")
        except Exception as e:
            print(f"⚠ Failed to load SilentCipher watermarker: {e}")
            self.watermarker = None

        # Warmup and compiled decoder initialization
        # Capture the model as closure to prevent direct nn.Module parameter-dict conversion
        @mx.compile
        def _decode_frame_compiled(last_h, c0_logits, topk, temperature):
            c0_sample = sample_topk(c0_logits, topk, temperature)
            c0_embed = model._embed_audio(0, c0_sample)
            curr_h = mx.concatenate([mx.expand_dims(last_h, axis=1), c0_embed], axis=1)
            curr_sample = c0_sample
            
            curr_pos = mx.arange(curr_h.shape[1])[None, :]
            
            for cb in range(1, model.args.audio_num_codebooks):
                decoder_input = model.projection(curr_h)
                dec_mask = nn.MultiHeadAttention.create_additive_causal_mask(decoder_input.shape[1])
                decoder_h, _ = model.decoder(
                    decoder_input,
                    mask=dec_mask,
                    caches=None,
                    input_pos=curr_pos
                )
                
                ci_logits = mx.matmul(decoder_h[:, -1, :], model.audio_head[cb - 1])
                ci_sample = sample_topk(ci_logits, topk, temperature)
                ci_embed = model._embed_audio(cb, ci_sample)
                
                curr_sample = mx.concatenate([curr_sample, ci_sample], axis=-1)
                curr_h = mx.concatenate([curr_h, ci_embed], axis=1)
                curr_pos = mx.arange(curr_h.shape[1])[None, :]
                
            return curr_sample

        self._decode_frame_compiled = _decode_frame_compiled

        @mx.compile
        def _decode_frame_cfg_compiled(last_h_cond, last_h_uncond, c0_logits_cond, c0_logits_uncond, cfg_scale, topk, temperature):
            # c0 CFG
            c0_logits = c0_logits_uncond + cfg_scale * (c0_logits_cond - c0_logits_uncond)
            c0_sample = sample_topk(c0_logits, topk, temperature)
            c0_embed = model._embed_audio(0, c0_sample)
            
            curr_h_cond = mx.concatenate([mx.expand_dims(last_h_cond, axis=1), c0_embed], axis=1)
            curr_h_uncond = mx.concatenate([mx.expand_dims(last_h_uncond, axis=1), c0_embed], axis=1)
            
            curr_sample = c0_sample
            curr_pos = mx.arange(curr_h_cond.shape[1])[None, :]
            
            for cb in range(1, model.args.audio_num_codebooks):
                # Conditional pass
                decoder_input_cond = model.projection(curr_h_cond)
                dec_mask_cond = nn.MultiHeadAttention.create_additive_causal_mask(decoder_input_cond.shape[1])
                decoder_h_cond, _ = model.decoder(
                    decoder_input_cond,
                    mask=dec_mask_cond,
                    caches=None,
                    input_pos=curr_pos
                )
                ci_logits_cond = mx.matmul(decoder_h_cond[:, -1, :], model.audio_head[cb - 1])
                
                # Unconditional pass
                decoder_input_uncond = model.projection(curr_h_uncond)
                dec_mask_uncond = nn.MultiHeadAttention.create_additive_causal_mask(decoder_input_uncond.shape[1])
                decoder_h_uncond, _ = model.decoder(
                    decoder_input_uncond,
                    mask=dec_mask_uncond,
                    caches=None,
                    input_pos=curr_pos
                )
                ci_logits_uncond = mx.matmul(decoder_h_uncond[:, -1, :], model.audio_head[cb - 1])
                
                # Apply CFG to codebook i logits
                ci_logits = ci_logits_uncond + cfg_scale * (ci_logits_cond - ci_logits_uncond)
                
                ci_sample = sample_topk(ci_logits, topk, temperature)
                ci_embed = model._embed_audio(cb, ci_sample)
                
                curr_sample = mx.concatenate([curr_sample, ci_sample], axis=-1)
                curr_h_cond = mx.concatenate([curr_h_cond, ci_embed], axis=1)
                curr_h_uncond = mx.concatenate([curr_h_uncond, ci_embed], axis=1)
                curr_pos = mx.arange(curr_h_cond.shape[1])[None, :]
                
            return curr_sample

        self._decode_frame_cfg_compiled = _decode_frame_cfg_compiled

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[mx.array, mx.array]:
        """
        Formats text segment to '[speaker] text' and tokenizes it to MLX representation.
        Returns:
            tokens: (seq_len, audio_num_codebooks + 1)
            masks:  (seq_len, audio_num_codebooks + 1)
        """
        formatted_text = f"[{speaker}] {text.lstrip()}"
        text_tokens = self.text_tokenizer.encode(formatted_text)
        
        seq_len = len(text_tokens)
        frame_size = self.model.args.audio_num_codebooks + 1
        
        tokens = mx.zeros((seq_len, frame_size), dtype=mx.int32)
        masks = mx.zeros((seq_len, frame_size), dtype=mx.bool_)
        
        # Place text tokens in the final channel
        tokens[:, -1] = mx.array(text_tokens, dtype=mx.int32)
        masks[:, -1] = True
        
        return tokens, masks

    def _tokenize_audio(self, audio: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Tokenizes reference/prompt audio to Mimi codebooks.
        """
        if self.audio_tokenizer is not None and len(audio) > 0:
            import torch
            import numpy as np
            
            # Convert MLX array to PyTorch CPU tensor
            np_audio = np.array(audio)
            torch_audio = torch.from_numpy(np_audio).float()
            
            # (num_codebooks, seq_len)
            with torch.no_grad():
                audio_tokens = self.audio_tokenizer.encode(torch_audio.unsqueeze(0).unsqueeze(0))[0]
                # Add EOS frame (column of zeros)
                eos_frame = torch.zeros(audio_tokens.size(0), 1)
                audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
            
            # Convert back to MLX array
            np_tokens = audio_tokens.numpy() # (num_codebooks, seq_len + 1)
            audio_tokens_mx = mx.array(np_tokens, dtype=mx.int32)
            
            seq_len = audio_tokens_mx.shape[1]
            frame_size = self.model.args.audio_num_codebooks + 1
            
            tokens = mx.zeros((seq_len, frame_size), dtype=mx.int32)
            masks = mx.zeros((seq_len, frame_size), dtype=mx.bool_)
            
            tokens[:, :-1] = audio_tokens_mx.T
            masks[:, :-1] = True
            
            return tokens, masks
        else:
            # Fallback mock simulation for codebook extraction
            seq_len = max(1, len(audio) // 1920) # 24kHz / 12.5Hz = 1920 samples per frame
            frame_size = self.model.args.audio_num_codebooks + 1
            
            tokens = mx.zeros((seq_len, frame_size), dtype=mx.int32)
            masks = mx.zeros((seq_len, frame_size), dtype=mx.bool_)
            
            # Mimi codes occupy channels 0 to 31
            masks[:, :-1] = True
            return tokens, masks

    def _tokenize_segment(self, segment: Segment) -> Tuple[mx.array, mx.array]:
        """Combines text and audio embeddings for full contextual conditioning."""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        
        return (
            mx.concatenate([text_tokens, audio_tokens], axis=0),
            mx.concatenate([text_masks, audio_masks], axis=0)
        )

    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 10000.0,
        temperature: float = 0.9,
        topk: int = 50,
        no_watermark: bool = False,
        temp_start: Optional[float] = None,
        temp_min: Optional[float] = None,
        temp_decay_steps: Optional[int] = None,
        cfg_scale: float = 1.0,
        watermark_key: Optional[List[int]] = None,
        is_json_mode: bool = False,
    ) -> mx.array:
        """
        The main MLX GPU-accelerated autoregressive speech synthesis loop.
        """
        import json
        t_gen_start = time.perf_counter()
        
        use_cfg = cfg_scale != 1.0
        log_file = sys.stderr if is_json_mode else sys.stdout
        
        # 1. Compile prompt context
        t_token_start = time.perf_counter()
        tokens_list = []
        masks_list = []
        
        for segment in context:
            seg_tokens, seg_masks = self._tokenize_segment(segment)
            tokens_list.append(seg_tokens)
            masks_list.append(seg_masks)
            
        # Add the target text sequence to synthesize
        target_tokens, target_masks = self._tokenize_text_segment(text, speaker)
        tokens_list.append(target_tokens)
        masks_list.append(target_masks)
        
        prompt_tokens = mx.concatenate(tokens_list, axis=0)
        prompt_masks = mx.concatenate(masks_list, axis=0)
        
        prompt_tokenization_time = time.perf_counter() - t_token_start
        
        # Generation configuration
        max_generation_len = int(max_audio_length_ms / 80)  # 12.5 frames per second
        
        # 2. Set up model caches for Backbone 8B and Decoder 300M
        # In MLX, cache is initialized simply as None and grows dynamically
        backbone_caches_cond = [None] * self.model.args.backbone_layers
        backbone_caches_uncond = [None] * self.model.args.backbone_layers if use_cfg else None
        
        # Warmup backbone cache with prompt context
        # We process the entire prompt sequence in a single parallel step to fill caches
        t_warmup_start = time.perf_counter()
        curr_tokens = mx.expand_dims(prompt_tokens, axis=0)
        curr_masks_cond = mx.expand_dims(prompt_masks, axis=0)
        input_pos = mx.arange(prompt_tokens.shape[0])
        
        # Conditional pass
        c0_logits_cond, last_h_cond, backbone_caches_cond, _ = self.model(
            tokens=curr_tokens,
            tokens_mask=curr_masks_cond,
            input_pos=input_pos,
            backbone_caches=backbone_caches_cond
        )
        
        if use_cfg:
            # Unconditional pass: set last channel (text) of prompt masks to False
            uncond_channel_mask = mx.ones((prompt_masks.shape[1],), dtype=mx.bool_)
            uncond_channel_mask[-1] = False
            curr_masks_uncond = prompt_masks & uncond_channel_mask[None, :]
            curr_masks_uncond = mx.expand_dims(curr_masks_uncond, axis=0)
            
            c0_logits_uncond, last_h_uncond, backbone_caches_uncond, _ = self.model(
                tokens=curr_tokens,
                tokens_mask=curr_masks_uncond,
                input_pos=input_pos,
                backbone_caches=backbone_caches_uncond
            )
            
            # Force evaluation of both conditional and unconditional warmup passes
            mx.eval(c0_logits_cond, last_h_cond, backbone_caches_cond, c0_logits_uncond, last_h_uncond, backbone_caches_uncond)
        else:
            # Force evaluation of the warmup pass to trigger model weight JIT compiler load
            mx.eval(c0_logits_cond, last_h_cond, backbone_caches_cond)
            
        warmup_time = time.perf_counter() - t_warmup_start
        
        samples = []
        
        # 3. Step-by-step frame generation loop on GPU
        print(f"[MLX] Starting frame generation loop (total {max_generation_len} steps)...", file=log_file, flush=True)
        t_loop_start = time.perf_counter()
        first_step_time = 0.0
        subsequent_steps_time = 0.0
        
        for step in range(max_generation_len):
            t_step_start = time.perf_counter()
            
            # Dynamic temperature decay scheduling
            if temp_start is not None and temp_min is not None:
                N = temp_decay_steps if temp_decay_steps is not None else max_generation_len
                if N <= 1:
                    current_temp = temp_min
                elif step >= N:
                    current_temp = temp_min
                else:
                    if temp_start > 0 and temp_min > 0:
                        current_temp = temp_start * ((temp_min / temp_start) ** (step / (N - 1)))
                    else:
                        current_temp = temp_start + (step / (N - 1)) * (temp_min - temp_start)
            else:
                current_temp = temperature
            
            # Print state details
            cfg_suffix = f" (CFG: {cfg_scale})" if use_cfg else ""
            temp_suffix = f" (Temp: {current_temp:.4f})" if (temp_start is not None and temp_min is not None) else ""
            print(f"[MLX] Step {step+1}/{max_generation_len}{cfg_suffix}{temp_suffix} - Running compiled decoder frame...", end="", file=log_file, flush=True)
            
            if is_json_mode:
                print(json.dumps({
                    "type": "progress",
                    "step": step + 1,
                    "total_steps": max_generation_len,
                    "cfg": cfg_scale if use_cfg else 1.0,
                    "temp": float(current_temp)
                }), file=sys.stdout, flush=True)
            
            # Convert scalar parameters to MLX arrays to avoid recompilation
            temp_mx = mx.array(current_temp, dtype=mx.float32)
            
            # Synthesize all codebooks using compiled Metal shaders
            if use_cfg:
                cfg_mx = mx.array(cfg_scale, dtype=mx.float32)
                curr_sample = self._decode_frame_cfg_compiled(
                    last_h_cond, last_h_uncond,
                    c0_logits_cond, c0_logits_uncond,
                    cfg_mx, topk, temp_mx
                )
            else:
                curr_sample = self._decode_frame_compiled(
                    last_h_cond, c0_logits_cond, topk, temp_mx
                )
            
            if mx.all(curr_sample == 0):
                print(" [EOS] reached.", file=log_file, flush=True)
                break
                
            # Store the fully synthesized frame
            samples.append(curr_sample)
            
            # Feed current sample back as input for next backbone step
            next_step_token = mx.concatenate([curr_sample, mx.zeros((1, 1), dtype=mx.int32)], axis=-1)
            next_step_mask = mx.ones((1, self.model.args.audio_num_codebooks + 1), dtype=mx.bool_)
            next_step_mask[0, -1] = False # Text token is missing/ignored
            
            # Forward pass through backbone for next step logits
            print(" Done. Running backbone forward...", end="", file=log_file, flush=True)
            
            # Conditional step forward pass
            c0_logits_cond, last_h_cond, backbone_caches_cond, _ = self.model(
                tokens=mx.expand_dims(next_step_token, axis=0),
                tokens_mask=mx.expand_dims(next_step_mask, axis=0),
                input_pos=mx.array([prompt_tokens.shape[0] + step]),
                backbone_caches=backbone_caches_cond
            )
            
            if use_cfg:
                # Unconditional step forward pass
                c0_logits_uncond, last_h_uncond, backbone_caches_uncond, _ = self.model(
                    tokens=mx.expand_dims(next_step_token, axis=0),
                    tokens_mask=mx.expand_dims(next_step_mask, axis=0),
                    input_pos=mx.array([prompt_tokens.shape[0] + step]),
                    backbone_caches=backbone_caches_uncond
                )
                
                # Force immediate evaluation of step results to prevent lazy graph compilation explosion
                print(" Done. Force evaluating step...", end="", file=log_file, flush=True)
                mx.eval(c0_logits_cond, last_h_cond, backbone_caches_cond, c0_logits_uncond, last_h_uncond, backbone_caches_uncond, curr_sample)
            else:
                # Force immediate evaluation of step results to prevent lazy graph compilation explosion
                print(" Done. Force evaluating step...", end="", file=log_file, flush=True)
                mx.eval(c0_logits_cond, last_h_cond, backbone_caches_cond, curr_sample)
                
            print(" Step completed!", file=log_file, flush=True)
            
            step_duration = time.perf_counter() - t_step_start
            if step == 0:
                first_step_time = step_duration
            else:
                subsequent_steps_time += step_duration
                
        generation_loop_time = time.perf_counter() - t_loop_start
        
        # 4. Reconstruct audio wave from predicted codebooks using real Mimi decoder
        print(f"[MLX] Successfully generated {len(samples)} audio frames on GPU.", file=log_file, flush=True)
        
        t_mimi_start = time.perf_counter()
        audio_output = None
        
        if self.audio_tokenizer is not None and len(samples) > 0:
            import torch
            import numpy as np
            import torchaudio
            from watermarking import MISO_TTS_WATERMARK, watermark
            
            print("[MLX] Reconstructing audio waveform using Mimi decoder...", end="", file=log_file, flush=True)
            np_samples = [np.array(s) for s in samples]
            torch_samples = torch.from_numpy(np.stack(np_samples)).long() # (seq_len, 1, 32)
            
            # Reshape to (1, 32, seq_len) to match Mimi's expected input shape
            torch_samples = torch_samples.permute(1, 2, 0)
            
            # Decode using standard Mimi decoder
            with torch.no_grad():
                audio = self.audio_tokenizer.decode(torch_samples)
                audio = audio.squeeze(0).squeeze(0)
                mimi_decoding_time = time.perf_counter() - t_mimi_start
                
                # Apply SilentCipher watermark if available and not bypassed
                t_wm_start = time.perf_counter()
                if not no_watermark and self.watermarker is not None:
                    print(f" Done.\n[MLX] Applying SilentCipher acoustic watermark...", end="", file=log_file, flush=True)
                    wm_key = watermark_key if watermark_key is not None else MISO_TTS_WATERMARK
                    audio, wm_sample_rate = watermark(self.watermarker, audio, self.sample_rate, wm_key)
                    audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
                else:
                    if no_watermark:
                        print(f" Done.\n[MLX] Bypassing SilentCipher watermarker as requested.", end="", file=log_file, flush=True)
                watermarking_time = time.perf_counter() - t_wm_start
                
            print(" Done.", file=log_file, flush=True)
            audio_output = mx.array(audio.cpu().numpy(), dtype=mx.float32)
        else:
            # Returning dummy array in fallback mode
            mimi_decoding_time = time.perf_counter() - t_mimi_start
            watermarking_time = 0.0
            audio_output = mx.zeros((len(samples) * 1920,), dtype=mx.float32)
            
        total_generate_time = time.perf_counter() - t_gen_start
        
        self.last_run_stats = {
            "prompt_tokenization_time": prompt_tokenization_time,
            "warmup_time": warmup_time,
            "generation_loop_time": generation_loop_time,
            "first_step_time": first_step_time,
            "subsequent_steps_time": subsequent_steps_time,
            "avg_subsequent_step_time": subsequent_steps_time / (len(samples) - 1) if len(samples) > 1 else 0.0,
            "mimi_decoding_time": mimi_decoding_time,
            "watermarking_time": watermarking_time,
            "total_generate_time": total_generate_time,
            "steps_count": len(samples),
        }
        
        return audio_output
