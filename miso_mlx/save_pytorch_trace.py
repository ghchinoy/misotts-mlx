import os
import sys
import torch
import numpy as np
from pathlib import Path

project_root = Path("/Users/ghchinoy/projects/misotts")
sys.path.append(str(project_root / "sources" / "MisoTTS"))

from models import Model, MISO_TTS_8B_CONFIG
from generator import Generator

print("Loading PyTorch model...")
pt_model = Model(MISO_TTS_8B_CONFIG)
from safetensors.torch import load_file
model_file = "/Users/ghchinoy/.cache/huggingface/hub/models--MisoLabs--MisoTTS/snapshots/ef6b096cc35d3cde6aa0721013648416c14c36b2/model.safetensors"
pt_state_dict = load_file(model_file, device="cpu")
pt_model.load_state_dict(pt_state_dict, strict=True)
pt_model = pt_model.to(torch.bfloat16)
pt_model.eval()

pt_gen = Generator(pt_model)

text = "Hello"
speaker = 0

print("Tokenizing text segment...")
pt_tokens, pt_masks = pt_gen._tokenize_text_segment(text, speaker)

print("Running Warmup Step 1...")
pt_model.reset_caches()
pt_pos = torch.arange(0, pt_tokens.size(0)).unsqueeze(0).to(pt_tokens.device)
pt_mask = pt_gen._model.backbone_causal_mask[pt_pos, :]

with torch.no_grad():
    pt_embeds = pt_model._embed_tokens(pt_tokens.unsqueeze(0))
    pt_masked_embeds = pt_embeds * pt_masks.unsqueeze(0).unsqueeze(-1)
    pt_h = pt_masked_embeds.sum(dim=2)
    pt_backbone_out = pt_model.backbone(pt_h, input_pos=pt_pos, mask=pt_mask).to(torch.bfloat16)
    pt_last_h = pt_backbone_out[:, -1, :]
    pt_c0_logits = pt_model.codebook0_head(pt_last_h)

# Let's perform deterministic argmax sampling to make comparisons perfectly identical
pt_c0_sample = pt_c0_logits.argmax(dim=-1, keepdim=True)
print("pt_c0_sample value:", pt_c0_sample.item())

# Run Decoder for remaining codebooks
pt_c0_embed = pt_model._embed_audio(0, pt_c0_sample)
pt_curr_h = torch.cat([pt_last_h.unsqueeze(1), pt_c0_embed], dim=1)
pt_curr_pos = torch.arange(0, pt_curr_h.size(1)).unsqueeze(0)
pt_decoder_mask = pt_model.decoder_causal_mask[pt_curr_pos, :]

pt_model.decoder.reset_caches()
pt_curr_sample = pt_c0_sample.clone()

with torch.no_grad():
    for i in range(1, pt_model.config.audio_num_codebooks):
        pt_dec_input = pt_model.projection(pt_curr_h)
        pt_decoder_mask = pt_model.decoder_causal_mask[pt_curr_pos, :]
        pt_decoder_h = pt_model.decoder(pt_dec_input, input_pos=pt_curr_pos, mask=pt_decoder_mask).to(torch.bfloat16)
        pt_ci_logits = torch.mm(pt_decoder_h[:, -1, :], pt_model.audio_head[i - 1])
        pt_ci_sample = pt_ci_logits.argmax(dim=-1, keepdim=True)
        pt_ci_embed = pt_model._embed_audio(i, pt_ci_sample)
        
        pt_curr_sample = torch.cat([pt_curr_sample, pt_ci_sample], dim=1)
        pt_curr_h = pt_ci_embed
        pt_curr_pos = pt_curr_pos[:, -1:] + 1

print("Generated frame at Step 1:", pt_curr_sample.tolist()[0])

# Prepare Step 2
pt_next_token = torch.cat([pt_curr_sample, torch.zeros(1, 1).long()], dim=1).unsqueeze(1)
pt_next_mask = torch.cat([torch.ones_like(pt_curr_sample).bool(), torch.zeros(1, 1).bool()], dim=1).unsqueeze(1)
pt_pos2 = torch.tensor([[pt_tokens.size(0)]]).long()
pt_mask2 = pt_model.backbone_causal_mask[pt_pos2, :]

with torch.no_grad():
    pt_embeds2 = pt_model._embed_tokens(pt_next_token)
    pt_masked_embeds2 = pt_embeds2 * pt_next_mask.unsqueeze(-1)
    pt_h2 = pt_masked_embeds2.sum(dim=2).to(pt_embeds2.dtype)
    pt_backbone_out2 = pt_model.backbone(pt_h2, input_pos=pt_pos2, mask=pt_mask2).to(torch.bfloat16)
    pt_last_h2 = pt_backbone_out2[:, -1, :]
    pt_c0_logits2 = pt_model.codebook0_head(pt_last_h2)

pt_c0_sample2 = pt_c0_logits2.argmax(dim=-1, keepdim=True)
print("pt_c0_sample2 value (Step 2 argmax):", pt_c0_sample2.item())

# Save tensors as numpy files for MLX comparison
out_dir = Path("/Users/ghchinoy/projects/misotts/miso_mlx/pt_trace")
out_dir.mkdir(exist_ok=True)

np.save(out_dir / "pt_tokens.npy", pt_tokens.cpu().numpy())
np.save(out_dir / "pt_masks.npy", pt_masks.cpu().numpy())
np.save(out_dir / "pt_last_h.npy", pt_last_h.float().cpu().numpy())
np.save(out_dir / "pt_c0_logits.npy", pt_c0_logits.float().cpu().numpy())
np.save(out_dir / "pt_curr_sample.npy", pt_curr_sample.cpu().numpy())
np.save(out_dir / "pt_next_token.npy", pt_next_token.cpu().numpy())
np.save(out_dir / "pt_next_mask.npy", pt_next_mask.cpu().numpy())
np.save(out_dir / "pt_pos2.npy", pt_pos2.cpu().numpy())
np.save(out_dir / "pt_last_h2.npy", pt_last_h2.float().cpu().numpy())
np.save(out_dir / "pt_c0_logits2.npy", pt_c0_logits2.float().cpu().numpy())

print("PyTorch trace saved successfully!")
