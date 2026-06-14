import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    backbone_layers: int = 32
    backbone_heads: int = 32
    backbone_kv_heads: int = 8
    backbone_dim: int = 4096
    backbone_intermediate_dim: int = 14336

    decoder_layers: int = 8
    decoder_heads: int = 24
    decoder_kv_heads: int = 6
    decoder_dim: int = 1536
    decoder_intermediate_dim: int = 6912

    text_vocab_size: int = 128256
    audio_vocab_size: int = 2051
    audio_num_codebooks: int = 32
    rope_base: float = 500000.0
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = x.square().mean(-1, keepdims=True)
        return x * mx.rsqrt(variance + self.eps) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dims: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dims, intermediate_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(intermediate_dim, dims, bias=False)  # Down projection
        self.w3 = nn.Linear(dims, intermediate_dim, bias=False)  # Up projection

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Llama3ScaledRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 500000,
        scale_factor: int = 32,
        low_freq_factor: int = 1,
        high_freq_factor: int = 4,
        old_context_len: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        
        self.rope_init()

    def rope_init(self):
        freqs = 1.0 / (
            self.base
            ** (mx.arange(0, self.dim, 2)[: (self.dim // 2)].astype(mx.float32) / self.dim)
        )
        theta = self.apply_scaling(
            freqs,
            self.scale_factor,
            self.low_freq_factor,
            self.high_freq_factor,
            self.old_context_len,
        )
        self.theta = mx.array(theta, dtype=mx.float32)
        self.build_rope_cache(self.max_seq_len)

    def apply_scaling(
        self,
        freqs: mx.array,
        scale_factor: int,
        low_freq_factor: int,
        high_freq_factor: int,
        old_context_len: int,
    ):
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs.tolist():
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return new_freqs

    def build_rope_cache(self, max_seq_len: int):
        seq_idx = mx.arange(max_seq_len, dtype=mx.float32)
        idx_theta = seq_idx[:, None] * self.theta[None, :]
        cos = mx.cos(idx_theta)
        sin = mx.sin(idx_theta)
        self.cache = mx.stack([cos, sin], axis=-1)

    def __call__(self, x: mx.array, input_pos: Optional[mx.array] = None) -> mx.array:
        B, H, L, D = x.shape
        if input_pos is None:
            rope_cache = self.cache[:L]
            rope_cache = mx.expand_dims(mx.expand_dims(rope_cache, axis=0), axis=0)
        else:
            rope_cache = self.cache[input_pos]
            if input_pos.ndim == 1:
                rope_cache = mx.expand_dims(mx.expand_dims(rope_cache, axis=0), axis=0)
            elif input_pos.ndim == 2:
                rope_cache = mx.expand_dims(rope_cache, axis=1)
                
        xshaped = x.reshape(B, H, L, D // 2, 2)
        cos = rope_cache[..., 0]
        sin = rope_cache[..., 1]
        
        x_out_real = xshaped[..., 0] * cos - xshaped[..., 1] * sin
        x_out_imag = xshaped[..., 1] * cos + xshaped[..., 0] * sin
        
        x_out = mx.stack([x_out_real, x_out_imag], axis=-1)
        return x_out.reshape(B, H, L, D)


class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int, num_kv_heads: int, rope_base: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dims // num_heads

        self.wq = nn.Linear(dims, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dims, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dims, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dims, bias=False)

        self.rope = Llama3ScaledRoPE(self.head_dim, base=int(rope_base))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        input_pos: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        queries = self.wq(x)
        keys = self.wk(x)
        values = self.wv(x)

        # Reshape to (B, L, H, D)
        queries = queries.reshape(B, L, self.num_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim)

        # Transpose to (B, H, L, D)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # Apply Rotary Position Embeddings (RoPE)
        queries = self.rope(queries, input_pos=input_pos)
        keys = self.rope(keys, input_pos=input_pos)

        # Update Attention KV Cache
        if cache is not None:
            k_cache, v_cache = cache
            keys = mx.concatenate([k_cache, keys], axis=2)
            values = mx.concatenate([v_cache, values], axis=2)
        new_cache = (keys, values)

        # Grouped-Query Attention (GQA) replication if kv_heads < num_heads
        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            keys = mx.repeat(keys, repeats, axis=1)
            values = mx.repeat(values, repeats, axis=1)

        # Compute scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * scale
        
        if mask is not None:
            scores = scores + mask

        probs = mx.softmax(scores, axis=-1)
        output = mx.matmul(probs, values)

        # Reshape and project output back to model dimension
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, num_kv_heads: int, intermediate_dim: int, rope_base: float, eps: float):
        super().__init__()
        self.attention_norm = RMSNorm(dims, eps=eps)
        self.attention = Attention(dims, num_heads, num_kv_heads, rope_base)
        self.ffn_norm = RMSNorm(dims, eps=eps)
        self.feed_forward = FeedForward(dims, intermediate_dim)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        input_pos: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        r, new_cache = self.attention(self.attention_norm(x), mask=mask, cache=cache, input_pos=input_pos)
        h = x + r
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, new_cache


class LlamaTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dims: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_dim: int,
        rope_base: float,
        eps: float,
    ):
        super().__init__()
        self.layers = [
            TransformerBlock(dims, num_heads, num_kv_heads, intermediate_dim, rope_base, eps)
            for _ in range(num_layers)
        ]
        self.norm = RMSNorm(dims, eps=eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        caches: Optional[List[Tuple[mx.array, mx.array]]] = None,
        input_pos: Optional[mx.array] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        new_caches = []
        h = x
        for idx, layer in enumerate(self.layers):
            cache = caches[idx] if caches is not None else None
            h, new_cache = layer(h, mask=mask, cache=cache, input_pos=input_pos)
            new_caches.append(new_cache)
        return self.norm(h), new_caches


class MisoTTSModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Backbone 8B Transformer (minus embeddings/output heads)
        self.backbone = LlamaTransformer(
            num_layers=args.backbone_layers,
            dims=args.backbone_dim,
            num_heads=args.backbone_heads,
            num_kv_heads=args.backbone_kv_heads,
            intermediate_dim=args.backbone_intermediate_dim,
            rope_base=args.rope_base,
            eps=args.norm_eps,
        )

        # Decoder 300M Transformer (minus embeddings/output heads)
        self.decoder = LlamaTransformer(
            num_layers=args.decoder_layers,
            dims=args.decoder_dim,
            num_heads=args.decoder_heads,
            num_kv_heads=args.decoder_kv_heads,
            intermediate_dim=args.decoder_intermediate_dim,
            rope_base=args.rope_base,
            eps=args.norm_eps,
        )

        # Embeddings & Projections
        self.text_embeddings = nn.Embedding(args.text_vocab_size, args.backbone_dim)
        self.audio_embeddings = nn.Embedding(args.audio_vocab_size * args.audio_num_codebooks, args.backbone_dim)
        self.projection = nn.Linear(args.backbone_dim, args.decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(args.backbone_dim, args.audio_vocab_size, bias=False)
        
        # Audio head multi-codebook weights (num_codebooks-1, decoder_dim, audio_vocab_size)
        self.audio_head = mx.zeros((args.audio_num_codebooks - 1, args.decoder_dim, args.audio_vocab_size))

    def _embed_tokens(self, tokens: mx.array) -> mx.array:
        """
        Embeds standard interleaved tokens into model space.
        tokens: (B, L, audio_num_codebooks + 1)
        """
        # Text embedding (extract last channel)
        text_embeds = self.text_embeddings(tokens[:, :, -1])
        text_embeds = mx.expand_dims(text_embeds, axis=-2)  # (B, L, 1, backbone_dim)

        # Audio embedding
        audio_tokens = tokens[:, :, :-1]
        # Offset each codebook index to match the unified vocab mapping
        codebook_offsets = self.args.audio_vocab_size * mx.arange(self.args.audio_num_codebooks)
        audio_tokens = audio_tokens + codebook_offsets
        
        B, L, C = audio_tokens.shape
        audio_embeds = self.audio_embeddings(audio_tokens.reshape(-1))
        audio_embeds = audio_embeds.reshape(B, L, C, -1)  # (B, L, audio_num_codebooks, backbone_dim)

        # Concat audio codebook embeddings and text embeddings along codebook axis
        return mx.concatenate([audio_embeds, text_embeds], axis=-2)

    def _embed_audio(self, codebook: int, tokens: mx.array) -> mx.array:
        """Embeds a single specific codebook's tokens."""
        offset_tokens = tokens + (codebook * self.args.audio_vocab_size)
        return self.audio_embeddings(offset_tokens)

    def __call__(
        self,
        tokens: mx.array,
        tokens_mask: mx.array,
        input_pos: mx.array,
        backbone_caches: Optional[List[Tuple[mx.array, mx.array]]] = None,
        decoder_caches: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, mx.array, List[Tuple[mx.array, mx.array]], List[Tuple[mx.array, mx.array]]]:
        """
        Runs one step of frame generation.
        tokens: (B, L, C+1)
        tokens_mask: (B, L, C+1)
        """
        # 1. Embed and sum codebooks
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * mx.expand_dims(tokens_mask, axis=-1)
        h = masked_embeds.sum(axis=-2)  # Sum across codebooks -> (B, L, backbone_dim)

        # 2. Forward pass through Backbone 8B
        # Build causal attention mask if doing multi-token prompting
        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])

        h, new_backbone_caches = self.backbone(h, mask=mask, caches=backbone_caches, input_pos=input_pos)

        # 3. Get codebook 0 logits
        last_h = h[:, -1, :]  # Extract last sequence frame -> (B, backbone_dim)
        c0_logits = self.codebook0_head(last_h)

        return c0_logits, last_h, new_backbone_caches, decoder_caches
