# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch port of the Magenta RealTime 2 Depthformer (encoder + multivariate decoder)."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from . import layers as L


@dataclass
class SpecDims:
    num_layers: int
    model_dims: int
    hidden_dims: int
    num_heads: int
    dim_per_head: int


@dataclass
class DepthformerConfig:
    # encoder (conditioning embedder) dims
    encoder_model_dims: int
    musiccoca_rvq: int            # 12
    musiccoca_per_rvq_vocab: int  # 1031
    musiccoca_embed_dim: int      # 768
    regular_num_embeddings_per_channel: list  # per regular channel vocab sizes
    regular_num_channels: int     # 132
    # decoder
    temporal: SpecDims
    depth: SpecDims
    temporal_max_past: int        # 41 (small) / 25 (base)
    depth_max_past: int           # 12
    num_sinks: int                # 1
    num_codebooks: int            # 12
    codebook_size: int            # 1024
    num_reserved_tokens: int      # 6
    vocab_size: int               # 12294
    soft_cap_logits: float = 30.0


def _mean_f32(x, axis):
    return x.float().mean(axis).to(x.dtype)


class TransformerStack(nn.Module):
    """Stack of primer-hybrid transformer layers; optional streaming cross-attn."""

    def __init__(self, spec: SpecDims, max_past, num_sinks, use_cross, source_dim=None):
        super().__init__()
        self.use_cross = use_cross
        self.layers = nn.ModuleList()
        for _ in range(spec.num_layers):
            blk = nn.ModuleDict()
            blk["self_attention"] = L.SelfAttention(
                spec.model_dims, spec.num_heads, spec.dim_per_head, max_past,
                num_sinks=num_sinks)
            if use_cross:
                blk["cross_attention"] = L.CrossAttention(
                    spec.model_dims, source_dim, spec.num_heads, spec.dim_per_head,
                    max_past, num_sinks=num_sinks)
            blk["ffn"] = L.FFN(spec.model_dims, spec.hidden_dims)
            self.layers.append(blk)

    def forward(self, x, source=None):
        for blk in self.layers:
            x = blk["self_attention"](x)
            if self.use_cross:
                x = blk["cross_attention"](x, source)
            x = blk["ffn"](x)
        return x

    # ---- streaming ----
    def init_state(self, batch, device, dtype, source=None):
        st = []
        for blk in self.layers:
            s = {"self": blk["self_attention"].init_state(batch, device, dtype)}
            if self.use_cross:
                s["source_kv"] = (None, None)
            st.append(s)
        return st

    def step(self, x, state, source_step=None):
        # source_step: newly-encoded source frame [b,1,source_dim] to append to KV.
        for blk, s in zip(self.layers, state):
            x = blk["self_attention"].step(x, s["self"])
            if self.use_cross:
                ca = blk["cross_attention"]
                k, v = ca._kv(source_step)
                pk, pv = s["source_kv"]
                k = k if pk is None else torch.cat([pk, k], dim=1)
                v = v if pv is None else torch.cat([pv, v], dim=1)
                keep = ca.max_past_horizon + 1
                if k.shape[1] > keep:
                    k = k[:, -keep:]
                    v = v[:, -keep:]
                s["source_kv"] = (k, v)
                x = ca.step(x, (k, v))
            x = blk["ffn"](x)
        return x

    def step_fn(self, x, self_kv, cross_kv, source_frame):
        """Functional, export-clean per-frame step over all layers.

        self_kv / cross_kv: lists of (k, v) per layer ([b,T,nh,uph], T may be 0).
        Returns (out[b,1,d], new_self_kv, new_cross_kv) with untrimmed KV."""
        new_self = []
        new_cross = []
        for i, blk in enumerate(self.layers):
            x, k, v = blk["self_attention"].step_fn(x, self_kv[i][0], self_kv[i][1])
            new_self.append((k, v))
            if self.use_cross:
                ca = blk["cross_attention"]
                sk, sv = ca._kv(source_frame)
                sk = torch.cat([cross_kv[i][0], sk], dim=1)
                sv = torch.cat([cross_kv[i][1], sv], dim=1)
                new_cross.append((sk, sv))
                x = ca.attend_fn(x, sk, sv)
            x = blk["ffn"](x)
        return x, new_self, new_cross


class EncoderEmbedding(nn.Module):
    """Embeds the conditioning block [b,t,num_channels] -> source [b,t,enc_dim]."""

    def __init__(self, cfg: DepthformerConfig):
        super().__init__()
        self.cfg = cfg
        m = cfg.musiccoca_rvq
        self.m = m
        # mulan branch
        self.register_buffer("mulan_offset", torch.arange(m) * cfg.musiccoca_per_rvq_vocab)
        self.mulan_dequantizer = nn.Parameter(
            torch.zeros(m * cfg.musiccoca_per_rvq_vocab, cfg.musiccoca_embed_dim))
        self.mulan_adapter = L.JaxLinear(cfg.musiccoca_embed_dim, cfg.encoder_model_dims, use_bias=False)
        # regular branch (MultiChannelEmbedding)
        per = cfg.regular_num_embeddings_per_channel
        total = sum(per)
        total = (total + 127) // 128 * 128
        self.regular_embedding = nn.Parameter(torch.zeros(total, cfg.encoder_model_dims))
        offs = [0]
        for p in per[:-1]:
            offs.append(offs[-1] + p)
        self.register_buffer("regular_offsets", torch.tensor(offs, dtype=torch.long))
        self.encoder_ln = L.LayerNorm(cfg.encoder_model_dims)

    def forward(self, x):
        # x: [b,t,num_channels] int
        m = self.m
        mulan = x[..., :m]                  # [b,t,m]
        regular = x[..., m:]                # [b,t,132]
        off = self.mulan_offset.to(x.device)
        idx = mulan + off                   # [b,t,m]
        emb = self.mulan_dequantizer[idx]   # [b,t,m,768]
        emb = emb.sum(dim=-2)               # [b,t,768]
        mulan_out = self.mulan_adapter(emb)  # [b,t,enc_dim]
        # regular
        ridx = regular + self.regular_offsets.to(x.device)
        remb = self.regular_embedding[ridx]  # [b,t,132,enc_dim]
        regular_out = _mean_f32(remb, axis=-2)  # mean over channels in fp32
        # branch combine via MEAN
        src = (mulan_out + regular_out) / 2.0
        return self.encoder_ln(src)


class MultivariateDecoder(nn.Module):
    def __init__(self, cfg: DepthformerConfig):
        super().__init__()
        self.cfg = cfg
        td, dd = cfg.temporal, cfg.depth
        self.embedding = nn.Parameter(torch.zeros(cfg.vocab_size, td.model_dims))
        self.embed_scale = math.sqrt(td.model_dims)
        self.temporal_body = TransformerStack(
            td, cfg.temporal_max_past, cfg.num_sinks, use_cross=True,
            source_dim=cfg.encoder_model_dims)
        # depth input adapter (Dense temporal->depth, no bias) or identity
        if td.model_dims != dd.model_dims:
            self.depth_input_adapter = L.JaxLinear(td.model_dims, dd.model_dims, use_bias=False)
        else:
            self.depth_input_adapter = None
        self.depth_body = TransformerStack(
            dd, cfg.depth_max_past, num_sinks=0, use_cross=False)
        self.final_ln = L.LayerNorm(dd.model_dims)
        self.to_logits = L.JaxLinear(dd.model_dims, cfg.vocab_size, use_bias=True)

    def embed(self, tokens):
        # tokens: [...,] int -> [..., td_dim]
        return self.embedding[tokens] * self.embed_scale

    def _depth_forward(self, depth_inputs):
        # depth_inputs: [N, Q, td_dim]
        h = depth_inputs
        if self.depth_input_adapter is not None:
            h = self.depth_input_adapter(h)
        h = self.depth_body(h)
        h = self.final_ln(h)
        logits = self.to_logits(h)
        return logits

    def forward(self, target, source):
        """Teacher-forced. target: [b,T,Q] int; source: [b,Tc,enc_dim]. -> logits [b,T,Q,vocab]."""
        cfg = self.cfg
        b, T, Q = target.shape
        sos = target.new_zeros((b, 1, Q))  # sos_id=0
        x = torch.cat([sos, target], dim=1)             # [b,T+1,Q]
        embedded = self.embed(x)                         # [b,T+1,Q,D]
        temporal_inputs = _mean_f32(embedded, axis=-2)[:, :-1]  # [b,T,D]
        temporal_outputs = self.temporal_body(temporal_inputs, source)  # [b,T,D]
        depth_inputs = torch.cat(
            [temporal_outputs[..., None, :], embedded[:, 1:, :-1]], dim=-2)  # [b,T,Q,D]
        N = b * T
        logits = self._depth_forward(depth_inputs.reshape(N, Q, -1)).reshape(b, T, Q, -1)
        if cfg.soft_cap_logits is not None:
            c = cfg.soft_cap_logits
            logits = torch.tanh(logits / c) * c
        return logits

    # ---- functional (AOTI-compilable) streaming -------------------------
    def temporal_step_fn(self, prev_frame, self_kv, cross_kv, source_frame):
        """Functional temporal step: prev_frame[b,1,Q] -> (temporal_out, kv...)."""
        embedded = self.embed(prev_frame)
        ti = _mean_f32(embedded, axis=-2)
        return self.temporal_body.step_fn(ti, self_kv, cross_kv, source_frame)

    def depth_step_fn(self, depth_input, depth_kv):
        """Functional depth step: depth_input[b,1,Dt] + kv -> (logits, new_kv)."""
        h = depth_input
        if self.depth_input_adapter is not None:
            h = self.depth_input_adapter(h)
        h, nk, _ = self.depth_body.step_fn(h, depth_kv, [], None)
        h = self.final_ln(h)
        return self._soft_cap(self.to_logits(h)), nk

    def init_streaming_f(self, batch, device, dtype=torch.float32):
        td = self.cfg.temporal
        z = torch.zeros(batch, 0, td.num_heads, td.dim_per_head, device=device, dtype=dtype)
        kv = [(z, z) for _ in range(td.num_layers)]
        return {
            "self": [(k, v) for k, v in kv],
            "cross": [(k, v) for k, v in kv],
            "prev": torch.zeros((batch, 1, self.cfg.num_codebooks), dtype=torch.long, device=device),
        }

    def step_f(self, state, source_frame, sampler=None, forced=None,
               temporal_step=None, depth_step=None):
        """One functional frame. temporal_step/depth_step override the eager fns
        (e.g. with AOTI-compiled callables). Updates state in place; returns [b,1,Q]."""
        cfg = self.cfg
        tstep = temporal_step or self.temporal_step_fn
        dstep = depth_step or self.depth_step_fn
        to, new_self, new_cross = tstep(state["prev"], state["self"], state["cross"], source_frame)
        keep = cfg.temporal_max_past + 1
        state["self"] = [(k[:, -keep:], v[:, -keep:]) for k, v in new_self]
        state["cross"] = [(k[:, -keep:], v[:, -keep:]) for k, v in new_cross]
        dd = cfg.depth
        z = torch.zeros(to.shape[0], 0, dd.num_heads, dd.dim_per_head, device=to.device, dtype=to.dtype)
        depth_kv = [(z, z) for _ in range(dd.num_layers)]
        depth_input = to
        samples = []
        for q in range(cfg.num_codebooks):
            logits, depth_kv = dstep(depth_input, depth_kv)
            lo = cfg.num_reserved_tokens + q * cfg.codebook_size
            hi = lo + cfg.codebook_size
            tok = forced[..., q] if forced is not None else sampler(logits.float(), q, lo, hi)
            samples.append(tok)
            depth_input = self.embed(tok)
        frame = torch.stack(samples, dim=-1)
        state["prev"] = frame
        return frame

    # ---- streaming generation -------------------------------------------
    def init_streaming(self, batch, device, dtype=torch.float32):
        return {
            "temporal": self.temporal_body.init_state(batch, device, dtype, source=True),
            "prev": torch.zeros((batch, 1, self.cfg.num_codebooks), dtype=torch.long, device=device),
        }

    def _soft_cap(self, logits):
        c = self.cfg.soft_cap_logits
        return torch.tanh(logits / c) * c if c is not None else logits

    def _depth_step_logits(self, depth_inputs, depth_state):
        h = depth_inputs
        if self.depth_input_adapter is not None:
            h = self.depth_input_adapter(h)
        h = self.depth_body.step(h, depth_state)
        h = self.final_ln(h)
        return self.to_logits(h)

    def step(self, state, source_step, sampler=None, forced_frame=None):
        """One streaming frame.

        state: from init_streaming (updated in place).
        source_step: [b,1,enc_dim] encoded conditioning for this frame.
        sampler: fn(logits[b,1,vocab], rvq_index, valid_lo, valid_hi) -> token[b,1] long.
        forced_frame: [b,1,Q] tokens to force (teacher forcing) instead of sampling.
        Returns sampled frame [b,1,Q] long.
        """
        cfg = self.cfg
        prev = state["prev"]
        embedded = self.embed(prev)                      # [b,1,Q,D]
        temporal_inputs = _mean_f32(embedded, axis=-2)   # [b,1,D]
        temporal_out = self.temporal_body.step(temporal_inputs, state["temporal"], source_step)
        depth_state = self.depth_body.init_state(prev.shape[0], prev.device, temporal_out.dtype)
        depth_inputs = temporal_out
        samples = []
        for q in range(cfg.num_codebooks):
            logits = self._soft_cap(self._depth_step_logits(depth_inputs, depth_state)).float()
            lo = cfg.num_reserved_tokens + q * cfg.codebook_size
            hi = lo + cfg.codebook_size
            if forced_frame is not None:
                tok = forced_frame[..., q]               # [b,1]
            else:
                tok = sampler(logits, q, lo, hi)         # [b,1]
            samples.append(tok)
            depth_inputs = self.embed(tok.unsqueeze(-1)).squeeze(-2) if tok.dim() == 2 else self.embed(tok)
        frame = torch.stack(samples, dim=-1)             # [b,1,Q]
        state["prev"] = frame
        return frame, logits  # last logits returned for debugging

    def streaming_logits(self, target, source):
        """Re-derive per-(t,q) logits via the streaming step path with forced tokens.
        Used to validate the KV-cache step path against teacher forcing."""
        b, T, Q = target.shape
        state = self.init_streaming(b, target.device)
        all_logits = []
        for t in range(T):
            frame_logits = []
            prev = state["prev"]
            embedded = self.embed(prev)
            temporal_inputs = _mean_f32(embedded, axis=-2)
            temporal_out = self.temporal_body.step(temporal_inputs, state["temporal"], source[:, t:t+1])
            depth_state = self.depth_body.init_state(b, target.device, temporal_out.dtype)
            depth_inputs = temporal_out
            for q in range(Q):
                logits = self._soft_cap(self._depth_step_logits(depth_inputs, depth_state))
                frame_logits.append(logits)
                tok = target[:, t:t+1, q]
                depth_inputs = self.embed(tok)
            state["prev"] = target[:, t:t+1, :]
            all_logits.append(torch.stack(frame_logits, dim=2))  # [b,1,Q,vocab]
        return torch.cat(all_logits, dim=1)


class Depthformer(nn.Module):
    def __init__(self, cfg: DepthformerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = EncoderEmbedding(cfg)
        self.decoder = MultivariateDecoder(cfg)

    def encode(self, cond):
        return self.encoder(cond)

    def forward(self, cond, target):
        source = self.encode(cond)
        return self.decoder(target, source)


# --- config builders ---------------------------------------------------------

_REGULAR_PER_CHANNEL = [11] * 128 + [9] * 1 + [47] * 2 + [15] * 1  # 132 channels

_COMMON = dict(
    musiccoca_rvq=12, musiccoca_per_rvq_vocab=1031, musiccoca_embed_dim=768,
    regular_num_embeddings_per_channel=_REGULAR_PER_CHANNEL, regular_num_channels=132,
    depth_max_past=12, num_sinks=1, num_codebooks=12, codebook_size=1024,
    num_reserved_tokens=6, vocab_size=12294, soft_cap_logits=30.0,
)


def config_for(size: str) -> DepthformerConfig:
    if size == "mrt2_small":
        return DepthformerConfig(
            encoder_model_dims=256,
            temporal=SpecDims(12, 1024, 4096, 8, 128),
            depth=SpecDims(2, 768, 3072, 6, 128),
            temporal_max_past=41,
            **_COMMON,
        )
    if size == "mrt2_base":
        return DepthformerConfig(
            encoder_model_dims=1024,
            temporal=SpecDims(20, 3072, 8192, 24, 128),
            depth=SpecDims(6, 1024, 4096, 8, 128),
            temporal_max_past=25,
            **_COMMON,
        )
    raise ValueError(f"unknown size {size}")
