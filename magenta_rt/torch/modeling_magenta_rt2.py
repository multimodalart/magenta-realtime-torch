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

"""Transformers-compatible Magenta RealTime 2 (PyTorch).

`MagentaRT2ForConditionalGeneration` is a `PreTrainedModel` wrapping the
Depthformer LLM + SpectroStream codec decoder. Generation is per-frame RVQ
autoregression with a depth transformer + streaming codec decode, exposed as
custom `generate` / `stream` methods (it does not fit `GenerationMixin`, whose
loop is a single token stream). MusicCoCa style encoding is a separate
`MusicCoCaProcessor`.
"""

import json
import os
import warnings

import numpy as np
import torch
from transformers import PreTrainedModel

from .configuration_magenta_rt2 import MagentaRT2Config
from .depthformer import Depthformer, DepthformerConfig, SpecDims
from .cudagraph import CudaGraphStreamer, GUIDANCE_CFG_WARN, _warn_high_cfg  # noqa: F401 (re-export)
from .spectrostream import SpectroStreamDecoder, codes_to_embeddings

# Force `trust_remote_code` to bundle every dependency module. transformers only
# traces `from .X import ...` (not `from . import X`), so we name them explicitly.
from .layers import JaxLinear as _ensure_layers  # noqa: F401
from .musiccoca import MusicCoCa as _ensure_musiccoca  # noqa: F401
from .processing_musiccoca import MusicCoCaProcessor as _ensure_processor  # noqa: F401
from .aoti import load_compiled_steps as _ensure_aoti  # noqa: F401

SR = 48000
FRAME_SAMPLES = 1920
STREAM_DECODE_CONTEXT = 16
STREAM_DECODE_MARGIN = 2


def discretize_cfg(value, step, max_bin):
    clamped = max(-1.0, min(7.0, value))
    return max(0, min(max_bin, int(round((clamped - (-1.0)) / step))))


def convert_from_unique_codes(tokens, codebook_size, num_reserved=6):
    return (tokens - num_reserved) % codebook_size


def _float_to_int16(samples, gain=0.5):
    samples = np.clip(gain * samples, -1, 1)
    samples = np.round((np.iinfo(np.int16).max + 0.5) * samples - 0.5)
    return samples.astype(np.int16)


def make_sampler(temperature, top_k, generator):
    """jax-parity: valid-range mask, top-k, gumbel-max. NEG is finite (avoids NaN)."""
    NEG = -1e9

    def sampler(logits, rvq_index, lo, hi):
        logits = logits.float()
        v = logits.shape[-1]
        idx = torch.arange(v, device=logits.device)
        valid = (idx >= lo) & (idx < hi)
        logits = torch.where(valid, logits, torch.full_like(logits, NEG))
        if top_k is not None and int(top_k) < v:
            k = min(max(int(top_k), 1), v)
            kth = torch.topk(logits, k, dim=-1).values[..., -1:]
            logits = torch.where(logits >= kth, logits, torch.full_like(logits, NEG))
        if temperature and float(temperature) > 0:
            u = torch.rand(logits.shape, generator=generator, device=logits.device)
            gumbel = -torch.log(-torch.log(u.clamp(1e-10, 1 - 1e-7)))
            logits = logits + gumbel * float(temperature)
        return logits.argmax(dim=-1)

    return sampler


def _depthformer_config(c: MagentaRT2Config) -> DepthformerConfig:
    return DepthformerConfig(
        encoder_model_dims=c.encoder_model_dims,
        musiccoca_rvq=c.musiccoca_rvq,
        musiccoca_per_rvq_vocab=c.musiccoca_per_rvq_vocab,
        musiccoca_embed_dim=c.musiccoca_embed_dim,
        regular_num_embeddings_per_channel=list(c.regular_num_embeddings_per_channel),
        regular_num_channels=c.regular_num_channels,
        temporal=SpecDims(*c.temporal),
        depth=SpecDims(*c.depth),
        temporal_max_past=c.temporal_max_past,
        depth_max_past=c.depth_max_past,
        num_sinks=c.num_sinks,
        num_codebooks=c.num_codebooks,
        codebook_size=c.codebook_size,
        num_reserved_tokens=c.num_reserved_tokens,
        vocab_size=c.vocab_size,
        soft_cap_logits=c.soft_cap_logits,
    )


def _empty_codec(shapes):
    return SpectroStreamDecoder({k: torch.zeros(v) for k, v in shapes.items()})


class MagentaRT2PreTrainedModel(PreTrainedModel):
    config_class = MagentaRT2Config
    base_model_prefix = "magenta_rt2"
    _no_split_modules = ["TransformerStack", "SpectroStreamDecoder"]
    main_input_name = "style_tokens"

    def _init_weights(self, module):
        pass  # weights come from the checkpoint; no random init needed


class MagentaRT2ForConditionalGeneration(MagentaRT2PreTrainedModel):
    """Depthformer LLM + SpectroStream codec. Custom streaming generation."""

    def __init__(self, config):
        super().__init__(config)
        self.depthformer = Depthformer(_depthformer_config(config))
        shapes = getattr(config, "codec_param_shapes", None)
        if not shapes:  # dev/package fallback: file next to the module
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "codec_shapes.json")) as f:
                shapes = json.load(f)
        self.codec = _empty_codec(shapes)
        self.register_buffer("quant", torch.zeros(64, config.codebook_size, 256))
        self.num_musiccoca = config.musiccoca_rvq
        self.num_notes = config.num_notes
        self.num_drums = config.num_drums
        self.codebook_size = config.codebook_size
        self.num_reserved_tokens = config.num_reserved_tokens
        self.sample_rate = config.sample_rate
        self._temporal_step = None
        self._depth_step = None
        self.processor = None
        self.post_init()

    # ---- helpers ----
    @property
    def _dt(self):
        return next(self.depthformer.parameters()).dtype

    @property
    def _dev(self):
        return next(self.depthformer.parameters()).device

    def set_processor(self, processor):
        """Attach a MusicCoCaProcessor so `generate(style="text"|audio)` works."""
        self.processor = processor
        return self

    def load_processor(self, repo_id="magenta-torch/magenta-rt-musiccoca-torch", device=None):
        """Load + attach the MusicCoCa style processor (text/audio -> RVQ tokens)."""
        from .processing_musiccoca import MusicCoCaProcessor
        self.processor = MusicCoCaProcessor.from_pretrained(repo_id, device=device or str(self._dev))
        return self

    # ---- speedups ----
    def compile_steps(self, dynamic=True, **kwargs):
        """`torch.compile` the hot per-frame step paths (dynamic shapes for the
        growing KV cache; one-time warmup). Portable — works on any CUDA GPU,
        unlike the prebuilt AOTI artifacts, which are GPU-arch-specific."""
        dec = self.depthformer.decoder
        dec.temporal_body.step = torch.compile(dec.temporal_body.step, dynamic=dynamic, **kwargs)
        dec._depth_step_logits = torch.compile(dec._depth_step_logits, dynamic=dynamic, **kwargs)
        return self

    # ---- AOTI: export your own ahead-of-time graphs (skip runtime compile) ----
    def export_aoti(self, out_dir):
        """AOTInductor-compile the per-frame step to `out_dir` (temporal.pt2 +
        depth.pt2). Run once on your target GPU; the graphs are architecture-
        specific. Reload with `load_aoti(out_dir)` to generate with no compile-time."""
        import os
        from . import aoti
        os.makedirs(out_dir, exist_ok=True)
        dec = self.depthformer.decoder
        torch._inductor.aoti_compile_and_package(
            aoti.export_temporal(dec), package_path=os.path.join(out_dir, "temporal.pt2"))
        torch._inductor.aoti_compile_and_package(
            aoti.export_depth(dec), package_path=os.path.join(out_dir, "depth.pt2"))
        return out_dir

    def load_aoti(self, out_dir):
        """Load AOTI step graphs produced by `export_aoti` and use them for generation."""
        import os
        t = torch._inductor.aoti_load_package(os.path.join(out_dir, "temporal.pt2"))
        d = torch._inductor.aoti_load_package(os.path.join(out_dir, "depth.pt2"))
        return self.apply_compiled(t, d)

    def apply_compiled(self, temporal_step=None, depth_step=None):
        if temporal_step is not None:
            self._temporal_step = temporal_step
        if depth_step is not None:
            self._depth_step = depth_step
        return self

    def load_compiled(self, repo_id=None, local_dir=None):
        from . import aoti
        t, d = aoti.load_compiled_steps(self.depthformer.decoder, repo_id=repo_id, local_dir=local_dir)
        return self.apply_compiled(t, d)

    # ---- conditioning ----
    def _tokenize_style(self, style, pca_coeffs=None):
        if self.processor is None:
            raise ValueError("No MusicCoCaProcessor attached; pass `style` as a list of "
                             f"{self.num_musiccoca} RVQ token ids, or call set_processor().")
        return np.asarray(self.processor.tokenize(self.processor.embed(style), pca_coeffs)).tolist()

    def set_pca(self, components):
        if self.processor is None:
            raise ValueError("No MusicCoCaProcessor attached; call load_processor() first.")
        return self.processor.set_pca(components)

    def compute_pca(self, texts, k=8):
        if self.processor is None:
            raise ValueError("No MusicCoCaProcessor attached; call load_processor() first.")
        return self.processor.compute_pca(texts, k)

    def _conditioning(self, style_tokens, notes, drums, cfgs):
        offset = self.num_reserved_tokens + 1
        vals = list(style_tokens) + list(notes) + list(drums) + list(cfgs)
        arr = np.array(vals, dtype=np.int64) + offset
        return torch.from_numpy(arr).view(1, 1, -1).to(self._dev)

    def _resolve_conditioning(self, style, notes, drums, cfg_musiccoca, cfg_notes, cfg_drums, pca_coeffs=None):
        c = self.config
        if style is None:
            style_tokens = [-1] * self.num_musiccoca
        elif isinstance(style, (list, np.ndarray)) and np.asarray(style).ndim == 1 \
                and np.asarray(style).dtype.kind in "iu" and len(style) == self.num_musiccoca:
            style_tokens = list(style)
        else:
            style_tokens = self._tokenize_style(style, pca_coeffs)
        style_tokens = (list(style_tokens) + [-1] * self.num_musiccoca)[:self.num_musiccoca]
        notes = notes if notes is not None else [-1] * self.num_notes
        drums = drums if drums is not None else [-1] * self.num_drums
        cfgs = [
            discretize_cfg(c.cfg_musiccoca if cfg_musiccoca is None else cfg_musiccoca, 0.2, 40),
            discretize_cfg(c.cfg_notes if cfg_notes is None else cfg_notes, 0.2, 40),
            discretize_cfg(c.cfg_drums if cfg_drums is None else cfg_drums, 1.0, 8),
        ]
        return self._conditioning(style_tokens, notes, drums, cfgs)

    def _guidance_source(self, style, notes, drums, cfg_musiccoca, cfg_notes, pca_coeffs=None):
        """OPTIONAL classifier-free-guidance conditioning (the native MLX/.mlxfn path).
        Builds a 3-row batch [positive, neg_musiccoca, neg_notes] + per-component scales.
        cfg tokens are neutralized (guidance replaces them); negatives mask style / notes.
        Returns (source[3,Tc,enc], (cfg_musiccoca, cfg_notes))."""
        c = self.config
        if style is None:
            st = [-1] * self.num_musiccoca
        elif isinstance(style, (list, np.ndarray)) and np.asarray(style).ndim == 1 \
                and np.asarray(style).dtype.kind in "iu" and len(style) == self.num_musiccoca:
            st = list(style)
        else:
            st = self._tokenize_style(style, pca_coeffs)
        st = (list(st) + [-1] * self.num_musiccoca)[:self.num_musiccoca]
        notes = notes if notes is not None else [-1] * self.num_notes
        drums = drums if drums is not None else [-1] * self.num_drums
        CM = [-1, -1, -1]                                   # neutralized cfg tokens
        cond   = self._conditioning(st, notes, drums, CM)
        neg_mc = self._conditioning([-1] * self.num_musiccoca, notes, drums, CM)
        neg_n  = self._conditioning(st, [-1] * self.num_notes, drums, CM)
        source = self.depthformer.encode(torch.cat([cond, neg_mc, neg_n], 0)).to(self._dt)
        cfg_mc = c.cfg_musiccoca if cfg_musiccoca is None else cfg_musiccoca
        cfg_n = c.cfg_notes if cfg_notes is None else cfg_notes
        _warn_high_cfg(cfg_mc, cfg_n)
        return source, (float(cfg_mc), float(cfg_n))

    # ---- codec ----
    def _decode_stream(self, history, emitted, context=STREAM_DECODE_CONTEXT,
                       margin=STREAM_DECODE_MARGIN, flush=False):
        m = 0 if flush else margin
        Ttot = history.shape[1]
        emittable = (Ttot - 1) - m
        avail = emittable - emitted
        if avail <= 0:
            return history.new_zeros((1, 0, 2), dtype=self._dt), emitted
        w0 = max(0, Ttot - (avail + m + context + 1))
        window = history[:, w0:]
        codes = convert_from_unique_codes(window, self.codebook_size, self.num_reserved_tokens)
        emb = codes_to_embeddings(codes, self.quant)
        wav = self.codec(emb.to(self._dt))
        end = wav.shape[1] - m * FRAME_SAMPLES
        new = wav[:, end - avail * FRAME_SAMPLES: end]
        return new, emitted + avail

    def init_decode_state(self):
        """Fresh state dict for streaming decode (decode_stream)."""
        return {}

    @torch.no_grad()
    def prefill_f(self, dstate, source_frame, seed_codes):
        """Teacher-force seed_codes [1,N,Q] (raw 0..codebook_size-1) through the
        temporal transformer to populate its KV cache (native mlx_engine prefill
        parity), so generation CONTINUES from the seed. Advances `dstate` in place.
        Returns unique-code frames [1,N,Q] for the codec decoder."""
        dec = self.depthformer.decoder
        Q = self.config.num_codebooks
        per_cb = (torch.arange(Q, device=seed_codes.device) * self.codebook_size
                  + self.num_reserved_tokens).view(1, 1, Q)
        unique = seed_codes.to(torch.long) + per_cb
        N = unique.shape[1]
        for step in range(max(0, N - 1)):
            dec.step_f(dstate, source_frame, forced=unique[:, step:step + 1, :],
                       temporal_step=self._temporal_step, depth_step=self._depth_step)
        if N > 0:
            dstate["prev"] = unique[:, N - 1:N, :]
        return unique

    def decode_stream(self, new_codes, state):
        """Incremental codec decode of new token frames [b, t_new, Q] -> audio [b, N, 2].
        FLOP-optimal stateful streaming (no overlap-save re-decode); bf16-equivalent to
        _decode_stream/forward, with a 1-frame (40ms) decoder latency. `state` starts as {}."""
        codes = convert_from_unique_codes(new_codes, self.codebook_size, self.num_reserved_tokens)
        emb = codes_to_embeddings(codes, self.quant)
        return self.codec.decode_streaming(emb.to(self._dt), state)

    # ---- forward: one teacher-forced pass (logits), for parity / training hooks ----
    def forward(self, style_tokens=None, target=None, source=None, **kwargs):
        """If `source` is given, returns per-frame logits for `target` [b,T,Q].
        This is the teacher-forced path; for sampling use `generate`/`stream`."""
        if source is None:
            cond = self._resolve_conditioning(style_tokens, None, None, None, None, None)
            source = self.depthformer.encode(cond).to(self._dt)
        logits = self.depthformer.decoder(target, source)   # MultivariateDecoder.forward(target, source); was self.depthformer(target,source)=Depthformer.forward(cond,target) — swapped slots
        return {"logits": logits, "source": source}

    # ---- generation (custom; not GenerationMixin) ----
    @torch.no_grad()
    def generate(self, style=None, notes=None, drums=None, cfg_musiccoca=None,
                 cfg_notes=None, cfg_drums=None, temperature=None, top_k=None,
                 frames=25, seed=0, state=None, flush=False, return_int16=False,
                 guidance=False, pca_coeffs=None):
        """`guidance=False` (default): cfg_* are discretized conditioning tokens — the
        validated in-process/JAX path, unchanged. `guidance=True`: cfg_musiccoca/cfg_notes
        become classifier-free-guidance scales (negatives + per-codebook logit combine),
        matching the native MLX/Mac-app path. Guidance uses eager steps (batch>1).
        pca_coeffs: optional style shift along the MusicCoCa PCA basis (see compute_pca)."""
        c = self.config
        temperature = c.temperature if temperature is None else temperature
        top_k = c.top_k if top_k is None else top_k
        if guidance:
            source, cfg_scales = self._guidance_source(style, notes, drums, cfg_musiccoca, cfg_notes, pca_coeffs)
            arity = len(cfg_scales) + 1
        else:
            cond = self._resolve_conditioning(style, notes, drums, cfg_musiccoca, cfg_notes, cfg_drums, pca_coeffs)
            source = self.depthformer.encode(cond).to(self._dt)
            cfg_scales, arity = None, 1
        if state is None:
            dstate = self.depthformer.decoder.init_streaming_f(arity, self._dev, self._dt)
            gen = torch.Generator(device=self._dev).manual_seed(seed)
            decode_state = self.init_decode_state()
        else:
            dstate, gen, decode_state = state["dstate"], state["gen"], state["decode_state"]
        sampler = make_sampler(temperature, top_k, gen)
        # dynamic-batch AOTI (or eager fallback) handles guidance B>1 and no-guidance B=1 alike.
        toks = [self.depthformer.decoder.step_f(
            dstate, source, sampler=sampler, cfg_scales=cfg_scales,
            temporal_step=self._temporal_step, depth_step=self._depth_step) for _ in range(frames)]
        audio = self.decode_stream(torch.cat(toks, dim=1), decode_state)   # stateful per-frame streaming decode (40ms frames)
        new_state = {"dstate": dstate, "gen": gen, "decode_state": decode_state}
        wav = audio[0].float().cpu().numpy()
        i16 = _float_to_int16(wav)
        out = i16 if return_int16 else i16.astype(np.float32) / 32768.0
        return out, new_state

    @torch.no_grad()
    def stream(self, control, chunk_frames=10, max_seconds=55.0, seed=0,
               time_fn=None, sleep_fn=None, notes=None, drums=None, guidance=False,
               cudagraph=False):
        """Continuous generation. `control()` returns {style_tokens, temperature,
        top_k, cfg_*} read every chunk for mid-stream steering. Yields int16 [N,2].

        guidance=False (default): cfg_* are conditioning tokens (validated token path,
        unchanged). guidance=True: cfg_musiccoca/cfg_notes are classifier-free-guidance
        scales read live every chunk. cudagraph=True: single-dispatch CUDA-graph stepping
        (one capture at start, ~4-5x faster), steered via static input buffers."""
        if cudagraph:
            yield from self._stream_cudagraph(control, chunk_frames, max_seconds, seed,
                                              time_fn, sleep_fn, notes, drums, guidance)
            return
        import time as _time
        time_fn = time_fn or _time.time
        sleep_fn = sleep_fn or _time.sleep
        c = self.config
        dev, dt = self._dev, self._dt
        notes = notes if notes is not None else [-1] * self.num_notes
        drums = drums if drums is not None else [-1] * self.num_drums
        arity = 3 if guidance else 1
        dstate = self.depthformer.decoder.init_streaming_f(arity, dev, dt)
        gen = torch.Generator(device=dev).manual_seed(seed)
        decode_state = self.init_decode_state()
        emitted_samples = 0
        cur_tokens = None
        source = None
        t0 = time_fn()
        while time_fn() - t0 < max_seconds:
            ctl = control()
            if ctl is None:
                sleep_fn(0.02)
                continue
            tokens = ctl["style_tokens"]
            if tokens != cur_tokens:
                cur_tokens = tokens
                st = (list(tokens) + [-1] * self.num_musiccoca)[:self.num_musiccoca]
                if guidance:                                  # [pos, neg_mc, neg_n]; cfg tokens neutralized
                    source, _ = self._guidance_source(st, notes, drums, None, None)
                else:
                    cfgs = [discretize_cfg(ctl.get("cfg_musiccoca", c.cfg_musiccoca), 0.2, 40),
                            discretize_cfg(ctl.get("cfg_notes", c.cfg_notes), 0.2, 40),
                            discretize_cfg(ctl.get("cfg_drums", c.cfg_drums), 1.0, 8)]
                    source = self.depthformer.encode(self._conditioning(st, notes, drums, cfgs)).to(dt)
            cfg_scales = ((float(ctl.get("cfg_musiccoca", c.cfg_musiccoca)),     # live scales (unclamped)
                           float(ctl.get("cfg_notes", c.cfg_notes))) if guidance else None)
            sampler = make_sampler(ctl.get("temperature", c.temperature), ctl.get("top_k", c.top_k), gen)
            toks = [self.depthformer.decoder.step_f(
                dstate, source, sampler=sampler, cfg_scales=cfg_scales,
                temporal_step=self._temporal_step, depth_step=self._depth_step) for _ in range(chunk_frames)]
            audio = self.decode_stream(torch.cat(toks, dim=1), decode_state)
            emitted_samples += audio.shape[1]
            if audio.shape[1] > 0:
                yield _float_to_int16(audio[0].float().cpu().numpy())
            ahead = (emitted_samples / SR) - (time_fn() - t0)
            if ahead > 1.0:
                sleep_fn(min(ahead - 1.0, 0.5))

    @torch.no_grad()
    def _stream_cudagraph(self, control, chunk_frames, max_seconds, seed,
                          time_fn, sleep_fn, notes, drums, guidance):
        """CUDA-graph backend for stream(cudagraph=True): one capture at start
        (warmup ~KEEP frames), then single-dispatch replay per frame. Steering
        goes through the streamer's static input buffers — cfg/temperature are
        buffer writes; a style change re-encodes + set_source (windowed ramp)."""
        import time as _time
        time_fn = time_fn or _time.time
        sleep_fn = sleep_fn or _time.sleep
        c = self.config
        dt = self._dt
        notes = notes if notes is not None else [-1] * self.num_notes
        drums = drums if drums is not None else [-1] * self.num_drums

        def encode_src(tokens, cfg_mc, cfg_n):
            st = (list(tokens) + [-1] * self.num_musiccoca)[:self.num_musiccoca]
            if guidance:
                return self._guidance_source(st, notes, drums, cfg_mc, cfg_n)[0]
            cfgs = [discretize_cfg(cfg_mc, 0.2, 40), discretize_cfg(cfg_n, 0.2, 40),
                    discretize_cfg(c.cfg_drums, 1.0, 8)]
            return self.depthformer.encode(self._conditioning(st, notes, drums, cfgs)).to(dt)

        # bounded wait for the first conditioning, then build + capture the graph
        t0 = time_fn()
        ctl = control()
        while ctl is None and time_fn() - t0 < max_seconds:
            sleep_fn(0.02); ctl = control()
        if ctl is None:
            return
        cur_tokens = ctl["style_tokens"]
        cur_cfg = (float(ctl.get("cfg_musiccoca", c.cfg_musiccoca)),
                   float(ctl.get("cfg_notes", c.cfg_notes)))
        streamer = self.make_cudagraph_streamer(
            style=cur_tokens, notes=notes, drums=drums,
            cfg_musiccoca=cur_cfg[0], cfg_notes=cur_cfg[1],
            temperature=ctl.get("temperature", c.temperature),
            top_k=ctl.get("top_k", c.top_k), seed=seed, guidance=guidance)
        decode_state = self.init_decode_state()
        emitted_samples = 0
        t0 = time_fn()
        while time_fn() - t0 < max_seconds:
            ctl = control()
            if ctl is None:
                sleep_fn(0.005); continue
            tokens = ctl["style_tokens"]
            cfg_mc = float(ctl.get("cfg_musiccoca", c.cfg_musiccoca))
            cfg_n = float(ctl.get("cfg_notes", c.cfg_notes))
            if guidance:
                if (cfg_mc, cfg_n) != cur_cfg:
                    streamer.set_cfg([cfg_mc, cfg_n]); cur_cfg = (cfg_mc, cfg_n)
                if tokens != cur_tokens:
                    streamer.set_source(encode_src(tokens, cfg_mc, cfg_n)); cur_tokens = tokens
            elif tokens != cur_tokens or (cfg_mc, cfg_n) != cur_cfg:   # token path: cfg lives in source
                streamer.set_source(encode_src(tokens, cfg_mc, cfg_n))
                cur_tokens, cur_cfg = tokens, (cfg_mc, cfg_n)
            streamer.set_temperature(ctl.get("temperature", c.temperature))
            toks = [streamer.step() for _ in range(chunk_frames)]
            audio = self.decode_stream(torch.cat(toks, dim=1), decode_state)
            emitted_samples += audio.shape[1]
            if audio.shape[1] > 0:
                yield _float_to_int16(audio[0].float().cpu().numpy())
            ahead = (emitted_samples / SR) - (time_fn() - t0)
            if ahead > 1.0:
                sleep_fn(min(ahead - 1.0, 0.5))

    @torch.no_grad()
    def make_cudagraph_streamer(self, style=None, notes=None, drums=None,
                                cfg_musiccoca=None, cfg_notes=None, cfg_drums=None,
                                temperature=None, top_k=None, seed=0, guidance=False,
                                warmup=None):
        """One-dispatch-per-frame CUDA-graph streaming: captures the whole frame
        (temporal + N-codebook depth + in-graph sampler + optional CFG) as a single
        `torch.cuda.graph` replay over fixed-size static KV buffers — ~MLX `.mlxfn`.
        Returns a `CudaGraphStreamer`; call `.step()` for the next frame [1,1,Q]
        (decode with `decode_stream`), and `.set_cfg/.set_temperature/.set_source`
        for live steering (no re-capture). `top_k` is fixed at capture time."""
        if guidance:
            source, scales = self._guidance_source(style, notes, drums, cfg_musiccoca, cfg_notes)
            num_neg = len(scales)
        else:
            cond = self._resolve_conditioning(style, notes, drums, cfg_musiccoca, cfg_notes, cfg_drums)
            source = self.depthformer.encode(cond).to(self._dt)
            scales, num_neg = None, 0
        temperature = self.config.temperature if temperature is None else temperature
        top_k = self.config.top_k if top_k is None else top_k
        return CudaGraphStreamer(self.depthformer.decoder, source, self._dt, num_neg, scales,
                                 temperature, top_k, seed, warmup)


__all__ = ["MagentaRT2ForConditionalGeneration", "MagentaRT2PreTrainedModel", "CudaGraphStreamer"]
