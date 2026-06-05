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

import numpy as np
import torch
from transformers import PreTrainedModel

from .configuration_magenta_rt2 import MagentaRT2Config
from .depthformer import Depthformer, DepthformerConfig, SpecDims
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

    # ---- AOTI ----
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
    def _tokenize_style(self, style):
        if self.processor is None:
            raise ValueError("No MusicCoCaProcessor attached; pass `style` as a list of "
                             f"{self.num_musiccoca} RVQ token ids, or call set_processor().")
        return np.asarray(self.processor.tokenize(self.processor.embed(style))).tolist()

    def _conditioning(self, style_tokens, notes, drums, cfgs):
        offset = self.num_reserved_tokens + 1
        vals = list(style_tokens) + list(notes) + list(drums) + list(cfgs)
        arr = np.array(vals, dtype=np.int64) + offset
        return torch.from_numpy(arr).view(1, 1, -1).to(self._dev)

    def _resolve_conditioning(self, style, notes, drums, cfg_musiccoca, cfg_notes, cfg_drums):
        c = self.config
        if style is None:
            style_tokens = [-1] * self.num_musiccoca
        elif isinstance(style, (list, np.ndarray)) and np.asarray(style).ndim == 1 \
                and np.asarray(style).dtype.kind in "iu" and len(style) == self.num_musiccoca:
            style_tokens = list(style)
        else:
            style_tokens = self._tokenize_style(style)
        style_tokens = (list(style_tokens) + [-1] * self.num_musiccoca)[:self.num_musiccoca]
        notes = notes if notes is not None else [-1] * self.num_notes
        drums = drums if drums is not None else [-1] * self.num_drums
        cfgs = [
            discretize_cfg(c.cfg_musiccoca if cfg_musiccoca is None else cfg_musiccoca, 0.2, 40),
            discretize_cfg(c.cfg_notes if cfg_notes is None else cfg_notes, 0.2, 40),
            discretize_cfg(c.cfg_drums if cfg_drums is None else cfg_drums, 1.0, 8),
        ]
        return self._conditioning(style_tokens, notes, drums, cfgs)

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

    # ---- forward: one teacher-forced pass (logits), for parity / training hooks ----
    def forward(self, style_tokens=None, target=None, source=None, **kwargs):
        """If `source` is given, returns per-frame logits for `target` [b,T,Q].
        This is the teacher-forced path; for sampling use `generate`/`stream`."""
        if source is None:
            cond = self._resolve_conditioning(style_tokens, None, None, None, None, None)
            source = self.depthformer.encode(cond).to(self._dt)
        logits = self.depthformer(target, source)
        return {"logits": logits, "source": source}

    # ---- generation (custom; not GenerationMixin) ----
    @torch.no_grad()
    def generate(self, style=None, notes=None, drums=None, cfg_musiccoca=None,
                 cfg_notes=None, cfg_drums=None, temperature=None, top_k=None,
                 frames=25, seed=0, state=None, flush=False, return_int16=False):
        c = self.config
        temperature = c.temperature if temperature is None else temperature
        top_k = c.top_k if top_k is None else top_k
        cond = self._resolve_conditioning(style, notes, drums, cfg_musiccoca, cfg_notes, cfg_drums)
        source = self.depthformer.encode(cond).to(self._dt)
        if state is None:
            dstate = self.depthformer.decoder.init_streaming_f(1, self._dev, self._dt)
            gen = torch.Generator(device=self._dev).manual_seed(seed)
            history = torch.zeros((1, 0, c.num_codebooks), dtype=torch.long, device=self._dev)
            emitted = 0
        else:
            dstate, gen, history, emitted = state["dstate"], state["gen"], state["history"], state["emitted"]
        sampler = make_sampler(temperature, top_k, gen)
        toks = [self.depthformer.decoder.step_f(
            dstate, source, sampler=sampler,
            temporal_step=self._temporal_step, depth_step=self._depth_step) for _ in range(frames)]
        history = torch.cat([history] + toks, dim=1)
        audio, emitted = self._decode_stream(history, emitted, flush=flush)
        new_state = {"dstate": dstate, "gen": gen, "history": history, "emitted": emitted}
        wav = audio[0].float().cpu().numpy()
        i16 = _float_to_int16(wav)
        out = i16 if return_int16 else i16.astype(np.float32) / 32768.0
        return out, new_state

    @torch.no_grad()
    def stream(self, control, chunk_frames=10, max_seconds=55.0, seed=0,
               time_fn=None, sleep_fn=None, notes=None, drums=None):
        """Continuous generation. `control()` returns {style_tokens, temperature,
        top_k, cfg_*} read every chunk for mid-stream steering. Yields int16 [N,2]."""
        import time as _time
        time_fn = time_fn or _time.time
        sleep_fn = sleep_fn or _time.sleep
        c = self.config
        dev, dt = self._dev, self._dt
        notes = notes if notes is not None else [-1] * self.num_notes
        drums = drums if drums is not None else [-1] * self.num_drums
        dstate = self.depthformer.decoder.init_streaming_f(1, dev, dt)
        gen = torch.Generator(device=dev).manual_seed(seed)
        history = torch.zeros((1, 0, c.num_codebooks), dtype=torch.long, device=dev)
        emitted = 0
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
                cfgs = [discretize_cfg(ctl.get("cfg_musiccoca", c.cfg_musiccoca), 0.2, 40),
                        discretize_cfg(ctl.get("cfg_notes", c.cfg_notes), 0.2, 40),
                        discretize_cfg(ctl.get("cfg_drums", c.cfg_drums), 1.0, 8)]
                cond = self._conditioning((list(tokens) + [-1] * self.num_musiccoca)[:self.num_musiccoca],
                                          notes, drums, cfgs)
                source = self.depthformer.encode(cond).to(dt)
            sampler = make_sampler(ctl.get("temperature", c.temperature), ctl.get("top_k", c.top_k), gen)
            toks = [self.depthformer.decoder.step_f(
                dstate, source, sampler=sampler,
                temporal_step=self._temporal_step, depth_step=self._depth_step) for _ in range(chunk_frames)]
            history = torch.cat([history] + toks, dim=1)
            audio, emitted = self._decode_stream(history, emitted)
            if audio.shape[1] > 0:
                yield _float_to_int16(audio[0].float().cpu().numpy())
            ahead = (emitted * FRAME_SAMPLES / SR) - (time_fn() - t0)
            if ahead > 1.0:
                sleep_fn(min(ahead - 1.0, 0.5))


__all__ = ["MagentaRT2ForConditionalGeneration", "MagentaRT2PreTrainedModel"]
