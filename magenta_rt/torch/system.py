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

"""Pure-PyTorch Magenta RealTime 2 streaming system.

Generation uses the torch Depthformer (verified vs JAX) for autoregressive
token sampling and the torch SpectroStream decoder (verified vs JAX) for audio.
MusicCoCa style embedding/tokenization is reused from the framework-agnostic
TFLite component (same as the upstream JAX/MLX paths).
"""
import numpy as np
import torch

from .. import paths
from .depthformer import Depthformer, config_for
from .weights import load_depthformer
from .spectrostream import load_spectrostream_decoder, codes_to_embeddings

NUM_RESERVED_TOKENS = 6
SR = 48000                    # output sample rate
FRAME_SAMPLES = 1920          # 48kHz samples per 40ms codec frame
STREAM_DECODE_CONTEXT = 16    # token frames of left context for overlap-save decode
STREAM_DECODE_MARGIN = 2      # token frames held back (right context: lookahead + ISTFT overlap)
_CHECKPOINTS = {"mrt2_small": "mrt2_small.safetensors", "mrt2_base": "mrt2_base.safetensors"}


def discretize_cfg(value, step, max_bin):
    clamped = max(-1.0, min(7.0, value))
    return max(0, min(max_bin, int(round((clamped - (-1.0)) / step))))


def convert_from_unique_codes(tokens, codebook_size=1024):
    return (tokens - NUM_RESERVED_TOKENS) % codebook_size


def _float_to_int16(samples, gain=0.5):
    samples = np.clip(gain * samples, -1, 1)
    samples = np.round((np.iinfo(np.int16).max + 0.5) * samples - 0.5)
    return samples.astype(np.int16)


def make_sampler(temperature, top_k, generator):
    # Matches jax _sample_categorical_with_temperature: valid-range mask, top-k,
    # then gumbel-max (logits + gumbel * temperature, argmax). Uses a finite
    # large-negative for masking (a la get_large_negative_number).
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
        return logits.argmax(dim=-1)  # [b,1]
    return sampler


class MagentaRT2:
    def __init__(self, size="mrt2_small", device="cpu", dtype=torch.float32,
                 temperature=1.3, top_k=40, cfg_musiccoca=3.0, cfg_notes=1.0,
                 cfg_drums=1.0, style_model=None, compile=False):
        self.size = size
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.top_k = top_k
        self.cfg_musiccoca = cfg_musiccoca
        self.cfg_notes = cfg_notes
        self.cfg_drums = cfg_drums
        self._style_model = style_model

        cfg = config_for(size)
        self.cfg = cfg
        self.model = Depthformer(cfg).eval().to(device)
        ckpt = paths.checkpoints_dir() / _CHECKPOINTS[size]
        load_depthformer(self.model, ckpt, dtype=dtype)
        self.model.to(dtype)
        self.dec, quant = load_spectrostream_decoder(ckpt, dtype=dtype)
        self.dec = self.dec.eval().to(device)
        self.quant = quant.to(device)

        # AOTI-compiled per-frame step callables (None = eager). Set via
        # load_compiled() or apply_compiled().
        self._temporal_step = None
        self._depth_step = None

        if compile:
            # Compile the two hot per-frame step paths (dynamic shapes for the
            # growing KV caches). One-time warmup; ~1.8x and past real-time.
            dec = self.model.decoder
            dec.temporal_body.step = torch.compile(dec.temporal_body.step, dynamic=True)
            dec._depth_step_logits = torch.compile(dec._depth_step_logits, dynamic=True)

        self.num_musiccoca = cfg.musiccoca_rvq         # 12
        self.num_notes = 128
        self.num_drums = 1
        self.num_cfg = 3
        self.num_channels = self.num_musiccoca + self.num_notes + self.num_drums + self.num_cfg
        self.sample_rate = 48000
        self.codebook_size = cfg.codebook_size

    def apply_compiled(self, temporal_step=None, depth_step=None):
        """Wire AOTI-compiled per-frame step callables into generation."""
        if temporal_step is not None:
            self._temporal_step = temporal_step
        if depth_step is not None:
            self._depth_step = depth_step

    def load_compiled(self, repo_id=None, local_dir=None):
        """Load AOTI artifacts (weight-less, compiled on matching GPU arch) and
        bind them to this model's weights via aokit."""
        from . import aoti
        t, d = aoti.load_compiled_steps(self.model.decoder, repo_id=repo_id, local_dir=local_dir)
        self.apply_compiled(t, d)
        return self

    # ---- style ----
    @property
    def style_model(self):
        if self._style_model is None:
            from .. import musiccoca
            self._style_model = musiccoca.MusicCoCa()
        return self._style_model

    def embed_style(self, text_or_audio, **kw):
        return self.style_model.embed(text_or_audio, **kw)

    def tokenize_style(self, embedding):
        return self.style_model.tokenize(embedding)

    # ---- conditioning ----
    def _conditioning(self, style_tokens, notes, drums, cfgs):
        offset = NUM_RESERVED_TOKENS + 1
        vals = list(style_tokens) + list(notes) + list(drums) + list(cfgs)
        arr = np.array(vals, dtype=np.int64) + offset
        return torch.from_numpy(arr).view(1, 1, -1).to(self.device)

    def _resolve_conditioning(self, style, notes, drums, cfg_musiccoca, cfg_notes, cfg_drums):
        if style is None:
            style_tokens = [-1] * self.num_musiccoca
        elif isinstance(style, (list, np.ndarray)) and len(np.asarray(style).shape) == 1 and np.asarray(style).dtype.kind in "iu" and len(style) == self.num_musiccoca:
            style_tokens = list(style)
        else:
            style_tokens = self.tokenize_style(style).tolist()
        style_tokens = (style_tokens + [-1] * self.num_musiccoca)[:self.num_musiccoca]
        notes = notes if notes is not None else [-1] * self.num_notes
        drums = drums if drums is not None else [-1] * self.num_drums
        cfgs = [
            discretize_cfg(self.cfg_musiccoca if cfg_musiccoca is None else cfg_musiccoca, 0.2, 40),
            discretize_cfg(self.cfg_notes if cfg_notes is None else cfg_notes, 0.2, 40),
            discretize_cfg(self.cfg_drums if cfg_drums is None else cfg_drums, 1.0, 8),
        ]
        return self._conditioning(style_tokens, notes, drums, cfgs)

    def _decode_stream(self, history, emitted, context=STREAM_DECODE_CONTEXT,
                       margin=STREAM_DECODE_MARGIN, flush=False):
        """Overlap-save: decode recent token context, emit only the safe new tail.

        The SpectroStream decoder has a small left receptive field and ~1 frame
        of lookahead + ISTFT overlap on the right. Decoding `[left-context ...
        new ... right-margin]` and emitting the interior gives output identical
        to a fully stateful streaming codec. `margin` frames are held back until
        their future context exists (flush=True emits them at stream end).
        Returns (new_samples [1,N,2], new_emitted)."""
        m = 0 if flush else margin
        Ttot = history.shape[1]
        emittable = (Ttot - 1) - m
        avail = emittable - emitted
        if avail <= 0:
            return history.new_zeros((1, 0, 2), dtype=self.dtype), emitted
        w0 = max(0, Ttot - (avail + m + context + 1))
        window = history[:, w0:]
        codes = convert_from_unique_codes(window, self.codebook_size)
        emb = codes_to_embeddings(codes, self.quant)
        wav = self.dec(emb.to(self.dtype))            # [1, (len(window)-1)*1920, 2]
        end = wav.shape[1] - m * FRAME_SAMPLES
        new = wav[:, end - avail * FRAME_SAMPLES: end]
        return new, emitted + avail

    @torch.no_grad()
    def stream_session(self, control, chunk_frames=10, max_seconds=55.0,
                       seed=0, time_fn=None, sleep_fn=None, notes=None, drums=None):
        """Continuous generation for an interactive session. `control()` returns a
        dict {style_tokens, temperature, top_k, cfg_*} read every chunk, so the
        prompt can change mid-stream. Yields int16 [N,2] audio chunks. Keeps LLM
        state across chunks; re-encodes the conditioning `source` when style changes.
        Paces to ~real-time so steering stays responsive."""
        import time as _time
        time_fn = time_fn or _time.time
        sleep_fn = sleep_fn or _time.sleep
        dev, dt = self.device, self.dtype
        notes = notes if notes is not None else [-1] * self.num_notes
        drums = drums if drums is not None else [-1] * self.num_drums
        dstate = self.model.decoder.init_streaming_f(1, dev, dt)
        gen = torch.Generator(device=dev).manual_seed(seed)
        history = torch.zeros((1, 0, self.cfg.num_codebooks), dtype=torch.long, device=dev)
        emitted = 0
        cur_tokens = None
        source = None
        t0 = time_fn()
        while time_fn() - t0 < max_seconds:
            c = control()
            if c is None:
                sleep_fn(0.02)
                continue
            tokens = c["style_tokens"]
            if tokens != cur_tokens:
                cur_tokens = tokens
                cfgs = [discretize_cfg(c.get("cfg_musiccoca", self.cfg_musiccoca), 0.2, 40),
                        discretize_cfg(c.get("cfg_notes", self.cfg_notes), 0.2, 40),
                        discretize_cfg(c.get("cfg_drums", self.cfg_drums), 1.0, 8)]
                cond = self._conditioning((list(tokens) + [-1] * self.num_musiccoca)[:self.num_musiccoca],
                                          notes, drums, cfgs)
                source = self.model.encode(cond).to(dt)
            sampler = make_sampler(c.get("temperature", self.temperature), c.get("top_k", self.top_k), gen)
            toks = []
            for _ in range(chunk_frames):
                toks.append(self.model.decoder.step_f(
                    dstate, source, sampler=sampler,
                    temporal_step=self._temporal_step, depth_step=self._depth_step))
            history = torch.cat([history] + toks, dim=1)
            audio, emitted = self._decode_stream(history, emitted)
            if audio.shape[1] > 0:
                yield _float_to_int16(audio[0].float().cpu().numpy())
            # pace: keep generated audio ~1s ahead of wall-clock
            ahead = (emitted * FRAME_SAMPLES / SR) - (time_fn() - t0)
            if ahead > 1.0:
                sleep_fn(min(ahead - 1.0, 0.5))

    @torch.no_grad()
    def generate(self, style=None, notes=None, drums=None, cfg_musiccoca=None,
                 cfg_notes=None, cfg_drums=None, temperature=None, top_k=None,
                 frames=25, seed=0, state=None, flush=False, return_int16=False):
        """Generate `frames` of audio. Pass the returned `state` back in to
        continue seamlessly (continuous/live generation); conditioning args may
        change between calls to steer the stream. Audio is emitted incrementally
        (only the newly-available chunk is returned each call). Set flush=True on
        the final call to emit the held-back tail frames."""
        temperature = self.temperature if temperature is None else temperature
        top_k = self.top_k if top_k is None else top_k
        cond = self._resolve_conditioning(style, notes, drums, cfg_musiccoca, cfg_notes, cfg_drums)
        source = self.model.encode(cond).to(self.dtype)  # constant per frame this call

        if state is None:
            dstate = self.model.decoder.init_streaming_f(1, self.device, self.dtype)
            gen = torch.Generator(device=self.device).manual_seed(seed)
            history = torch.zeros((1, 0, self.cfg.num_codebooks), dtype=torch.long, device=self.device)
            emitted = 0
        else:
            dstate, gen, history, emitted = state["dstate"], state["gen"], state["history"], state["emitted"]

        sampler = make_sampler(temperature, top_k, gen)
        toks = []
        for _ in range(frames):
            frame = self.model.decoder.step_f(
                dstate, source, sampler=sampler,
                temporal_step=self._temporal_step, depth_step=self._depth_step)
            toks.append(frame)
        history = torch.cat([history] + toks, dim=1)
        audio, emitted = self._decode_stream(history, emitted, flush=flush)

        new_state = {"dstate": dstate, "gen": gen, "history": history, "emitted": emitted}
        wav = audio[0].float().cpu().numpy()
        i16 = _float_to_int16(wav)  # 0.5 gain + clip, pointwise (chunk-safe)
        out = i16 if return_int16 else i16.astype(np.float32) / 32768.0
        return out, new_state
