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

"""Single-dispatch CUDA-graph streaming for the Depthformer decoder.

`CudaGraphStreamer` captures the whole per-frame step (temporal + N-codebook depth
+ in-graph sampler + optional CFG) as one `torch.cuda.graph` replay over fixed-size
static KV buffers — the PyTorch analog of the native MLX `.mlxfn` (one dispatch per
frame, ~4-5x real-time). It is **decoder-coupled and transformers-free** so both the
`transformers` modeling class and the lighter system class can use it without pulling
in the model wrapper.
"""

import warnings

import torch

# Classifier-free guidance scales above this can run away / collapse the output
# to silence under *sustained constant* conditioning over long runs (the native
# UI uses a 0-5 slider, default 2.4). We don't clamp — values pass through to
# match the native range — but we warn once so the caller knows the risk.
GUIDANCE_CFG_WARN = 3.5


def _warn_high_cfg(*scales):
    hi = [round(float(s), 2) for s in scales if float(s) > GUIDANCE_CFG_WARN]
    if hi:
        warnings.warn(
            f"CFG guidance scale(s) {hi} exceed ~{GUIDANCE_CFG_WARN}; sustained high "
            "guidance on constant conditioning can make the output run away / collapse "
            "to silence over long runs. (Changing notes/style during play avoids this.)",
            stacklevel=3)
        return True
    return False


class CudaGraphStreamer:
    """Single-dispatch CUDA-graph frame stepper over fixed-size static KV buffers.

    Warms `KEEP` frames eagerly to fill the temporal/cross KV to steady state,
    snapshots them into static buffers, then captures one frame (temporal + depth +
    sampler) with `torch.cuda.graph`. `.step()` replays it (one GPU dispatch) and
    returns the new frame tokens. Live steering writes into static input buffers
    (`source`, `cfg`, `temperature`) — the captured graph reads them, no re-capture.
    Conditioning changes ramp in via the windowed cross-KV (optional hard flush)."""

    def __init__(self, decoder, source, decode_dtype, num_neg=0, cfg_scales=None,
                 temperature=1.1, top_k=50, seed=0, warmup=None):
        """decoder: a MultivariateDecoder (`model.depthformer.decoder` for the
        modeling class, `model.model.decoder` for the system class). `source` is the
        pre-encoded conditioning [B, Tc, enc] (B = 1 + num_neg); `decode_dtype` the
        compute dtype. Class-agnostic so both model wrappers can build it."""
        dec = decoder
        c = dec.cfg
        self.dec = dec
        self.Q, self.CB, self.NR = c.num_codebooks, c.codebook_size, c.num_reserved_tokens
        self.KEEP = c.temporal_max_past + 1
        self.num_neg = num_neg
        self.top_k = int(top_k)
        dev, dt = source.device, decode_dtype
        B = source.shape[0]; self.B = B
        # live-steering static inputs
        self.source = source.clone()
        self.cfg = (torch.zeros(0, device=dev, dtype=torch.float32) if not num_neg
                    else torch.tensor([float(s) for s in cfg_scales], device=dev, dtype=torch.float32))
        self.temp = torch.tensor(float(temperature), device=dev, dtype=torch.float32)
        torch.manual_seed(seed)
        # 1) prime to steady state (KV == KEEP on every layer)
        st = dec.init_streaming_f(B, dev, dt)
        K = self.KEEP
        for _ in range(K + 8 if warmup is None else warmup):
            to, ns, nc = dec.temporal_step_fn(st["prev"], st["self"], st["cross"], self.source)
            st["self"] = [(k[:, -K:], v[:, -K:]) for k, v in ns]
            st["cross"] = [(k[:, -K:], v[:, -K:]) for k, v in nc]
            frame = self._depth_sample(to)
            st["prev"] = frame.expand(B, -1, -1)
        # 2) static KV + state buffers
        L = len(st["self"]); self.L = L
        self.SK = [st["self"][i][0].clone() for i in range(L)]; self.SV = [st["self"][i][1].clone() for i in range(L)]
        self.CK = [st["cross"][i][0].clone() for i in range(L)]; self.CV = [st["cross"][i][1].clone() for i in range(L)]
        self.prev = st["prev"].clone()
        self.out = torch.zeros(1, 1, self.Q, dtype=torch.long, device=dev)
        # 3) capture (side-stream warmup is required before graph capture)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._frame_static()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._frame_static()

    def _depth_sample(self, to):
        dec = self.dec; B = self.B; Q, CB, NR = self.Q, self.CB, self.NR
        dd = dec.cfg.depth
        z = torch.zeros(B, 0, dd.num_heads, dd.dim_per_head, device=to.device, dtype=to.dtype)
        dk = [(z, z) for _ in range(dd.num_layers)]
        di = to; toks = []
        for q in range(Q):
            logits, dk = dec.depth_step_fn(di, dk)            # [B,1,V]
            lo = NR + q * CB
            ls = logits[..., lo:lo + CB]
            cond = ls[0:1]; comb = cond
            for i in range(self.num_neg):                     # classifier-free guidance combine
                comb = comb + self.cfg[i] * (cond - ls[i + 1:i + 2])
            kth = torch.topk(comb, self.top_k, dim=-1).values[..., -1:]
            comb = torch.where(comb >= kth, comb, torch.full_like(comb, -1e9))
            u = torch.rand(1, 1, CB, device=to.device, dtype=torch.float32)   # graph-safe RNG
            g = -torch.log(-torch.log(u.clamp(1e-10, 1 - 1e-7)))
            tok = (comb + g * self.temp).argmax(-1) + lo
            toks.append(tok)
            di = dec.embed(tok.expand(B, -1))
        return torch.stack(toks, dim=-1)                       # [1,1,Q]

    def _frame_static(self):
        dec = self.dec; K = self.KEEP; L = self.L
        to, ns, nc = dec.temporal_step_fn(
            self.prev, [(self.SK[i], self.SV[i]) for i in range(L)],
            [(self.CK[i], self.CV[i]) for i in range(L)], self.source)
        for i in range(L):
            self.SK[i].copy_(ns[i][0][:, -K:]); self.SV[i].copy_(ns[i][1][:, -K:])
            self.CK[i].copy_(nc[i][0][:, -K:]); self.CV[i].copy_(nc[i][1][:, -K:])
        frame = self._depth_sample(to)
        self.out.copy_(frame)
        self.prev.copy_(frame.expand(self.B, -1, -1))

    # ---- live steering (no re-capture) ----
    def set_cfg(self, scales):
        if self.num_neg:
            if not getattr(self, "_cfg_warned", False):
                self._cfg_warned = _warn_high_cfg(*scales)
            self.cfg.copy_(torch.tensor([float(s) for s in scales],
                                        device=self.cfg.device, dtype=torch.float32))

    def set_temperature(self, t):
        self.temp.fill_(float(t))

    def set_source(self, source, flush=False):
        """Update conditioning. Ramps in via the windowed cross-KV; flush=True
        overwrites all cross-KV slots for an immediate change."""
        self.source.copy_(source if source.shape[0] == self.B else source.expand(self.B, -1, -1))
        if flush:
            for i in range(self.L):
                sk, sv = self.dec.temporal_body.layers[i]["cross_attention"]._kv(self.source)
                self.CK[i].copy_(sk[:, -self.KEEP:]); self.CV[i].copy_(sv[:, -self.KEEP:])

    def step(self):
        """Advance one frame (single CUDA-graph dispatch). Returns tokens [1,1,Q]."""
        self.graph.replay()
        return self.out.clone()

    def close(self):
        """Free the captured CUDA graph + its private memory pool. Idempotent;
        call at session end (the WS worker should). Safe during interpreter
        shutdown — swallows teardown-ordering errors."""
        g = getattr(self, "graph", None)
        if g is not None:
            try:
                g.reset()
            except Exception:
                pass
            self.graph = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["CudaGraphStreamer", "GUIDANCE_CFG_WARN"]
