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

"""PyTorch port of the SpectroStream decoder (codes -> embeddings -> waveform).

Feature maps are carried as torch NCHW tensors [b, C, T(time), F(freq)] so
F.conv2d applies directly. JAX conv kernels are stored [kh, kw, cin, cout]
(HWIO) and permuted to OIHW. Padding replicates sequence_layers semicausal
(time) / symmetric (freq) for Conv2D, and causal(time)/same(freq) transpose
conv via input dilation + explicit pad + valid conv.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# Architecture constants for the 40ms 48kHz stereo config.
RATIOS = ((1, 2), (1, 2), (1, 3), (1, 2), (1, 2), (2, 2), (2, 1))
CHANNEL_SPLITS = 2
INPUT_BINS = 5
INPUT_CHANNELS = 512
FRAME_LENGTH = 960
FRAME_STEP = 480
FFT_LENGTH = 960
NUM_BINS = 480
TOTAL_TIME_STRIDE = 4
DECODER_LOOKAHEAD = 1


def _semicausal_pad(k, s, d=1):
    eff = (k - 1) * d + 1
    left = max(eff - s, 0)
    return left, (eff - 1) - left


def _sym_freq_pad(kw, sw, dw=1):
    pad = max((kw - 1) * dw + 1 - sw, 0)
    return pad // 2, pad - pad // 2


def _transpose_pad(k, s, mode):
    eff = k
    if mode == "causal":
        amt = eff + s - 2
        left = eff - 1
        return left, amt - left
    elif mode == "same":
        amt = eff + s - 2
        if s > eff - 1:
            left = eff - 1
        else:
            left = int(np.ceil(amt / 2))
        return left, amt - left
    raise ValueError(mode)


def _hann_window(n):
    # periodic raised cosine, a=b=0.5 (matches signal.hann_window)
    even = 1 - n % 2
    denom = n + even - 1  # periodic -> n
    count = np.arange(n)
    return (0.5 - 0.5 * np.cos(2 * np.pi * count / denom)).astype(np.float32)


def _inverse_stft_window(frame_length, frame_step):
    fwd = _hann_window(frame_length)
    denom = fwd ** 2
    overlaps = -(-frame_length // frame_step)
    denom = np.pad(denom, (0, overlaps * frame_step - frame_length))
    denom = denom.reshape(overlaps, frame_step).sum(0, keepdims=True)
    denom = np.tile(denom, (overlaps, 1)).reshape(overlaps * frame_step)[:frame_length]
    return np.where(denom == 0.0, 0.0, fwd / denom).astype(np.float32)


def _overlap_and_add(frames, frame_step):
    """frames: [..., n_frames, frame_length] -> [..., output] (naive, exact)."""
    *outer, n, fl = frames.shape
    out_len = (n - 1) * frame_step + fl
    out = frames.new_zeros(*outer, out_len)
    for i in range(n):
        out[..., i * frame_step: i * frame_step + fl] += frames[..., i, :]
    return out


def _dilate2d(x, strides):
    sh, sw = strides
    b, c, h, w = x.shape
    if sh > 1:
        y = x.new_zeros(b, c, (h - 1) * sh + 1, w)
        y[:, :, ::sh, :] = x
        x = y
        b, c, h, w = x.shape
    if sw > 1:
        y = x.new_zeros(b, c, h, (w - 1) * sw + 1)
        y[:, :, :, ::sw] = x
        x = y
    return x


def elu(x):
    return F.elu(x, alpha=1.0)


class SpectroStreamDecoder(nn.Module):
    """Functional decoder driven by a dict of checkpoint tensors."""

    def __init__(self, weights: dict):
        super().__init__()
        # weights: name -> torch tensor (kernels in HWIO; conv biases 1d).
        self.w = {k: nn.Parameter(v, requires_grad=False) for k, v in weights.items()}
        self.w = nn.ParameterDict({k.replace("/", "__"): v for k, v in self.w.items()})
        self.register_buffer("inv_window", torch.from_numpy(
            _inverse_stft_window(FRAME_LENGTH, FRAME_STEP)))

    def _g(self, name):
        return self.w[name.replace("/", "__")]

    # ---- conv primitives ----
    def _conv1x1(self, x, prefix):
        w = self._g(prefix + "/conv/kernel")  # [1,1,cin,cout]
        b = self._g(prefix + "/conv/bias")
        wk = w.permute(3, 2, 0, 1).to(x.dtype)
        return F.conv2d(x, wk, bias=b.to(x.dtype))

    def _conv2d(self, x, prefix, kh, kw, strides=(1, 1), dil=(1, 1)):
        w = self._g(prefix + "/conv/kernel")
        b = self._g(prefix + "/conv/bias")
        pt = _semicausal_pad(kh, strides[0], dil[0])
        pf = _sym_freq_pad(kw, strides[1], dil[1])
        x = F.pad(x, (pf[0], pf[1], pt[0], pt[1]))
        wk = w.permute(3, 2, 0, 1).to(x.dtype)
        return F.conv2d(x, wk, bias=b.to(x.dtype), stride=strides, dilation=dil)

    def _conv_transpose(self, x, prefix, kh, kw, strides):
        w = self._g(prefix + "/conv/kernel")  # [kh,kw,cin,cout]
        b = self._g(prefix + "/conv/bias")
        x = _dilate2d(x, strides)
        pt = _transpose_pad(kh, strides[0], "causal")
        pf = _transpose_pad(kw, strides[1], "same")
        x = F.pad(x, (pf[0], pf[1], pt[0], pt[1]))
        wk = w.permute(3, 2, 0, 1).to(x.dtype)
        return F.conv2d(x, wk, bias=b.to(x.dtype), stride=1)

    def _upsample(self, x, strides):
        if strides[0] > 1:
            x = x.repeat_interleave(strides[0], dim=2)
        if strides[1] > 1:
            x = x.repeat_interleave(strides[1], dim=3)
        return x

    def _residual_unit(self, x, prefix, strides, transposed_resample, kt):
        """act->[convT or conv3x3_a]->act->conv3x3 + shortcut."""
        inp = x
        y = elu(x)
        if transposed_resample:
            kh, kw = kt
            y = self._conv_transpose(y, prefix + "/conv2dtranspose_%dx%d" % (kh, kw), kh, kw, strides)
        else:
            y = self._conv2d(y, prefix + "/conv2d_3x3_a", 3, 3)
        y = elu(y)
        y = self._conv2d(y, prefix + "/conv2d_3x3", 3, 3)
        # shortcut
        sc = inp
        has_conv = (prefix + "/shortcut_layer/conv1x1/conv/kernel").replace("/", "__") in self.w
        if has_conv:
            sc = self._conv1x1(sc, prefix + "/shortcut_layer/conv1x1")
        if strides != (1, 1):
            sc = self._upsample(sc, strides)
        return y + sc

    def decode_embeddings(self, emb):
        """emb: [b,t,256] -> spectrogram feature map [b,4,T,480] (NCHW)."""
        b, t, _ = emb.shape
        x = emb.permute(0, 2, 1).unsqueeze(-1)  # [b,256,t,1]
        # input_layer residual
        main = self._conv1x1(x, "input_layer/conv1x1_first")
        sc = self._conv1x1(x, "input_layer/shortcut_layer/conv1x1_b1")
        sc = elu(sc)
        sc = self._conv1x1(sc, "input_layer/shortcut_layer/conv1x1_b2")
        x = main + sc  # [b,2560,t,1]
        # reshape (1,2560)->(5,512): [b,2560,t,1]->[b,5,512,t]->[b,512,t,5]
        x = x.squeeze(-1).view(b, INPUT_BINS, INPUT_CHANNELS, t).permute(0, 2, 3, 1)
        # input_layers_residual_unit (stride1)
        x = self._residual_unit(x, "input_layers_residual_unit", (1, 1), False, None)
        # decoder_0
        rev = RATIOS[::-1]
        kt0 = (max(3, 2 * rev[0][0]), max(3, 2 * rev[0][1]))
        x = self._residual_unit(x, "decoder_0", rev[0], True, kt0)
        # ParallelChannels(2): split channels, shared decoder_1..6 + output, concat
        groups = torch.chunk(x, CHANNEL_SPLITS, dim=1)
        outs = []
        for g in groups:
            h = g
            for i in range(1, len(RATIOS)):
                s = rev[i]
                kt = (max(3, 2 * s[0]), max(3, 2 * s[1]))
                h = self._residual_unit(h, f"decoder_{i}", s, True, kt)
            # output_layer: act -> conv7x7 (->2)
            h = elu(h)
            h = self._conv2d(h, "output_layer/base_conv_last", 7, 7)
            outs.append(h)
        x = torch.cat(outs, dim=1)  # [b,4,T,480]
        # lookahead trim
        trim = DECODER_LOOKAHEAD * TOTAL_TIME_STRIDE
        if trim:
            x = x[:, :, trim:, :]
        return x

    def forward(self, emb):
        x = self.decode_embeddings(emb)
        return self._istft(x)

    # ---- streaming decode (per-frame, stateful) — bit-exact-in-bf16 vs forward,
    #      FLOP-optimal (no overlap-save re-decode). state = mutable dict of caches. ----
    def _s_conv2d(self, x, prefix, kh, kw, st, key, strides=(1, 1), dil=(1, 1)):
        pt = _semicausal_pad(kh, strides[0], dil[0])
        pf = _sym_freq_pad(kw, strides[1], dil[1])
        c = st.get(key)
        if c is None:
            c = x.new_zeros(x.shape[0], x.shape[1], pt[0], x.shape[3])
        xc = torch.cat([c, x], dim=2)
        st[key] = xc[:, :, xc.shape[2] - pt[0]:, :] if pt[0] > 0 else c
        xp = F.pad(xc, (pf[0], pf[1], 0, pt[1]))
        w = self._g(prefix + "/conv/kernel"); b = self._g(prefix + "/conv/bias")
        return F.conv2d(xp, w.permute(3, 2, 0, 1).to(x.dtype), bias=b.to(x.dtype),
                        stride=strides, dilation=dil)

    def _s_conv_transpose(self, x, prefix, kh, kw, strides, st, key):
        sh, sw = strides
        pt = _transpose_pad(kh, sh, "causal"); pf = _transpose_pad(kw, sw, "same")
        ctx = (pt[0] + sh - 1) // sh + 1
        c = st.get(key)
        if c is None:
            c = x.new_zeros(x.shape[0], x.shape[1], ctx, x.shape[3])
        C = x.shape[2]
        xc = torch.cat([c, x], dim=2)
        st[key] = xc[:, :, xc.shape[2] - ctx:, :]
        xp = F.pad(_dilate2d(xc, strides), (pf[0], pf[1], pt[0], pt[1]))
        w = self._g(prefix + "/conv/kernel"); b = self._g(prefix + "/conv/bias")
        out = F.conv2d(xp, w.permute(3, 2, 0, 1).to(x.dtype), bias=b.to(x.dtype), stride=1)
        return out[:, :, out.shape[2] - C * sh:, :]

    def _s_resunit(self, x, prefix, strides, transposed, kt, st, key):
        inp = x; y = elu(x)
        if transposed:
            kh, kw = kt
            y = self._s_conv_transpose(y, prefix + "/conv2dtranspose_%dx%d" % (kh, kw), kh, kw, strides, st, key + "/ct")
        else:
            y = self._s_conv2d(y, prefix + "/conv2d_3x3_a", 3, 3, st, key + "/a")
        y = elu(y)
        y = self._s_conv2d(y, prefix + "/conv2d_3x3", 3, 3, st, key + "/b")
        sc = inp
        if (prefix + "/shortcut_layer/conv1x1/conv/kernel").replace("/", "__") in self.w:
            sc = self._conv1x1(sc, prefix + "/shortcut_layer/conv1x1")
        if strides != (1, 1):
            sc = self._upsample(sc, strides)
        return y + sc

    def _s_decode_emb(self, emb_new, st):
        b, t, _ = emb_new.shape
        x = emb_new.permute(0, 2, 1).unsqueeze(-1)
        main = self._conv1x1(x, "input_layer/conv1x1_first")
        sc = self._conv1x1(x, "input_layer/shortcut_layer/conv1x1_b1"); sc = elu(sc)
        sc = self._conv1x1(sc, "input_layer/shortcut_layer/conv1x1_b2")
        x = (main + sc).squeeze(-1).view(b, INPUT_BINS, INPUT_CHANNELS, t).permute(0, 2, 3, 1)
        x = self._s_resunit(x, "input_layers_residual_unit", (1, 1), False, None, st, "ilru")
        rev = RATIOS[::-1]
        x = self._s_resunit(x, "decoder_0", rev[0], True, (max(3, 2 * rev[0][0]), max(3, 2 * rev[0][1])), st, "d0")
        outs = []
        for gi, g in enumerate(torch.chunk(x, CHANNEL_SPLITS, dim=1)):
            h = g
            for i in range(1, len(RATIOS)):
                s = rev[i]
                h = self._s_resunit(h, f"decoder_{i}", s, True, (max(3, 2 * s[0]), max(3, 2 * s[1])), st, f"g{gi}/d{i}")
            h = elu(h)
            h = self._s_conv2d(h, "output_layer/base_conv_last", 7, 7, st, f"g{gi}/out")
            outs.append(h)
        return torch.cat(outs, dim=1)

    def _s_istft(self, xnew, st):
        v = xnew.permute(0, 2, 3, 1).contiguous(); b, T, nb, nc = v.shape
        if T == 0:
            return xnew.new_zeros(b, 0, 2)
        v = F.pad(v, (0, 0, 0, 1)).float()
        comp = torch.view_as_complex(v.view(b, T, 481, nc // 2, 2).contiguous())
        frames = torch.fft.irfft(comp, n=FFT_LENGTH, dim=2) * self.inv_window.view(1, 1, FRAME_LENGTH, 1)
        fr = frames.permute(0, 3, 1, 2)
        tail = st.get("_tail")
        if tail is None:
            tail = fr.new_zeros(b, 2, FRAME_STEP)
        emits = []
        for i in range(T):
            f = fr[:, :, i, :]; emits.append(tail + f[:, :, :FRAME_STEP]); tail = f[:, :, FRAME_STEP:]
        st["_tail"] = tail
        return torch.cat(emits, dim=2).permute(0, 2, 1)

    def decode_streaming(self, emb_new, state):
        """Incremental decode. `state` is a mutable dict (start with {}). Returns the
        newly-available audio [b, N, 2] for `emb_new` [b, t_new, 256], carrying overlap
        + per-layer conv state across calls. Output == forward(full_emb), 1 frame latency."""
        x = self._s_decode_emb(emb_new, state)
        wm = state.get("_warm", DECODER_LOOKAHEAD * TOTAL_TIME_STRIDE)
        if wm > 0:
            d = min(wm, x.shape[2]); x = x[:, :, d:, :]; state["_warm"] = wm - d
        return self._s_istft(x, state)

    def _istft(self, x):
        v = x.permute(0, 2, 3, 1).contiguous()      # [b,T,480,4]
        b, T, nb, nc = v.shape
        # pad freq (axis=2) 480 -> 481 (keep_dc: pad right)
        v = F.pad(v, (0, 0, 0, 1))                   # pad dim=2 right by 1 -> [b,T,481,4]
        v = v.float()
        comp = torch.view_as_complex(v.view(b, T, 481, nc // 2, 2).contiguous())  # [b,T,481,2]
        frames = torch.fft.irfft(comp, n=FFT_LENGTH, dim=2)  # [b,T,960,2]
        frames = frames * self.inv_window.view(1, 1, FRAME_LENGTH, 1)
        # overlap-add over (T,960); move channel out: [b,2,T,960]
        fr = frames.permute(0, 3, 1, 2)              # [b,2,T,960]
        wav = _overlap_and_add(fr, FRAME_STEP)       # [b,2,samples]
        trim = max(FRAME_LENGTH - FRAME_STEP, 0)
        if trim:
            wav = wav[..., :-trim]
        return wav.permute(0, 2, 1)                  # [b,samples,2]


def codes_to_embeddings(codes, quantizer_embedding):
    """codes: [b,t,Q] long (0..1023); quantizer_embedding: [64,1024,256]. Sum over levels."""
    Q = codes.shape[-1]
    out = None
    for i in range(Q):
        e = quantizer_embedding[i][codes[..., i]]  # [b,t,256]
        out = e if out is None else out + e
    return out


def load_spectrostream_decoder(checkpoint_path, dtype=torch.float32, prefix="params/soundstream"):
    """Load decoder + quantizer tensors from a safetensors file."""
    dec_weights = {}
    quant = None
    with safe_open(str(checkpoint_path), "numpy") as f:
        for k in f.keys():
            if k.startswith(prefix + "/decoder/"):
                name = k[len(prefix + "/decoder/"):]
                dec_weights[name] = torch.from_numpy(np.asarray(f.get_tensor(k))).to(dtype)
            elif k.startswith(prefix + "/quantizer/embedding"):
                quant = torch.from_numpy(np.asarray(f.get_tensor(k))).to(dtype)
    dec = SpectroStreamDecoder(dec_weights)
    return dec, quant
