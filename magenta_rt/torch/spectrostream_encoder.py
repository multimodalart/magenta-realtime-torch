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

"""PyTorch port of the SpectroStream encoder (48kHz stereo audio -> RVQ codes).

Mirror of the decoder: STFT front-end -> conv2d downsample ResNet (with
channel_splits=2) -> bottleneck -> 256-d features -> residual vector quantize.
NCHW feature maps [b, C, T(time), F(freq)].
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .spectrostream import (_semicausal_pad, _sym_freq_pad, _hann_window, elu,
                            RATIOS, CHANNEL_SPLITS, FRAME_LENGTH, FRAME_STEP, FFT_LENGTH)

NUM_INPUT_BINS = 480
NUM_FEATURES = 256
# encoder block strides = forward ratios (decoder reverses them); recombo at block 6.


def stft_forward(wav, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FFT_LENGTH):
    """wav [b,S,2] -> spectrogram NCHW [b,4,T,480] (reverse-causal STFT, keep_dc)."""
    b, S, C = wav.shape
    wavp = F.pad(wav, (0, 0, 0, frame_length - 1))  # reverse_causal: pad time right
    nf = (wavp.shape[1] - frame_length) // frame_step + 1
    idx = (torch.arange(frame_length, device=wav.device)[None, :]
           + torch.arange(nf, device=wav.device)[:, None] * frame_step)
    frames = wavp[:, idx, :]                          # [b,nf,frame_length,2]
    win = torch.from_numpy(_hann_window(frame_length)).to(wav.device, wav.dtype)
    frames = frames * win[None, None, :, None]
    spec = torch.fft.rfft(frames.float(), n=fft_length, dim=2)  # [b,nf,481,2] complex
    sf = torch.view_as_real(spec).reshape(b, nf, fft_length // 2 + 1, C * 2)  # bitcast
    sf = sf[:, :, :-1, :]                            # keep_dc: drop Nyquist -> [b,nf,480,4]
    return sf.permute(0, 3, 1, 2).contiguous()        # [b,4,T,480]


class SpectroStreamEncoder(nn.Module):
    def __init__(self, weights: dict):
        super().__init__()
        self.w = nn.ParameterDict({k.replace("/", "__"): nn.Parameter(v, requires_grad=False)
                                   for k, v in weights.items()})

    def _g(self, name):
        return self.w[name.replace("/", "__")]

    def _conv(self, x, prefix, kh, kw, strides=(1, 1)):
        w = self._g(prefix + "/conv/kernel")
        bdat = self._g(prefix + "/conv/bias")
        pt = _semicausal_pad(kh, strides[0])
        pf = _sym_freq_pad(kw, strides[1])
        x = F.pad(x, (pf[0], pf[1], pt[0], pt[1]))
        return F.conv2d(x, w.permute(3, 2, 0, 1).to(x.dtype), bias=bdat.to(x.dtype), stride=strides)

    def _avgpool(self, x, strides):
        # semicausal time pad, valid freq; pool_size=strides
        pt = _semicausal_pad(strides[0], strides[0])
        x = F.pad(x, (0, 0, pt[0], pt[1]))
        return F.avg_pool2d(x, kernel_size=strides, stride=strides)

    def _block(self, x, prefix, strides, kt):
        """non-transposed residual unit: act->conv3x3(in)->act->convKxK_a(out,strided) + shortcut."""
        inp = x
        y = elu(x)
        y = self._conv(y, prefix + "/conv2d_3x3", 3, 3)
        y = elu(y)
        kh, kw = kt
        y = self._conv(y, prefix + "/conv2d_%dx%d_a" % (kh, kw), kh, kw, strides)
        sc = inp
        has_sc = (prefix + "/shortcut_layer/conv1x1/conv/kernel").replace("/", "__") in self.w
        if strides != (1, 1):
            sc = self._avgpool(sc, strides)
        if has_sc:
            w = self._g(prefix + "/shortcut_layer/conv1x1/conv/kernel")
            bdat = self._g(prefix + "/shortcut_layer/conv1x1/conv/bias")
            sc = F.conv2d(sc, w.permute(3, 2, 0, 1).to(sc.dtype), bias=bdat.to(sc.dtype))
        return y + sc

    def _bottleneck(self, x, prefix):
        # non-transposed stride-1 unit: conv2d_3x3 first, then conv2d_3x3_a.
        inp = x
        y = elu(x)
        y = self._conv(y, prefix + "/conv2d_3x3", 3, 3)
        y = elu(y)
        y = self._conv(y, prefix + "/conv2d_3x3_a", 3, 3)
        return y + inp

    def _group_stack(self, x):
        """base_conv + encoder_0..5 for one channel-split group."""
        x = self._conv(x, "base_conv_first", 7, 7)
        fwd = RATIOS  # forward order for encoder
        for i in range(6):  # blocks 0..5 are inside the groups (recombo at 6)
            s = fwd[i]
            kt = (max(3, 2 * s[0]), max(3, 2 * s[1]))
            x = self._block(x, f"encoder_{i}", s, kt)
        return x

    def forward_features(self, wav):
        """wav [b,S,2] -> embeddings [b,T,256]."""
        x = stft_forward(wav)                         # [b,4,T,480]
        groups = torch.chunk(x, CHANNEL_SPLITS, dim=1)  # 2 x [b,2,T,480]
        outs = [self._group_stack(g) for g in groups]
        x = torch.cat(outs, dim=1)                    # concat -> [b,512,T,5]
        s = RATIOS[6]
        kt = (max(3, 2 * s[0]), max(3, 2 * s[1]))
        x = self._block(x, "encoder_6", s, kt)        # [b,256,T,5]
        x = self._bottleneck(x, "bottleneck")
        # Flatten freq into channels: [b,256,T,5] -> [b,T,1280]
        b, c, t, fbin = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(b, t, fbin * c)  # jax flatten order (freq,channel)
        # output_convs residual on [b,T,1,1280] (NCHW with C=1280,F=1)
        h = flat.permute(0, 2, 1).unsqueeze(-1)        # [b,1280,T,1]
        main = self._conv1x1(h, "output_convs/conv1x1_last")
        sc = self._conv1x1(elu(h), "output_convs/shortcut_layer/conv1x1_b1")
        sc = self._conv1x1(elu(sc), "output_convs/shortcut_layer/conv1x1_b2")
        out = main + sc                                # [b,256,T,1]
        return out.squeeze(-1).permute(0, 2, 1)        # [b,T,256]

    def _conv1x1(self, x, prefix):
        w = self._g(prefix + "/conv/kernel")
        bdat = self._g(prefix + "/conv/bias")
        return F.conv2d(x, w.permute(3, 2, 0, 1).to(x.dtype), bias=bdat.to(x.dtype))


def rvq_encode(embeddings, quantizer_embedding, num_levels=12):
    """embeddings [b,T,256] -> codes [b,T,num_levels] via residual nearest-neighbor."""
    residual = embeddings
    codes = []
    for i in range(num_levels):
        cb = quantizer_embedding[i]                    # [1024,256]
        # squared L2 distance argmin
        d = (residual.pow(2).sum(-1, keepdim=True)
             - 2 * residual @ cb.t()
             + cb.pow(2).sum(-1)[None, None, :])        # [b,T,1024]
        idx = d.argmin(-1)                              # [b,T]
        codes.append(idx)
        residual = residual - cb[idx]
    return torch.stack(codes, dim=-1)


def load_spectrostream_encoder(encoder_path, dtype=torch.float32):
    w = {}
    with safe_open(str(encoder_path), "numpy") as f:
        for k in f.keys():
            if k.startswith("params/encoder/"):
                w[k[len("params/encoder/"):]] = torch.from_numpy(np.asarray(f.get_tensor(k))).to(dtype)
    return SpectroStreamEncoder(w)
