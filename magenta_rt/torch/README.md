# Magenta RealTime 2 — PyTorch port

A pure-PyTorch port of [Magenta RealTime 2](https://huggingface.co/google/magenta-realtime-2)
(`google/magenta-realtime-2`). It loads the published JAX/Linen `.safetensors`
checkpoints directly and reproduces the reference generation numerically — no
`jax` or `mlx` runtime dependency.

## What's here

| File | Contents |
|------|----------|
| `layers.py` | Core primitives: primer-hybrid RMSNorm, JAX-layout linear/einsum projections, local windowed attention with attention sinks + per-dim softplus query scale (NoPE), gated/non-gated FFN. |
| `depthformer.py` | The decoder-only LLM: conditioning encoder (MusicCoCa + multi-channel embedders), temporal transformer (self + streaming cross-attention to the conditioning), depth transformer (per-frame RVQ autoregression), soft-capped logits. Teacher-forced `forward` + streaming `step`. |
| `spectrostream.py` | SpectroStream neural codec **decoder** (RVQ dequant → conv2d/conv2dtranspose ResNet with `channel_splits` → ISTFT) and the residual-VQ `codes_to_embeddings`. |
| `weights.py` | Maps the slash-delimited JAX checkpoint keys onto the torch modules. |
| `system.py` | `MagentaRT2`: conditioning assembly, streaming generation (temperature / top-k / gumbel sampling, CFG-as-tokens), audio decode. MusicCoCa style embedding is reused from the framework-agnostic TFLite component (as upstream JAX/MLX do). |

## Architecture (mrt2_small / mrt2_base)

- **Conditioning encoder** = embedding only (no transformer body): MusicCoCa
  dequantizer (12 RVQ tokens → 768-d, summed, projected) mean-combined with a
  multi-channel embedder over notes(128) + drums(1) + CFG(3) channels, then a
  LayerNorm. Produces the cross-attention `source`.
- **Temporal body**: 12 (small) / 20 (base) layers, dim 1024/3072, windowed
  causal self-attention (past horizon 41/25) with 1 attention sink, streaming
  cross-attention to `source`, gated-off GeGLU FFN, primer-hybrid pre+post
  RMSNorm, NoPE.
- **Depth body**: 2 (small) / 6 (base) layers over the 12 RVQ levels per frame,
  causal self-attention, final LayerNorm + linear to the 12294-token vocab,
  `tanh`-soft-capped at 30.
- **SpectroStream**: 48 kHz stereo, 25 Hz frames, RVQ 64×1024×256 (12 used),
  STFT front-end (960/480/960), causal conv stack with `channel_splits=2`.

## Verified against the JAX reference

Run from the repo root (the `torch_port/` harness builds fp32 references via the
JAX implementation and compares):

```
python torch_port/test_parity.py          # depthformer logits + streaming
python torch_port/test_spectrostream.py    # codec decode
```

Measured (mrt2_small, fp32):

- Depthformer teacher-forced logits: max abs diff **5.8e-5**, per-codebook
  argmax agreement **100%**.
- Streaming step path vs teacher-forced: max abs diff **9e-5**.
- SpectroStream `codes → waveform`: max abs diff **2.7e-6**, correlation **1.0**.
- Deterministic (top-k=1) end-to-end generation: **60/60 tokens identical** to
  the JAX system in fp32. (In bf16 — the upstream default compute dtype —
  close-call argmaxes can differ, as expected.)

## Usage

```python
from magenta_rt.torch import MagentaRT2

mrt = MagentaRT2(size="mrt2_small", device="cuda")   # weights from the HF cache
wav, state = mrt.generate(style="disco funk", frames=25)   # needs MusicCoCa TFLite
# or drive it with explicit 12 MusicCoCa style tokens (no TFLite needed):
wav, state = mrt.generate(style=[660,597,668,315,857,217,930,175,655,343,534,137], frames=25)
import soundfile as sf; sf.write("out.wav", wav, mrt.sample_rate)
```

### Live / continuous streaming

`generate` is resumable: pass the returned `state` back in to continue
seamlessly, change the conditioning between calls to steer the stream live, and
each call returns only the newly-available audio chunk.

```python
mrt = MagentaRT2(size="mrt2_small", device="cuda", compile=True)  # ~real-time
state = None
for i in range(N):
    style = "disco funk" if i < 5 else "ambient pads"     # change prompt mid-stream
    chunk, state = mrt.generate(style=style, frames=25, state=state, flush=(i == N - 1))
    play(chunk)                                            # ~1s of 48kHz stereo audio
```

Streaming audio uses overlap-save decoding (a small left context + 2-frame right
margin), which is **bit-exact** vs the offline decode (≤1 int16 LSB; on GPU,
cuDNN conv nondeterminism adds ~1e-4, inaudible). `compile=True` torch.compiles
the per-frame step (one-time ~170 s warmup, dynamic shapes) and reaches
**~1.27× real-time / 31.8 fps** on an A100 for `mrt2_small` (eager is ~0.76×).

## Notes

- Weights are read from `~/Documents/Magenta/magenta-rt-v2/checkpoints/` (the
  same location the upstream `paths.py` resolves; download with `huggingface_hub`).
- Audio decode uses the offline (whole-sequence) SpectroStream path, which
  matches the JAX offline decode bit-for-bit. The JAX *streaming* generate emits
  one extra warm-up frame (`T·1920` vs `(T-1)·1920` samples) due to its
  stateful codec; the audio content is identical.
- Generation is heavy on CPU (per-frame autoregression); use a GPU.
