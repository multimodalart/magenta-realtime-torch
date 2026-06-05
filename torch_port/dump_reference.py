"""Dump deterministic JAX reference tensors for torch-port parity testing.

Captures, for a fixed synthetic input:
  - the encoder `source` encoding of a conditioning block
  - the decoder teacher-forced logits  [B, T, Q, vocab]
plus all inputs, into a .npz the torch port can compare against.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _compat_shim  # noqa
import numpy as np
import jax, jax.numpy as jnp
from magenta_rt.jax.system import MagentaRT2System
import sequence_layers.jax as sl

SIZE = os.environ.get("MRT_SIZE", "mrt2_small")
F32 = os.environ.get("MRT_F32", "0") == "1"
T = 6  # frames

if F32:
    # Force fp32 compute for a tight correctness oracle.
    from magenta_rt.jax import model as jmodel
    jmodel.MagentaRT2ModelBase.compute_dtype = jnp.float32
    jmodel.MagentaRT2ModelSmall.compute_dtype = jnp.float32

mrt = MagentaRT2System(size=SIZE)
model = mrt._model
dec_cfg = mrt._sampler.cfg.depthformer.decoder
num_channels = mrt._num_channels
Q = dec_cfg.num_codebooks
cb = dec_cfg.codebook_size
res = dec_cfg.num_reserved_tokens
vocab = Q * cb + res
print(f"size={SIZE} num_channels={num_channels} Q={Q} cb={cb} res={res} vocab={vocab}")

rng = np.random.RandomState(0)
# Conditioning block [B,T,num_channels] int32: use small valid indices.
cond = rng.randint(7, 17, size=(1, T, num_channels)).astype(np.int32)
# Target tokens [B,T,Q] int32: each codebook q in [res+q*cb, res+(q+1)*cb).
target = np.zeros((1, T, Q), np.int32)
for q in range(Q):
    target[..., q] = rng.randint(res + q * cb, res + (q + 1) * cb, size=(1, T))

bound = mrt._sampler.bind(mrt._params)
cond_seq = sl.Sequence.from_values(jnp.asarray(cond))
source = bound.depthformer.encoder.body.layer(cond_seq, training=False)
print("source", source.values.shape, source.values.dtype)

cname = mrt._sampler.cfg.depthformer.conditioning_name or "source"
tgt_seq = sl.Sequence.from_values(jnp.asarray(target))
logits, emits = bound.depthformer.decoder.layer_with_emits(
    tgt_seq, training=False, constants={cname: source}
)
print("logits", logits.values.shape, logits.values.dtype)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ref_{SIZE}{'_f32' if F32 else ''}.npz")
np.savez(
    out,
    cond=cond, target=target,
    source=np.asarray(source.values, np.float32),
    logits=np.asarray(logits.values, np.float32),
)
print("wrote", out)
