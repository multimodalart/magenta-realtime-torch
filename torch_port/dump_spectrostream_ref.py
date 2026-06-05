"""Dump JAX SpectroStream reference: codes -> embeddings -> waveform, for parity."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _compat_shim  # noqa
import numpy as np
import jax.numpy as jnp
from magenta_rt.jax.system import MagentaRT2System, convert_from_unique_codes
import sequence_layers.jax as sl

SIZE = os.environ.get("MRT_SIZE", "mrt2_small")
mrt = MagentaRT2System(size=SIZE)
bound = mrt._sampler.bind(mrt._params)

# Deterministic codes [1,T,12] in non-unique form (0..1023).
T = 8
rng = np.random.RandomState(1)
codes = rng.randint(0, 1024, size=(1, T, 12)).astype(np.int32)

q = bound.soundstream.quantizer
emb_layer = q.codes_to_embeddings_layer
wav_layer = bound.soundstream.embeddings_to_waveform_layer

codes_seq = sl.Sequence.from_values(jnp.asarray(codes))
emb = emb_layer.layer(codes_seq, training=False)
print("embeddings", emb.values.shape, emb.values.dtype)
wav = wav_layer.layer(emb, training=False)
print("waveform", wav.values.shape, wav.values.dtype)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ssref_{SIZE}.npz")
np.savez(out, codes=codes,
         embeddings=np.asarray(emb.values, np.float32),
         waveform=np.asarray(wav.values, np.float32))
print("wrote", out)
