"""Deterministic (top_k=1) JAX generation reference for end-to-end parity."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _compat_shim  # noqa
import numpy as np
import jax
import magenta_rt.jax.system as jsys
from magenta_rt.jax.system import MagentaRT2System

# Intercept the raw depthformer tokens (input to convert_from_unique_codes).
_captured = []
_orig_convert = jsys.convert_from_unique_codes
def _spy(tokens, codebook_size=1024):
    jax.debug.callback(lambda t: _captured.append(np.asarray(t)), tokens)
    return _orig_convert(tokens, codebook_size)
jsys.convert_from_unique_codes = _spy

SIZE = os.environ.get("MRT_SIZE", "mrt2_small")
F32 = os.environ.get("MRT_F32", "0") == "1"
if F32:
    import jax.numpy as jnp
    from magenta_rt.jax import model as jmodel
    jmodel.MagentaRT2ModelBase.compute_dtype = jnp.float32
    jmodel.MagentaRT2ModelSmall.compute_dtype = jnp.float32
mrt = MagentaRT2System(size=SIZE, top_k=1, temperature=1.0)
emb = mrt.embed_style("disco funk")
style_tokens = np.asarray(mrt.tokenize_style(emb)).astype(np.int64)
wav, _ = mrt.generate(style=emb, frames=5, top_k=1, temperature=1.0)
tokens = np.stack([np.asarray(t).reshape(-1) for t in _captured], axis=0)  # [frames, Q]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"genref_{SIZE}{'_f32' if F32 else ''}.npz")
np.savez(out, style_tokens=style_tokens, waveform=wav.samples.astype(np.float32), tokens=tokens)
print("style_tokens", style_tokens.tolist())
print("tokens", tokens.shape)
print(tokens)
print("wrote", out)
