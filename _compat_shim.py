"""Compatibility shim so the vendored sequence_layers (written for py3.11+ and a
newer JAX) imports under python 3.10 / jax 0.6.2. Import this BEFORE magenta_rt.
Used only to run the JAX reference as a parity oracle for the torch port."""
import typing
import typing_extensions

for _n in ("Self", "Never", "Unpack", "TypeVarTuple", "ParamSpec", "TypeAlias",
           "assert_never", "override", "dataclass_transform"):
    if not hasattr(typing, _n) and hasattr(typing_extensions, _n):
        setattr(typing, _n, getattr(typing_extensions, _n))

import jax
import jax.sharding as _sharding
if not hasattr(jax, "set_mesh"):
    jax.set_mesh = _sharding.set_mesh

# Stub resampy (pulls numba, which conflicts with the box's numpy 2.2). The
# oracle only needs transformer forward, not audio resampling.
import sys as _sys
import types as _types
if "resampy" not in _sys.modules:
    try:
        import resampy  # noqa: F401
    except Exception:
        _stub = _types.ModuleType("resampy")
        def _resample(*a, **k):
            raise RuntimeError("resampy stubbed out in oracle shim")
        _stub.resample = _resample
        _sys.modules["resampy"] = _stub

