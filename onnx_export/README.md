# ONNX / Transformers.js export

Builds the browser-deployable ONNX set (fp32 + int8 + int4) for Magenta RealTime 2,
validated bit-exact vs the torch port.

- `build_onnx.py {mrt2_small|mrt2_base}` — exports the 4 graphs (encoder, temporal step,
  depth step, SpectroStream decoder), rewrites attention/FFN einsums to MatMul so int4
  (MatMulNBits) and int8 cover them, quantizes (q8 + q4), writes tables + config, with
  external-data saves for >2GB graphs.
- `einsum_rewrite.py` — Einsum(constant weight) -> Reshape+MatMul graph rewrite.
- `../web/index.html` — onnxruntime-web demo (WebGPU/WASM, q8/q4/fp32), Cache-API persisted.

Notes: decoder is conv-based -> kept fp32 in-browser (int8 Conv -> ConvInteger, unsupported
in onnxruntime-web). The per-frame generation loop (temporal -> 12x depth -> sample ->
embed-gather -> decode) is hand-written in JS; the full pipeline is 100% token-identical
to torch (validated in onnxruntime-Python).
Artifacts: magenta-torch/magenta-rt-onnx-{small,base}; demo: magenta-torch/magenta-rt-web.
