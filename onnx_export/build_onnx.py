"""Build the full ONNX set (fp32 + int8 + int4) for a Magenta RT2 size.
Usage: build_onnx.py {mrt2_small|mrt2_base}
Produces /tmp/onnx_repo_<size>/ : onnx/{encoder,temporal_step,depth_step,
spectrostream_decoder}.{onnx,q8.onnx,q4.onnx} + token/quant tables + config.json.
Every export is validated vs torch."""
import sys, os, json, shutil, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/ubuntu/magenta2/magenta-realtime")
from magenta_rt import paths
from magenta_rt.torch.system import MagentaRT2
from magenta_rt.torch.spectrostream import FFT_LENGTH, FRAME_LENGTH, FRAME_STEP, codes_to_embeddings
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
import onnx
sys.path.insert(0,'/tmp'); from einsum_rewrite import rewrite_einsum_to_matmul


SIZE = sys.argv[1] if len(sys.argv) > 1 else "mrt2_small"
OUT = f"/tmp/onnx_repo_{SIZE}"
OD = f"{OUT}/onnx"
os.makedirs(OD, exist_ok=True)
so = ort.SessionOptions(); so.log_severity_level = 3
sess = lambda p: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
import glob as _glob
def save_onnx(model, path):
    """Save with external data when >~1.8GB (ONNX 2GB protobuf limit)."""
    nb = sum(int(np.prod(i.dims) if i.dims else 1) * (2 if i.data_type in (10,16) else 4) for i in model.graph.initializer)
    for f in _glob.glob(path + "_data*"):
        try: os.remove(f)
        except OSError: pass
    if nb > 1.8e9:
        onnx.save(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                  location=os.path.basename(path) + "_data", size_threshold=1024)
    else:
        onnx.save(model, path)

print(f"=== building {SIZE} ===", flush=True)
mrt = MagentaRT2(size=SIZE, device="cpu", dtype=torch.float32)
cfg = mrt.cfg; m = mrt.model; dec = m.decoder; codec = mrt.dec  # codec=SpectroStreamDecoder
NR, CB, Q = cfg.num_reserved_tokens, cfg.codebook_size, cfg.num_codebooks
Lt, nht, upht = cfg.temporal.num_layers, cfg.temporal.num_heads, cfg.temporal.dim_per_head
Ld, nhd, uphd = cfg.depth.num_layers, cfg.depth.num_heads, cfg.depth.dim_per_head
Fb = FFT_LENGTH // 2 + 1

# ---- decoder (matmul ISTFT) ----
eye = torch.eye(Fb, dtype=torch.float32)
Cmat = torch.fft.irfft(eye.to(torch.complex64), n=FFT_LENGTH, dim=1)
Smat = torch.fft.irfft((1j * eye).to(torch.complex64), n=FFT_LENGTH, dim=1)
class ExportDecoder(nn.Module):
    def __init__(s, d): super().__init__(); s.d = d; s.register_buffer("C", Cmat); s.register_buffer("S", Smat)
    def forward(s, emb):
        x = s.d.decode_embeddings(emb); v = x.permute(0, 2, 3, 1); b = v.shape[0]; T = v.shape[1]
        v = F.pad(v, (0, 0, 0, 1)); vc = v.reshape(b, T, Fb, 2, 2)
        fr = torch.einsum('bfkc,kt->bftc', vc[..., 0], s.C) + torch.einsum('bfkc,kt->bftc', vc[..., 1], s.S)
        fr = fr * s.d.inv_window.view(1, 1, FRAME_LENGTH, 1); fr = fr.permute(0, 3, 1, 2)
        A = fr[..., :FRAME_STEP]; B = fr[..., FRAME_STEP:]
        out = F.pad(A, (0, 0, 0, 1)) + F.pad(B, (0, 0, 1, 0))
        wav = out.reshape(b, 2, (T + 1) * FRAME_STEP)[..., :-(FRAME_LENGTH - FRAME_STEP)]
        return wav.permute(0, 2, 1)
codes0 = torch.randint(0, CB, (1, 30, Q)); emb0 = codes_to_embeddings(codes0, mrt.quant)
edec = ExportDecoder(codec).eval()
with torch.no_grad(): rdec = edec(emb0)
torch.onnx.export(edec, (emb0,), f"{OD}/spectrostream_decoder.onnx", input_names=["embeddings"],
    output_names=["waveform"], dynamic_axes={"embeddings": {1: "T"}, "waveform": {1: "S"}}, opset_version=18, do_constant_folding=True)
print("  decoder:", f'{np.abs(rdec.numpy()-sess(f"{OD}/spectrostream_decoder.onnx").run(None,{"embeddings":emb0.numpy()})[0]).max():.2e}', flush=True)

# ---- temporal + depth steps (stacked KV) ----
class StTemporal(nn.Module):
    def __init__(s, d): super().__init__(); s.d = d
    def forward(s, prev, sk, sv, ck, cv, src):
        L = sk.shape[0]; o, ns, nc = s.d.temporal_step_fn(prev, [(sk[i], sv[i]) for i in range(L)], [(ck[i], cv[i]) for i in range(L)], src)
        return (o, torch.stack([x[0] for x in ns]), torch.stack([x[1] for x in ns]), torch.stack([x[0] for x in nc]), torch.stack([x[1] for x in nc]))
class StDepth(nn.Module):
    def __init__(s, d): super().__init__(); s.d = d
    def forward(s, di, dk, dv):
        L = dk.shape[0]; lo, nk = s.d.depth_step_fn(di, [(dk[i], dv[i]) for i in range(L)])
        return lo, torch.stack([x[0] for x in nk]), torch.stack([x[1] for x in nk])
prev = torch.randint(NR, NR + CB, (1, 1, Q)); src = torch.randn(1, 1, cfg.encoder_model_dims)
sk = torch.randn(Lt, 1, 5, nht, upht); ck = torch.randn(Lt, 1, 5, nht, upht)
di = torch.randn(1, 1, cfg.temporal.model_dims); dk = torch.randn(Ld, 1, 3, nhd, uphd)
st, sd = StTemporal(dec).eval(), StDepth(dec).eval()
with torch.no_grad(): rt, rd = st(prev, sk, sk, ck, ck, src), sd(di, dk, dk)
torch.onnx.export(st, (prev, sk, sk, ck, ck, src), f"{OD}/temporal_step.onnx",
    input_names=["prev", "self_k", "self_v", "cross_k", "cross_v", "source"],
    output_names=["out", "new_self_k", "new_self_v", "new_cross_k", "new_cross_v"],
    dynamic_axes={k: {2: "T"} for k in ["self_k", "self_v", "cross_k", "cross_v", "new_self_k", "new_self_v", "new_cross_k", "new_cross_v"]}, opset_version=18, do_constant_folding=True)
torch.onnx.export(sd, (di, dk, dk), f"{OD}/depth_step.onnx", input_names=["depth_input", "depth_k", "depth_v"],
    output_names=["logits", "new_depth_k", "new_depth_v"],
    dynamic_axes={k: {2: "Td"} for k in ["depth_k", "depth_v", "new_depth_k", "new_depth_v"]}, opset_version=18, do_constant_folding=True)
to = sess(f"{OD}/temporal_step.onnx").run(None, {"prev": prev.numpy(), "self_k": sk.numpy(), "self_v": sk.numpy(), "cross_k": ck.numpy(), "cross_v": ck.numpy(), "source": src.numpy()})
do = sess(f"{OD}/depth_step.onnx").run(None, {"depth_input": di.numpy(), "depth_k": dk.numpy(), "depth_v": dk.numpy()})
print("  temporal:", f"{np.abs(rt[0].numpy()-to[0]).max():.2e}", "| depth:", f"{np.abs(rd[0].numpy()-do[0]).max():.2e}", flush=True)

# ---- encoder ----
class Enc(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, cond): return s.m.encode(cond)
off = NR + 1; vals = [100] * cfg.musiccoca_rvq + [-1] * 128 + [-1] + [20, 10, 8]
cond = torch.from_numpy((np.array(vals, np.int64) + off).reshape(1, 1, -1))
enc = Enc(m).eval()
with torch.no_grad(): rs = enc(cond)
torch.onnx.export(enc, (cond,), f"{OD}/encoder.onnx", input_names=["cond"], output_names=["source"], opset_version=18, do_constant_folding=True)
print("  encoder:", f'{np.abs(rs.numpy()-sess(f"{OD}/encoder.onnx").run(None,{"cond":cond.numpy()})[0]).max():.2e}', flush=True)

# ---- quantize int8 + int4 ----
for g in ["spectrostream_decoder", "temporal_step", "depth_step", "encoder"]:
    s = f"{OD}/{g}.onnx"
    if g != "spectrostream_decoder":              # conv decoder has no einsums
        mm, nconv = rewrite_einsum_to_matmul(onnx.load(s)); save_onnx(mm, s)   # einsum->MatMul in place
        print(f"  {g}: rewrote {nconv} einsum->matmul", flush=True)
    quantize_dynamic(s, f"{OD}/{g}.q8.onnx", weight_type=QuantType.QInt8, use_external_data_format=True)
    q4 = MatMulNBitsQuantizer(onnx.load(s), block_size=32, is_symmetric=True); q4.process()
    save_onnx(q4.model.model, f"{OD}/{g}.q4.onnx")
print("  quantized q8 + q4", flush=True)

# ---- tables + config ----
dec.embedding.detach().numpy().astype(np.float16).tofile(f"{OD}/token_embedding_fp16.bin")
mrt.quant.detach().numpy().astype(np.float16).tofile(f"{OD}/quantizer_fp16.bin")
config = {"model_type": "magenta_rt2", "size": SIZE, "sample_rate": 48000, "frames_per_second": 25,
    "num_codebooks": Q, "codebook_size": CB, "num_reserved_tokens": NR, "vocab_size": cfg.vocab_size,
    "model_dims": cfg.temporal.model_dims, "encoder_model_dims": cfg.encoder_model_dims,
    "temporal": {"num_layers": Lt, "num_heads": nht, "dim_per_head": upht, "max_past": cfg.temporal_max_past},
    "depth": {"num_layers": Ld, "num_heads": nhd, "dim_per_head": uphd},
    "quantizer_shape": list(mrt.quant.shape), "embedding_shape": list(dec.embedding.shape),
    "conditioning": {"musiccoca_rvq": cfg.musiccoca_rvq, "num_notes": 128, "num_drums": 1, "num_cfg": 3, "cond_offset": off}}
json.dump(config, open(f"{OUT}/config.json", "w"), indent=2)
print(f"  done -> {OUT}", flush=True)
