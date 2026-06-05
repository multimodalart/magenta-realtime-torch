"""Compare torch Depthformer teacher-forced logits against the JAX fp32 reference."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from magenta_rt.torch.depthformer import Depthformer, config_for
from magenta_rt.torch.weights import load_depthformer
from magenta_rt import paths

SIZE = os.environ.get("MRT_SIZE", "mrt2_small")
ref = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ref_{SIZE}_f32.npz"))
cond = torch.from_numpy(ref["cond"]).long()
target = torch.from_numpy(ref["target"]).long()
ref_source = torch.from_numpy(ref["source"])
ref_logits = torch.from_numpy(ref["logits"])

cfg = config_for(SIZE)
model = Depthformer(cfg).eval()
ckpt = paths.checkpoints_dir() / f"{SIZE}.safetensors"
load_depthformer(model, ckpt, dtype=torch.float32, verbose=True)

with torch.no_grad():
    source = model.encode(cond)
    print("source torch", tuple(source.shape), "ref", tuple(ref_source.shape))
    ds = (source - ref_source).abs()
    print(f"  source max abs diff {ds.max():.3e}  mean {ds.mean():.3e}")

    logits = model.decoder(target, source)
    dl = (logits - ref_logits).abs()
    print("logits torch", tuple(logits.shape))
    print(f"  logits max abs diff {dl.max():.3e}  mean {dl.mean():.3e}")
    # argmax agreement within each codebook's valid range
    cb, res = cfg.codebook_size, cfg.num_reserved_tokens
    agree = 0; total = 0
    for q in range(cfg.num_codebooks):
        lo, hi = res + q * cb, res + (q + 1) * cb
        a = logits[..., q, lo:hi].argmax(-1)
        b = ref_logits[..., q, lo:hi].argmax(-1)
        agree += (a == b).sum().item(); total += a.numel()
    print(f"  per-codebook argmax agreement {agree}/{total} = {agree/total:.3f}")

    # streaming (KV-cache step) path should reproduce teacher-forced logits
    s_logits = model.decoder.streaming_logits(target, source)
    sd = (s_logits - logits).abs()
    print(f"  streaming-vs-teacherforced logits max abs diff {sd.max():.3e} mean {sd.mean():.3e}")
