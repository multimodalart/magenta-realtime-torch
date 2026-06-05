"""Compare torch SpectroStream decode (codes->waveform) against JAX reference."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from magenta_rt.torch.spectrostream import load_spectrostream_decoder, codes_to_embeddings
from magenta_rt import paths

SIZE = os.environ.get("MRT_SIZE", "mrt2_small")
ref = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ssref_{SIZE}.npz"))
codes = torch.from_numpy(ref["codes"]).long()
ref_emb = torch.from_numpy(ref["embeddings"])
ref_wav = torch.from_numpy(ref["waveform"])

ckpt = paths.checkpoints_dir() / f"{SIZE}.safetensors"
dec, quant = load_spectrostream_decoder(ckpt)
print("quant", tuple(quant.shape))

with torch.no_grad():
    emb = codes_to_embeddings(codes, quant)
    de = (emb - ref_emb).abs()
    print(f"embeddings torch {tuple(emb.shape)} ref {tuple(ref_emb.shape)} | max diff {de.max():.3e} mean {de.mean():.3e}")

    wav = dec(emb)
    print(f"waveform torch {tuple(wav.shape)} ref {tuple(ref_wav.shape)}")
    if tuple(wav.shape) == tuple(ref_wav.shape):
        dw = (wav - ref_wav).abs()
        denom = ref_wav.abs().max().clamp_min(1e-6)
        print(f"  waveform max abs diff {dw.max():.3e}  mean {dw.mean():.3e}  rel {dw.max()/denom:.3e}")
        # correlation
        a = wav.flatten().double(); b = ref_wav.flatten().double()
        corr = (a*b).mean() / (a.std()*b.std() + 1e-12)
        print(f"  corr {corr:.6f}")
