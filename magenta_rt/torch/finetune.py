"""LoRA fine-tuning for Magenta RealTime 2 (torch port), via 🤗 peft.

Teacher-forced next-token training on the Depthformer:
  audio        --SpectroStream encoder + RVQ-->  target codes [b,T,Q] (global ids)
  conditioning --InputEmbedder.encode-------->   source [b,Tc,enc]
  logits = depthformer(cond, target)             # [b,T,Q,vocab]; logits[t] predicts target[t]
  loss   = cross_entropy(logits, target)

Three data modes (per manifest row):
  - caption : text style caption -> MusicCoCa style tokens (constant conditioning)
  - midi    : MIDI -> per-frame note vector (CrossAttention is causal, so leak-free)
  - raw     : style+notes+drums masked -> self-supervised continuation

LoRA via peft on the activation-free JaxLinear layers, which we first convert to nn.Linear
(EXACT: weight = kernel.T) so peft can wrap them. Targets: ffn_layer2 (MLP down),
depth_input_adapter, to_logits. NOT covered: attention q/k/v/o (fused multi-head params,
not nn.Linear) and ffn_layer1 (fused gelu) — a v1 limitation; extend by splitting those.
peft freezes the base; adapters save as standard adapter_model.safetensors + adapter_config.json.

Manifest = JSONL, one object per line:
  {"audio": "a.wav"}                      -> raw
  {"audio": "a.wav", "caption": "funk"}   -> caption
  {"audio": "a.wav", "midi": "a.mid"}     -> midi (optionally + caption)

Train:
  python -m magenta_rt.torch.finetune --manifest data.jsonl --model small \\
      --encoder-path ~/.../resources/spectrostream/encoder.safetensors --steps 2000 --out lora_out

Load a trained adapter (the swap MUST be reapplied identically — it is deterministic):
  from transformers import AutoModel; from peft import PeftModel
  from magenta_rt.torch.finetune import swap_jaxlinear_to_linear
  base = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16)
  swap_jaxlinear_to_linear(base.depthformer)            # same SWAP_SUFFIXES as training
  model = PeftModel.from_pretrained(base, "lora_out")
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectrostream_encoder import load_spectrostream_encoder, rvq_encode

NUM_RESERVED_TOKENS = 6          # raw code -> global id: code + NUM_RESERVED + q*codebook_size
NUM_NOTES = 128                  # conditioning note channel (MIDI pitch range)
NUM_MUSICCOCA = 12               # style RVQ tokens
SR = 48000
FPS = 25.0                       # SpectroStream frame rate (40 ms/frame); for MIDI alignment
SWAP_SUFFIXES = ("ffn_layer2", "depth_input_adapter", "to_logits")  # activation-free JaxLinear
MODEL_REPOS = {
    "small": "magenta-community/magenta-realtime-2-small",
    "base": "magenta-community/magenta-realtime-2",
}


def _dformer(model):
    """Depthformer submodule (transformers class -> `depthformer`, demo class -> `model`).
    Callable: `dformer(target, source) -> logits`; has `.encode` and `.decoder`."""
    return getattr(model, "depthformer", None) or model.model


def swap_jaxlinear_to_linear(root, suffixes=SWAP_SUFFIXES):
    """In-place replace targeted activation-free JaxLinear (kernel [in,out], y=x@kernel+b)
    with an EXACT nn.Linear (weight = kernel.T). Deterministic — reapply identically before
    PeftModel.from_pretrained. Skips any JaxLinear with a fused activation. Returns names."""
    todo, conv = [], []
    for _, module in root.named_modules():
        for cname, child in module.named_children():
            if type(child).__name__ == "JaxLinear" and cname in suffixes:
                if getattr(child, "activation", None) is not None:
                    continue                                   # never drop a fused activation
                todo.append((module, cname, child))
    for module, cname, jl in todo:
        in_f, out_f = jl.kernel.shape
        lin = nn.Linear(in_f, out_f, bias=jl.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(jl.kernel.t())                    # x@kernel+b == x@W.T+b
            if jl.bias is not None:
                lin.bias.copy_(jl.bias)
        setattr(module, cname, lin.to(jl.kernel.dtype).to(jl.kernel.device))
        conv.append(cname)
    return conv


# ---------------------------------------------------------------- audio / midi io
def load_audio(path, target_sr=SR):
    """-> float32 [N,2] @ target_sr. torchaudio (wav/mp3/flac) with soundfile fallback."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)          # [C,N]
    except Exception:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # [N,C]
        wav = torch.from_numpy(data).T           # [C,N]
    if sr != target_sr:
        import torchaudio.functional as AF
        wav = AF.resample(wav, sr, target_sr)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav.T.contiguous().float()            # [N,2]


def midi_to_notes(path, n_frames, fps=FPS, held=3, rest=-1):
    """MIDI -> [T,128] per-frame note vector. held pitches=`held` (3=active), others=`rest`
    (-1=masked, jam-style). Requires pretty_midi (optional)."""
    try:
        import pretty_midi
    except ImportError as e:
        raise ImportError("MIDI mode needs pretty_midi: pip install pretty_midi") from e
    pm = pretty_midi.PrettyMIDI(path)
    notes = np.full((n_frames, NUM_NOTES), rest, dtype=np.int64)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for nt in inst.notes:
            f0, f1 = int(nt.start * fps), min(n_frames, int(nt.end * fps) + 1)
            if 0 <= nt.pitch < NUM_NOTES and f0 < n_frames:
                notes[max(0, f0):f1, nt.pitch] = held
    return notes


def _discretize_cfg(value, step, max_bin):
    clamped = max(-1.0, min(7.0, value))
    return max(0, min(max_bin, int(round((clamped + 1.0) / step))))


def default_cfgs(cfg_musiccoca=1.6, cfg_notes=2.4, cfg_drums=4.0):
    return [_discretize_cfg(cfg_musiccoca, 0.2, 40),
            _discretize_cfg(cfg_notes, 0.2, 40),
            _discretize_cfg(cfg_drums, 1.0, 8)]


# ---------------------------------------------------------------- featurization
@torch.no_grad()
def audio_to_target(enc, model, wav, device):
    """wav [N,2] -> target global ids [1,T,Q]."""
    Q = model.config.num_codebooks
    w = wav.unsqueeze(0).to(device).float()              # [1,N,2]
    feats = enc.forward_features(w)                      # [1,T,256]
    codes = rvq_encode(feats, model.quant.float(), Q)    # [1,T,Q] in 0..codebook_size-1
    per_cb = (torch.arange(Q, device=device) * model.config.codebook_size
              + NUM_RESERVED_TOKENS).view(1, 1, Q)
    return codes.long() + per_cb                         # [1,T,Q]


def build_cond(model, style_tokens, notes_seq, T, cfgs, device):
    """Raw conditioning tokens [1,Tc,144]; Depthformer.forward(cond, target) encodes them.
    style_tokens: list[12] (-1 masked). notes_seq: np[T,128] per-frame (MIDI) or None (constant)."""
    offset = NUM_RESERVED_TOKENS + 1
    drums = [-1]
    if notes_seq is not None:                            # per-frame conditioning (MIDI)
        ns = notes_seq.tolist()
        rows = [list(style_tokens) + ns[t] + drums + list(cfgs) for t in range(T)]
        arr = np.asarray(rows, dtype=np.int64) + offset  # [T,144]
        return torch.from_numpy(arr).view(1, T, -1).to(device)
    vals = list(style_tokens) + [-1] * NUM_NOTES + drums + list(cfgs)   # constant (caption/raw)
    arr = np.asarray(vals, dtype=np.int64) + offset
    return torch.from_numpy(arr).view(1, 1, -1).to(device)


def style_tokens_for(model, caption):
    if not caption:
        return [-1] * NUM_MUSICCOCA
    return list(model._tokenize_style(caption))


# ---------------------------------------------------------------- data
class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, clip_seconds=10.0):
        self.rows = [json.loads(l) for l in open(manifest) if l.strip()]
        self.win = int(clip_seconds * SR)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav = load_audio(r["audio"])                     # [N,2]
        if self.win and wav.shape[0] > self.win:         # random crop
            s = random.randint(0, wav.shape[0] - self.win)
            wav, crop = wav[s:s + self.win], (s / SR, (s + self.win) / SR)
        else:
            crop = (0.0, wav.shape[0] / SR)
        return wav, r.get("caption"), r.get("midi"), crop


# ---------------------------------------------------------------- step
def compute_loss(model, enc, wav, caption, midi_path, crop, cfgs, device):
    target = audio_to_target(enc, model, wav, device)    # [1,T,Q]
    T = target.shape[1]
    style = style_tokens_for(model, caption)
    notes_seq = None
    if midi_path:
        full = midi_to_notes(midi_path, int(crop[1] * FPS) + T)
        f0 = int(crop[0] * FPS)
        notes_seq = full[f0:f0 + T]
        if notes_seq.shape[0] < T:                       # pad tail masked
            notes_seq = np.pad(notes_seq, ((0, T - notes_seq.shape[0]), (0, 0)), constant_values=-1)
    cond = build_cond(model, style, notes_seq, T, cfgs, device)
    logits = _dformer(model)(cond, target)               # Depthformer.forward(cond, target) -> [1,T,Q,vocab]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1).long())


# ---------------------------------------------------------------- train
def build_peft_model(model, rank=16, alpha=32, dropout=0.05):
    """Swap JaxLinear->Linear then wrap with peft LoRA. Returns the PeftModel (base modified
    in-place, so `_dformer(model)` carries the adapters)."""
    from peft import LoraConfig, get_peft_model
    conv = swap_jaxlinear_to_linear(_dformer(model))
    print(f"swapped {len(conv)} JaxLinear->Linear: {sorted(set(conv))}", flush=True)
    peft_model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=list(SWAP_SUFFIXES), task_type=None))
    peft_model.print_trainable_parameters()
    return peft_model


def train(args):
    from transformers import AutoModel
    device = args.device
    path = args.model_path or MODEL_REPOS[args.model]
    print(f"loading {path} ...", flush=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True,
                                      dtype=torch.bfloat16).to(device).eval()
    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if any(r.get("caption") for r in rows):
        model.load_processor()
    enc = load_spectrostream_encoder(args.encoder_path, dtype=torch.float32).to(device)

    peft_model = build_peft_model(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    lp = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lp, lr=args.lr, weight_decay=args.weight_decay)
    ds = ManifestDataset(args.manifest, args.clip_seconds)
    cfgs = default_cfgs(args.cfg_musiccoca, args.cfg_notes, args.cfg_drums)
    os.makedirs(args.out, exist_ok=True)

    step = 0
    opt.zero_grad()
    while step < args.steps:
        for wav, caption, midi_path, crop in torch.utils.data.DataLoader(
                ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0]):
            loss = compute_loss(model, enc, wav, caption, midi_path, crop, cfgs, device)
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lp, args.grad_clip)
                opt.step()
                opt.zero_grad()
            if step % args.log_every == 0:
                print(f"step {step:6d}  loss {loss.item():.4f}", flush=True)
            step += 1
            if step >= args.steps:
                break
        if args.save_every and step % args.save_every < 1:
            peft_model.save_pretrained(os.path.join(args.out, f"step_{step}"))
    peft_model.save_pretrained(args.out)
    print(f"done -> {args.out}/adapter_model.safetensors", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", choices=["small", "base"], default="small")
    p.add_argument("--model-path", default=None, help="local dir / HF repo (overrides --model)")
    p.add_argument("--encoder-path", required=True, help="SpectroStream encoder.safetensors")
    p.add_argument("--out", default="lora_out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--clip-seconds", type=float, default=10.0)
    p.add_argument("--cfg-musiccoca", type=float, default=1.6)
    p.add_argument("--cfg-notes", type=float, default=2.4)
    p.add_argument("--cfg-drums", type=float, default=4.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=0)
    train(p.parse_args())


if __name__ == "__main__":
    main()
