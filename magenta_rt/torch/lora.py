"""Custom LoRA for the MRT2 torch Depthformer, via `torch.nn.utils.parametrize`.

peft-shaped API (no peft dependency):
  inject_lora(dformer)                      -> register LoRA, freeze base, return trainable params
  save_pretrained(dformer, dir)             -> adapter_model.safetensors + adapter_config.json
  from_pretrained(base_dformer, dir)        -> re-inject + load A/B (mirrors PeftModel.from_pretrained)
  merge_and_unload(dformer)                 -> bake delta into the original param, drop the wrapper

Works on BOTH 2D JaxLinear kernels [in,out] and 3D attention projection kernels [in,nh,uph]
(delta reshaped to the kernel's shape). Because we parametrize the *parameter* (not swap the
module), `ffn_layer1`'s fused gelu is covered too. After merge_and_unload the module graph is
byte-identical to the un-LoRA'd model, so inference perf, the AOTI step graphs, and the HF
model are all untouched.

Load a trained adapter:
  base = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16)
  from magenta_rt.torch.lora import from_pretrained, merge_and_unload
  from_pretrained(base.depthformer, "lora_out")   # base.model for the demo class
  merge_and_unload(base.depthformer)               # optional: bake in for deployment
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

# 3D attention projection params (direct on AttnProjection / SelfAttention) + 2D JaxLinear modules
ATTN_PARAMS = ("query_projection_kernel", "key_projection_kernel",
               "value_projection_kernel", "output_projection_kernel")
JAXLIN_MODULES = ("ffn_layer1", "ffn_layer2", "depth_input_adapter", "to_logits")


class LoRAParametrization(nn.Module):
    """W -> W + (alpha/r)*(A @ B).reshape(W.shape). A kaiming, B=0 so the delta starts at 0."""

    def __init__(self, shape, r=16, alpha=32, device=None, dtype=torch.float32):
        super().__init__()
        self.shape = tuple(shape)
        self.r, self.alpha = int(r), float(alpha)
        self.scale = self.alpha / self.r
        fan_in, fan_out = self.shape[0], math.prod(self.shape[1:]) if len(self.shape) > 1 else 1
        self.A = nn.Parameter(torch.empty(fan_in, self.r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(self.r, fan_out, device=device, dtype=dtype))

    def forward(self, W):
        delta = (self.scale * (self.A @ self.B)).reshape(self.shape)
        return W + delta.to(W.dtype)


def _iter_targets(root):
    """Yield (module, param_name) for every kernel we LoRA-ify."""
    for qn, mod in root.named_modules():
        for pn in ATTN_PARAMS:
            if mod._parameters.get(pn) is not None:
                yield mod, pn
        if qn.split(".")[-1] in JAXLIN_MODULES and mod._parameters.get("kernel") is not None:
            yield mod, "kernel"


def inject_lora(root, r=16, alpha=32):
    """Register LoRA on attention + MLP + adapter + output kernels; freeze everything else.
    Returns the trainable LoRA params (A/B). `root` is the Depthformer submodule."""
    lora_params = []
    for mod, pn in list(_iter_targets(root)):          # materialize first (registration mutates _parameters)
        w = getattr(mod, pn)
        lp = LoRAParametrization(w.shape, r, alpha, w.device, torch.float32)
        P.register_parametrization(mod, pn, lp)
        lora_params += [lp.A, lp.B]
    ids = {id(p) for p in lora_params}
    for p in root.parameters():
        p.requires_grad_(id(p) in ids)                 # freeze base + the stashed originals
    return lora_params


def _lora_items(root):
    for qn, mod in root.named_modules():
        if not P.is_parametrized(mod):
            continue
        for pn in list(mod.parametrizations.keys()):
            lp = mod.parametrizations[pn][0]
            if isinstance(lp, LoRAParametrization):
                yield (f"{qn}.{pn}" if qn else pn), mod, pn, lp


def save_pretrained(root, save_dir, base_model=None):
    """Write the A/B tensors + config (peft-shaped layout)."""
    from safetensors.torch import save_file
    os.makedirs(save_dir, exist_ok=True)
    tensors, targets, r, alpha = {}, [], None, None
    for key, _, _, lp in _lora_items(root):
        tensors[key + ".A"], tensors[key + ".B"] = lp.A.detach().cpu(), lp.B.detach().cpu()
        targets.append(key); r, alpha = lp.r, lp.alpha
    save_file(tensors, os.path.join(save_dir, "adapter_model.safetensors"))
    json.dump({"format": "magenta-rt-lora-v1", "r": r, "alpha": alpha,
               "targets": sorted(targets), "base_model": base_model},
              open(os.path.join(save_dir, "adapter_config.json"), "w"), indent=2)
    return save_dir


def from_pretrained(base_root, save_dir):
    """Re-inject LoRA on `base_root` (the Depthformer) and load the saved A/B. Mirrors peft."""
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(save_dir, "adapter_config.json")))
    inject_lora(base_root, r=cfg["r"], alpha=cfg["alpha"])
    tens = load_file(os.path.join(save_dir, "adapter_model.safetensors"))
    for key, _, _, lp in _lora_items(base_root):
        if key + ".A" in tens:
            lp.A.data.copy_(tens[key + ".A"].to(lp.A.device, lp.A.dtype))
            lp.B.data.copy_(tens[key + ".B"].to(lp.B.device, lp.B.dtype))
    return base_root


def merge_and_unload(root):
    """Bake each delta into its original param and remove the parametrization wrapper.
    Resulting module graph == the un-LoRA'd base (same shapes/names → same perf, AOTI/HF intact)."""
    mods = [(mod, list(mod.parametrizations.keys()))
            for _, mod in root.named_modules() if P.is_parametrized(mod)]
    for mod, names in mods:
        for pn in names:
            P.remove_parametrizations(mod, pn, leave_parametrized=True)
    return root
