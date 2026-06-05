# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AOTI export specs for the functional per-frame step.

The temporal and depth steps are export-clean (no inline constant tensors,
mask-free incremental KV) and exported with dynamic KV-length dims, so a single
compiled graph serves every frame. Artifacts are GPU-arch specific — compile on
the same hardware (e.g. the ZeroGPU Blackwell) you run on.
"""
import torch
import torch.nn as nn
from torch.export import Dim


class TemporalStepModule(nn.Module):
    """forward(prev_frame, self_kv, cross_kv, source) -> (out, new_self, new_cross)."""

    def __init__(self, decoder):
        super().__init__()
        self.d = decoder

    def forward(self, prev_frame, self_kv, cross_kv, source):
        return self.d.temporal_step_fn(prev_frame, self_kv, cross_kv, source)


class DepthStepModule(nn.Module):
    """forward(depth_input, depth_kv) -> (logits, new_kv)."""

    def __init__(self, decoder):
        super().__init__()
        self.d = decoder

    def forward(self, depth_input, depth_kv):
        return self.d.depth_step_fn(depth_input, depth_kv)


def _params_ref(decoder):
    p = next(decoder.parameters())
    return p.device, p.dtype


def temporal_export_inputs(decoder, kv_len=20):
    """(args, dynamic_shapes) for exporting the temporal step."""
    dev, dt = _params_ref(decoder)
    c = decoder.cfg
    L, nh, uph = c.temporal.num_layers, c.temporal.num_heads, c.temporal.dim_per_head
    prev = torch.randint(c.num_reserved_tokens, c.num_reserved_tokens + c.codebook_size,
                         (1, 1, c.num_codebooks), device=dev)
    mk = lambda: [(torch.randn(1, kv_len, nh, uph, device=dev, dtype=dt),
                   torch.randn(1, kv_len, nh, uph, device=dev, dtype=dt)) for _ in range(L)]
    self_kv, cross_kv = mk(), mk()
    source = torch.randn(1, 1, c.encoder_model_dims, device=dev, dtype=dt)
    T = Dim("T", min=0, max=c.temporal_max_past + 1)
    ds = (None, [({1: T}, {1: T}) for _ in range(L)], [({1: T}, {1: T}) for _ in range(L)], None)
    return (prev, self_kv, cross_kv, source), ds


def depth_export_inputs(decoder, kv_len=6):
    """(args, dynamic_shapes) for exporting the depth step."""
    dev, dt = _params_ref(decoder)
    c = decoder.cfg
    L, nh, uph = c.depth.num_layers, c.depth.num_heads, c.depth.dim_per_head
    depth_input = torch.randn(1, 1, c.temporal.model_dims, device=dev, dtype=dt)
    depth_kv = [(torch.randn(1, kv_len, nh, uph, device=dev, dtype=dt),
                 torch.randn(1, kv_len, nh, uph, device=dev, dtype=dt)) for _ in range(L)]
    Td = Dim("Td", min=0, max=c.num_codebooks)
    ds = (None, [({1: Td}, {1: Td}) for _ in range(L)])
    return (depth_input, depth_kv), ds


def export_temporal(decoder):
    args, ds = temporal_export_inputs(decoder)
    return torch.export.export(TemporalStepModule(decoder), args, dynamic_shapes=ds)


def export_depth(decoder):
    args, ds = depth_export_inputs(decoder)
    return torch.export.export(DepthStepModule(decoder), args, dynamic_shapes=ds)


def temporal_state_dict(decoder):
    return TemporalStepModule(decoder).state_dict()


def depth_state_dict(decoder):
    return DepthStepModule(decoder).state_dict()


# --- ZeroGPU-native compile / save / load (uses the `spaces` library) --------
# Artifacts are GPU-arch specific: compile and run on the same hardware.

def compile_step_archives(decoder, inductor_configs=None):
    """Export + AOTI-compile the temporal and depth steps with spaces.aoti_compile.
    Returns {'temporal': bytes, 'depth': bytes} — weight-less graph .pt2 blobs.
    Must run on a GPU (inside @spaces.GPU on ZeroGPU)."""
    import importlib
    spaces = importlib.import_module("spaces")  # optional (ZeroGPU); kept off the import graph
    out = {}
    for name, ep in (("temporal", export_temporal(decoder)),
                     ("depth", export_depth(decoder))):
        cm = spaces.aoti_compile(ep, inductor_configs)
        out[name] = bytes(cm.archive_file.getbuffer())
    return out


def load_compiled_steps(decoder, repo_id=None, local_dir=None, filenames=("temporal.pt2", "depth.pt2")):
    """Load weight-less step graphs and bind them to `decoder`'s own weights.
    Returns (temporal_callable, depth_callable) for step_f(temporal_step=, depth_step=)."""
    import importlib, os
    LazyAOTIModel = importlib.import_module("spaces.zero.torch.aoti").LazyAOTIModel
    paths = {}
    keys = ("temporal", "depth")
    if local_dir is not None:
        for k, fn in zip(keys, filenames):
            paths[k] = os.path.join(local_dir, fn)
    else:
        from huggingface_hub import hf_hub_download
        for k, fn in zip(keys, filenames):
            paths[k] = hf_hub_download(repo_id, fn)
    t = LazyAOTIModel(paths["temporal"]).with_weights(temporal_state_dict(decoder))
    d = LazyAOTIModel(paths["depth"]).with_weights(depth_state_dict(decoder))
    return t, d
