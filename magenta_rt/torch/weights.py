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

"""Map JAX/Linen safetensors checkpoint keys onto the torch Depthformer."""
import re

import numpy as np
import torch
from safetensors import safe_open


def _jax_key_to_torch(k: str):
    """Translate a 'params/depthformer/...' key to a torch named_parameter path,
    or return None to skip (e.g. soundstream)."""
    if not k.startswith("params/depthformer/"):
        return None
    s = k[len("params/depthformer/"):]

    # Encoder (conditioning embedders).
    if s.startswith("encoder/"):
        if s == "encoder/body/encoder_ln/scale":
            return "encoder.encoder_ln.scale"
        if s == "encoder/body/encoder_ln/bias":
            return "encoder.encoder_ln.bias"
        if s.endswith("mulan_dequantizer/embedding"):
            return "encoder.mulan_dequantizer"
        if s.endswith("mulan_embedder/depth_input_adapter/kernel"):
            return "encoder.mulan_adapter.kernel"
        if s.endswith("regular_embedder/embedding"):
            return "encoder.regular_embedding"
        return None

    # Decoder.
    if s == "decoder/decoder_embedding/embedding/embedding":
        return "decoder.embedding"
    if s == "decoder/depth_body/depth_input_adapter/kernel":
        return "decoder.depth_input_adapter.kernel"
    if s.startswith("decoder/depth_body/final_ln/"):
        return "decoder.final_ln." + s.split("/")[-1]
    if s.startswith("decoder/depth_body/to_logits/"):
        return "decoder.to_logits." + s.split("/")[-1]

    m = re.match(r"decoder/(temporal_body|depth_body)/transformer/x_layers_(\d+)/(.*)", s)
    if m:
        body, i, rest = m.group(1), int(m.group(2)), m.group(3)
        prefix = f"decoder.{body}.layers.{i}."
        return prefix + _layer_subkey(rest)

    return None


def _layer_subkey(rest: str):
    # rest like 'self_attention/attention/query_projection/kernel'
    parts = rest.split("/")
    sub = parts[0]  # self_attention | cross_attention | ffn
    tail = parts[1:]
    if sub in ("self_attention", "cross_attention"):
        if tail[0] == "attention":
            name = tail[1]
            if name.endswith("_projection"):  # query/key/value_projection/kernel
                return f"{sub}.attention.{name}_kernel"
            # per_dim_scale, sink_key_embeddings, sink_value_embeddings
            return f"{sub}.attention.{name}"
        if tail[0] == "output_projection":  # output_projection/kernel
            return f"{sub}.output_projection_kernel"
        if tail[0] in ("pre_norm", "post_norm"):
            return f"{sub}.{tail[0]}.scale"
    if sub == "ffn":
        if tail[0] in ("ffn_layer1", "ffn_layer2"):
            return f"ffn.{tail[0]}.{tail[1]}"  # kernel | bias
        if tail[0] in ("pre_norm", "post_norm"):
            return f"ffn.{tail[0]}.scale"
    raise KeyError(f"unhandled layer subkey: {rest}")


def load_depthformer(model, checkpoint_path, dtype=torch.float32, strict=True, verbose=False):
    """Load checkpoint into a torch Depthformer (module with .encoder/.decoder)."""
    params = dict(model.named_parameters())
    seen = set()
    with safe_open(str(checkpoint_path), "numpy") as f:
        for k in f.keys():
            tname = _jax_key_to_torch(k)
            if tname is None:
                continue
            if tname not in params:
                raise KeyError(f"{k} -> {tname} not found in model")
            arr = f.get_tensor(k)
            t = torch.from_numpy(np.asarray(arr)).to(dtype)
            p = params[tname]
            if tuple(p.shape) != tuple(t.shape):
                raise ValueError(f"shape mismatch {tname}: model {tuple(p.shape)} ckpt {tuple(t.shape)} ({k})")
            with torch.no_grad():
                p.copy_(t)
            seen.add(tname)
    missing = [n for n in params if n not in seen]
    if verbose:
        print(f"loaded {len(seen)} params, {len(missing)} missing")
    if strict and missing:
        raise RuntimeError(f"missing params not loaded: {missing[:20]} ... ({len(missing)} total)")
    return model
