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

"""HF config for the Magenta RealTime 2 PyTorch model."""

from transformers import PretrainedConfig


class MagentaRT2Config(PretrainedConfig):
    """Config for `MagentaRT2ForConditionalGeneration`.

    `temporal` / `depth` are [num_layers, model_dims, hidden_dims, num_heads,
    dim_per_head] for the two Depthformer transformer stacks.
    """

    model_type = "magenta_rt2"

    def __init__(
        self,
        size="mrt2_small",
        encoder_model_dims=256,
        temporal=(12, 1024, 4096, 8, 128),
        depth=(2, 768, 3072, 6, 128),
        temporal_max_past=41,
        depth_max_past=12,
        musiccoca_rvq=12,
        musiccoca_per_rvq_vocab=1031,
        musiccoca_embed_dim=768,
        regular_num_embeddings_per_channel=None,
        regular_num_channels=132,
        num_sinks=1,
        num_codebooks=12,
        codebook_size=1024,
        num_reserved_tokens=6,
        vocab_size=12294,
        soft_cap_logits=30.0,
        temperature=1.3,
        top_k=40,
        cfg_musiccoca=3.0,
        cfg_notes=1.0,
        cfg_drums=1.0,
        num_notes=128,
        num_drums=1,
        sample_rate=48000,
        frame_samples=1920,
        codec_param_shapes=None,
        **kwargs,
    ):
        self.size = size
        self.codec_param_shapes = codec_param_shapes
        self.encoder_model_dims = encoder_model_dims
        self.temporal = list(temporal)
        self.depth = list(depth)
        self.temporal_max_past = temporal_max_past
        self.depth_max_past = depth_max_past
        self.musiccoca_rvq = musiccoca_rvq
        self.musiccoca_per_rvq_vocab = musiccoca_per_rvq_vocab
        self.musiccoca_embed_dim = musiccoca_embed_dim
        self.regular_num_embeddings_per_channel = regular_num_embeddings_per_channel
        self.regular_num_channels = regular_num_channels
        self.num_sinks = num_sinks
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.num_reserved_tokens = num_reserved_tokens
        self.vocab_size = vocab_size
        self.soft_cap_logits = soft_cap_logits
        self.temperature = temperature
        self.top_k = top_k
        self.cfg_musiccoca = cfg_musiccoca
        self.cfg_notes = cfg_notes
        self.cfg_drums = cfg_drums
        self.num_notes = num_notes
        self.num_drums = num_drums
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        super().__init__(**kwargs)

    @classmethod
    def from_size(cls, size):
        from .depthformer import config_for
        from dataclasses import astuple
        c = config_for(size)
        return cls(
            size=size,
            encoder_model_dims=c.encoder_model_dims,
            temporal=list(astuple(c.temporal)),
            depth=list(astuple(c.depth)),
            temporal_max_past=c.temporal_max_past,
            depth_max_past=c.depth_max_past,
            musiccoca_rvq=c.musiccoca_rvq,
            musiccoca_per_rvq_vocab=c.musiccoca_per_rvq_vocab,
            musiccoca_embed_dim=c.musiccoca_embed_dim,
            regular_num_embeddings_per_channel=list(c.regular_num_embeddings_per_channel),
            regular_num_channels=c.regular_num_channels,
            num_sinks=c.num_sinks,
            num_codebooks=c.num_codebooks,
            codebook_size=c.codebook_size,
            num_reserved_tokens=c.num_reserved_tokens,
            vocab_size=c.vocab_size,
            soft_cap_logits=c.soft_cap_logits,
        )


__all__ = ["MagentaRT2Config"]
