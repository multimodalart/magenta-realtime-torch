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

"""MusicCoCa style processor for Magenta RealTime 2.

A processor-style component (like a feature extractor / tokenizer): turns a text
prompt OR an audio clip into 12 RVQ style tokens that condition the model. Pure
torch + sentencepiece (text tower, audio tower, RVQ all torch-native).
"""

import os

import numpy as np


class MusicCoCaProcessor:
    """Text/audio -> 12 RVQ style tokens (and 768-d embeddings, for layering)."""

    def __init__(self, musiccoca):
        self._mc = musiccoca

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, device="cpu", **kwargs):
        from .musiccoca import MusicCoCa
        p = pretrained_model_name_or_path
        if p is not None and os.path.isdir(p) and os.path.exists(os.path.join(p, "text_encoder.pt")):
            mc = MusicCoCa(resource_dir=p, device=device)
        else:
            mc = MusicCoCa(repo_id=p, device=device) if p else MusicCoCa(device=device)
        return cls(mc)

    def save_pretrained(self, save_directory, **kwargs):
        # Artifacts live in the MusicCoCa hub repo; nothing extra to serialize here.
        os.makedirs(save_directory, exist_ok=True)

    @property
    def device(self):
        return self._mc.device

    def to(self, device):
        self._mc.to(device)
        return self

    def embed(self, text_or_audio):
        """Text str / audio (Waveform | (samples, sr) | np@16kHz) -> [768] torch."""
        return self._mc.embed(text_or_audio)

    def tokenize(self, embedding):
        """[768] embedding -> [12] int RVQ tokens (np.int64)."""
        return self._mc.tokenize(embedding)

    def layer(self, prompts, weights=None):
        """Blend several prompts (text/audio) by weighted-mean of embeddings,
        then tokenize. `prompts` is a list; `weights` defaults to uniform."""
        embs = [self.embed(p) for p in prompts]
        w = weights or [1.0 / len(embs)] * len(embs)
        emb = sum(wi * e for wi, e in zip(w, embs))
        return self.tokenize(emb).tolist()

    def __call__(self, text_or_audio, return_tokens=True):
        """-> 12 style tokens (list[int]) by default, or the [768] embedding."""
        emb = self.embed(text_or_audio)
        return self.tokenize(emb).tolist() if return_tokens else emb


__all__ = ["MusicCoCaProcessor"]
