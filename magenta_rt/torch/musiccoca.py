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

"""Pure-PyTorch MusicCoCa style encoder (text path).

The upstream MusicCoCa ships only as TFLite. The text tower and RVQ quantizer
were converted TFLite -> ONNX -> torch and traced to TorchScript (token-exact
vs the TFLite reference). Runtime deps: torch + sentencepiece only (no
ai_edge_litert / tflite). The audio-prompt tower is not yet ported.
"""
import os

import numpy as np
import torch

MAX_TEXT_LEN = 128
TARGET_SOS_ID = 1
EMBEDDING_DIM = 768
RVQ_DEPTH = 12
DEFAULT_REPO = "magenta-torch/magenta-rt-musiccoca-torch"
AUDIO_SR = 16000
CLIP_SAMPLES = 160000  # 10s @ 16kHz
_MEL_FL, _MEL_HOP, _MEL_NFFT, _PREEMPH = 400, 160, 2048, 0.97


class MusicCoCa:
    """Text -> 768-d style embedding -> 12 RVQ style tokens, all in torch."""

    def __init__(self, repo_id=DEFAULT_REPO, resource_dir=None, device="cpu"):
        import sentencepiece
        if resource_dir is not None:
            te = os.path.join(resource_dir, "text_encoder.pt")
            q = os.path.join(resource_dir, "quantizer.pt")
            spm = os.path.join(resource_dir, "spm.model")
        else:
            from huggingface_hub import hf_hub_download
            te = hf_hub_download(repo_id, "text_encoder.pt")
            q = hf_hub_download(repo_id, "quantizer.pt")
            spm = hf_hub_download(repo_id, "spm.model")
        self.device = device
        self._te = torch.jit.load(te, map_location=device).eval()
        self._q = torch.jit.load(q, map_location=device).eval()
        self._sp = sentencepiece.SentencePieceProcessor()
        self._sp.Load(spm)
        self.embedding_dim = EMBEDDING_DIM
        self.rvq_depth = RVQ_DEPTH
        # Audio tower (mel preprocessor + music_encoder ViT). Lazy.
        self._me = None
        self._mel = None
        self._hann = None
        self._resource_dir = resource_dir
        self._repo_id = repo_id
        self.pca_basis = None  # [K,768] PCA axes for style steering

    def _ensure_audio(self):
        if self._me is not None:
            return
        if self._resource_dir is not None:
            mep = os.path.join(self._resource_dir, "music_encoder.pt")
            melp = os.path.join(self._resource_dir, "mel_params.npz")
        else:
            from huggingface_hub import hf_hub_download
            mep = hf_hub_download(self._repo_id, "music_encoder.pt")
            melp = hf_hub_download(self._repo_id, "mel_params.npz")
        self._me = torch.jit.load(mep, map_location=self.device).eval()
        d = np.load(melp)
        self._mel = torch.from_numpy(d["mel"]).float().to(self.device)
        self._hann = torch.from_numpy(d["hann"]).float().to(self.device)

    def to(self, device):
        self.device = device
        self._te = self._te.to(device)
        self._q = self._q.to(device)
        if self._me is not None:
            self._me = self._me.to(device)
            self._mel = self._mel.to(device)
            self._hann = self._hann.to(device)
        return self

    def _log_mel(self, wav):
        """wav [S] (16kHz mono float) -> log-mel [992,128] (bit-exact vs TFLite)."""
        x = wav.to(self.device).float()
        y = x.clone()
        y[1:] = x[1:] - _PREEMPH * x[:-1]
        xp = torch.nn.functional.pad(y, (0, _MEL_FL))
        nf = (xp.shape[0] - _MEL_FL) // _MEL_HOP + 1
        idx = (torch.arange(_MEL_FL, device=self.device)[None, :]
               + torch.arange(nf, device=self.device)[:, None] * _MEL_HOP)
        fr = xp[idx] * self._hann
        power = torch.fft.rfft(fr, n=_MEL_NFFT, dim=1).abs() ** 2
        return torch.log(power[:, 1:1025] @ self._mel + 0.001)[:992]

    @torch.no_grad()
    def embed_audio(self, samples, sample_rate):
        """samples [S] or [S,C] float; -> [768] style embedding (clips mean-pooled)."""
        self._ensure_audio()
        x = np.asarray(samples, np.float32)
        if x.ndim == 2:
            x = x.mean(1)
        if sample_rate != AUDIO_SR:
            import importlib
            resampy = importlib.import_module("resampy")  # optional; off the import graph
            x = resampy.resample(x, sample_rate, AUDIO_SR)
        # split into 10s clips (pad last), embed each, mean-pool
        embs = []
        for s in range(0, max(len(x), 1), CLIP_SAMPLES):
            clip = x[s:s + CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES:
                clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            mel = self._log_mel(torch.from_numpy(clip))
            embs.append(self._me(mel[None]).reshape(-1))
        return torch.stack(embs).mean(0)

    def _tokenize_text(self, text):
        labels = self._sp.EncodeAsIds(text.lower())[: MAX_TEXT_LEN - 1]
        ids = [TARGET_SOS_ID] + labels
        n = len(ids)
        ids = ids + [0] * (MAX_TEXT_LEN - len(ids))
        ids_t = torch.tensor([ids], dtype=torch.int32, device=self.device)
        pad = torch.ones(1, MAX_TEXT_LEN, device=self.device)
        pad[0, :n] = 0.0
        return ids_t, pad

    @torch.no_grad()
    def embed(self, text_or_audio, *args, **kwargs):
        """Text string OR audio (Waveform / (samples, sr) / np array @ 16kHz)
        -> [768] style embedding (torch tensor on self.device)."""
        if isinstance(text_or_audio, str):
            ids, pad = self._tokenize_text(text_or_audio)
            return self._te(ids, pad).reshape(-1)
        # audio: accept a Waveform-like (has .samples/.sample_rate), (samples,sr), or np
        obj = text_or_audio
        if hasattr(obj, "samples") and hasattr(obj, "sample_rate"):
            return self.embed_audio(obj.samples, obj.sample_rate)
        if isinstance(obj, tuple) and len(obj) == 2:
            return self.embed_audio(obj[0], obj[1])
        return self.embed_audio(obj, AUDIO_SR)

    @torch.no_grad()
    def tokenize(self, embedding, pca_coeffs=None):
        """[768] embedding -> [12] int RVQ tokens (np.int64). Accepts np or torch.
        For layering, pass a (weighted) mean of several embeddings.
        pca_coeffs: optional [K] shift along self.pca_basis before quantizing."""
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)
        embedding = embedding.to(self.device).reshape(1, EMBEDDING_DIM).float()
        if pca_coeffs is not None and self.pca_basis is not None:
            c = torch.as_tensor(pca_coeffs, dtype=torch.float32, device=self.device).reshape(-1)
            k = min(c.numel(), self.pca_basis.shape[0])
            embedding = embedding + (c[:k] @ self.pca_basis[:k]).reshape(1, EMBEDDING_DIM)
        return self._q(embedding).reshape(-1).cpu().numpy().astype(np.int64)

    def set_pca(self, components):
        """components: [K,768] PCA basis. Steer style via tokenize(emb, pca_coeffs=[...])."""
        self.pca_basis = torch.as_tensor(components, dtype=torch.float32,
                                         device=self.device).reshape(-1, EMBEDDING_DIM)
        return self.pca_basis

    @torch.no_grad()
    def compute_pca(self, texts, k=8):
        """Fit a PCA basis from a corpus of style prompts; sets + returns self.pca_basis [k,768]."""
        embs = torch.stack([self.embed(t).float() for t in texts])  # [N,768]
        q = min(k, embs.shape[0] - 1, EMBEDDING_DIM)
        _, _, V = torch.pca_lowrank(embs - embs.mean(0, keepdim=True), q=q)
        return self.set_pca(V.t().contiguous())  # [q,768]

    def embed_tokens(self, text):
        """Convenience: text -> 12 style tokens (list[int])."""
        return self.tokenize(self.embed(text)).tolist()
