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

"""Minimal real-time audio streaming with Magenta RealTime 2 (PyTorch).

Generates continuously and plays it live (or writes a wav as it goes). Each
`generate` call returns the newly-available audio; pass `state` back to continue
seamlessly, and change `style` between calls to steer the stream.

    # live playback (needs `sounddevice`):
    python examples/streaming.py "disco funk" --seconds 20

    # write a wav instead (needs `soundfile`):
    python examples/streaming.py "ambient pads, ethereal" --out out.wav
"""

import argparse

import numpy as np
import torch
from transformers import AutoModel

SR = 48000
FPS = 25  # token frames per second of audio (25 frames -> 1.0 s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="lo-fi hip hop, mellow")
    ap.add_argument("--model", default="magenta-torch/magenta-realtime-2-small")
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--out", default=None, help="write a wav instead of live playback")
    ap.add_argument("--compile", action="store_true", help="torch.compile the step (faster, ~warmup)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).to(dev).eval()
    model.load_processor(device=dev)
    if args.compile and dev == "cuda":
        model.compile_steps()

    def gen():
        state = None
        for i in range(args.seconds):
            chunk, state = model.generate(
                style=args.prompt, frames=FPS, temperature=args.temperature,
                state=state, flush=(i == args.seconds - 1))
            print(f"\r{i + 1}/{args.seconds}s", end="", flush=True)
            yield chunk  # float32 [N, 2] in [-1, 1]

    if args.out:
        import soundfile as sf
        with sf.SoundFile(args.out, "w", SR, channels=2, subtype="PCM_16") as f:
            for chunk in gen():
                f.write(chunk)
        print(f"\nwrote {args.out}")
    else:
        import sounddevice as sd
        with sd.OutputStream(samplerate=SR, channels=2, dtype="float32") as out:
            for chunk in gen():
                out.write(np.ascontiguousarray(chunk, dtype=np.float32))
        print()


if __name__ == "__main__":
    main()
