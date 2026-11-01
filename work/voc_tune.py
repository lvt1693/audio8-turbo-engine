"""The streaming vocoder's step time at B rows, heuristic conv tiles vs tuned (StreamingVocoder.tune, no cache), and
the tuned output vs the untuned (must be the same up to fp16 rounding). GPU 0 of .30.
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 venv/bin/python work/voc_tune.py [B]"""
import os
import sys

ROOT = "/home/ubuntu/truc_ai/Audio8"
sys.path.insert(0, ROOT + "/serve")

import torch

from fast_engine import StreamingVocoder
from fast_kernels import GemmDispatch
from tts_engine import load_model

B = int(sys.argv[1]) if len(sys.argv) > 1 else 192
model, _ = load_model(ROOT + "/models/Audio8-TTS-Realistic-0.6b")
with torch.inference_mode():
    codec = model.load_codec(dtype=torch.float32)
    v = StreamingVocoder(codec, B, torch.device("cuda"))
    codes = torch.randint(0, 1024, (B, 10), device="cuda")
    v.clear_many(list(range(B)))
    t_def = GemmDispatch._time(lambda: v.step(codes, B))
    v.clear_many(list(range(B)))
    w0 = [v.step(codes, B).clone() for _ in range(3)]
    v.tune(B, log=print, cache_path=None)
    for c, T, _ in v.convs():
        print(f"  conv {c.Cin:4d}->{c.Cout:4d} k{c.taps} d{c.dil} ph{c.ph} T{T:5d}: {c.cfg}")
    v.clear_many(list(range(B)))
    t_tun = GemmDispatch._time(lambda: v.step(codes, B))
    v.clear_many(list(range(B)))
    w1 = [v.step(codes, B).clone() for _ in range(3)]
    d = max(float((a - b).abs().max()) for a, b in zip(w0, w1))
    print(f"vocoder step at B={B}: heuristic {t_def:.2f} ms -> tuned {t_tun:.2f} ms; output max|d| {d:.5f}")
print("VT_DONE")
