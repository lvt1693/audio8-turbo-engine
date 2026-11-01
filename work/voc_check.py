"""Streaming vocoder row independence + snapshot/restore (GPU 0 of .30).
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 venv/bin/python work/voc_check.py"""
import os
import sys

ROOT = "/home/ubuntu/truc_ai/Audio8"
sys.path.insert(0, ROOT + "/serve")

import torch

from fast_engine import StreamingVocoder
from tts_engine import load_model

model, _ = load_model(ROOT + "/models/Audio8-TTS-Realistic-0.6b")
torch.manual_seed(0)
T = 40
with torch.inference_mode():
    codec = model.load_codec(dtype=torch.float32)
    ref_codes = torch.randint(0, 1024, (10, T), device="cuda")
    ref = codec.decode(ref_codes[None].clone())[0, 0].float()
    v = StreamingVocoder(codec, 4, torch.device("cuda"))
    v.clear_many([0, 1, 2, 3])
    outs, half = [], T // 2
    for t in range(T):
        if t == half:
            snap = v.snapshot(0)
            d = (v.arena[0] != v.arena[1]).sum().item()
            print(f"before restore: arena rows 0/1 differ in {d} bytes; row 3 vs 0: {(v.arena[0] != v.arena[3]).sum().item()}")
            v.clear_many([3])
            v.load_many([3], snap)
            print(f"after restore: row 3 vs 0 differ in {(v.arena[0] != v.arena[3]).sum().item()} bytes")
        outs.append(v.step(ref_codes[:, t].view(1, -1).expand(4, -1).contiguous(), 4).clone())
    w = torch.stack(outs, 1).view(4, -1)
    snr = lambda a, r: float(10 * torch.log10((r ** 2).sum() / ((a - r) ** 2).sum().clamp(min=1e-20)))
    for r in range(4):
        print(f"row {r}: SNR full {snr(w[r], ref):.1f} dB, first half {snr(w[r, :half * 2048], ref[:half * 2048]):.1f}, "
              f"second half {snr(w[r, half * 2048:], ref[half * 2048:]):.1f}")
print("VOC_DONE")
