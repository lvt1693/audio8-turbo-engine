"""Baseline: the unmodified HF Audio8 model (trust_remote_code) on one GPU -- speed and a few wavs to listen to.
Run on .30: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 venv/bin/python work/baseline_hf.py"""
import os
import sys
import time

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

ROOT = "/home/ubuntu/truc_ai/Audio8"
MODEL = f"{ROOT}/models/Audio8-TTS-Realistic-0.6b"
SPK = "/home/ubuntu/truc_ai/Qwen3-TTS/data/speakers/speakers_06b_v3"
OUT = f"{ROOT}/work/out_baseline"
os.makedirs(OUT, exist_ok=True)

proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).eval().to("cuda")
print("loaded", sum(p.numel() for p in model.parameters()) / 1e6, "M params", flush=True)

CASES = [
    ("pika-pitch-2_vi_normal_happy", "vi1", "Xin chào các bạn nhỏ! Hôm nay chúng mình sẽ cùng nhau học về các loài động vật sống trong rừng nhé."),
    ("pika-2_vi_normal", "vi2", "Cậu có biết không, con voi là loài vật trên cạn to lớn nhất đấy. Nó có cái vòi rất dài để uống nước."),
    ("pika-pitch-2_en_normal", "en1", "Hello there! Today we are going to learn some new words about animals. Are you ready?"),
]
sr = model.config.codec_sample_rate
for spk, tag, text in CASES:
    ref_text = open(f"{SPK}/{spk}.txt", encoding="utf-8").read().strip()
    inputs = proc(text=[text], reference_audio=[f"{SPK}/{spk}.wav"], reference_text=[ref_text], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    for run in range(2):
        torch.manual_seed(run)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=50,
                                 do_sample=True, return_dict_in_generate=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            wav, lens = model.decode_audio(out.codes)
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        n = int(lens[0])
        dur = n / sr
        print(f"{tag} run{run}: frames={int(out.code_lengths[0])} audio={dur:.2f}s gen={t1-t0:.2f}s "
              f"decode={t2-t1:.2f}s RTF={(t2-t0)/max(dur,1e-3):.2f}", flush=True)
        sf.write(f"{OUT}/{tag}_run{run}.wav", wav[0, :n].float().cpu().numpy(), sr)
print("BASELINE_DONE")
