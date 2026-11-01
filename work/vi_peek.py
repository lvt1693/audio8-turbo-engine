"""Score the vi_eval wavs with whisper-large-v3-turbo called directly (resampled to 16 kHz, language vi forced; the
transformers ASR pipeline hallucinates "hãy subscribe..." on some of them): CER / syllable WER per condition, and per
file the duration, peak / RMS dBFS and transcript. Condition `refs`: the Vietnamese reference wavs of the voices against
their own transcripts (whisper's floor on good Vietnamese speech).
Run on .30: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 venv/bin/python work/vi_peek.py cond[,cond]"""
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vi_eval import LINES, OUT, SPK, edits, norm  # noqa: E402

NAME = "openai/whisper-large-v3-turbo"
REF_VOICES = ["pika-2_vi_normal", "pika-pitch-2_vi_normal", "pika-2_vi_normal_happy", "pika-pitch-2_vi_normal_happy",
              "ngan_vi_normal", "minh_vi_normal", "huyen_vi_normal", "mai_vi_normal", "phuong_vi_normal", "ly_vi_normal",
              "kieu-linh_vi_normal", "huynh-duong_vi_normal", "linh-clone_vi_normal", "nhu_vi_normal"]
proc = WhisperProcessor.from_pretrained(NAME)
model = WhisperForConditionalGeneration.from_pretrained(NAME, dtype=torch.float16).eval().to("cuda")


def transcribe(y, sr):
    if sr != 16000:
        y = AF.resample(torch.from_numpy(y), sr, 16000).numpy()
    x = proc(y, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
    with torch.inference_mode():
        gen = model.generate(x, language="vi", task="transcribe", max_new_tokens=160)
    return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()


summary = []
for cond in sys.argv[1].split(","):
    print(f"===== {cond}", flush=True)
    if cond == "refs":
        items = [(f"{SPK}/{v}.wav", open(f"{SPK}/{v}.txt", encoding="utf-8").read().strip()) for v in REF_VOICES
                 if os.path.exists(f"{SPK}/{v}.txt")]
    else:
        items = [(f, LINES[int(os.path.basename(f)[:2])]) for f in sorted(glob.glob(f"{OUT}/{cond}/*.wav"))]
    ce = cn = we = wn = 0
    for f, text in items:
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        peak = 20 * np.log10(max(np.abs(y).max(), 1e-9))
        rms = 20 * np.log10(max(np.sqrt((y ** 2).mean()), 1e-9))
        hyp = transcribe(y, sr)
        r, h = norm(text), norm(hyp)
        c, w = edits(r, h), edits(r.split(), h.split())
        ce, cn, we, wn = ce + c, cn + len(r), we + w, wn + len(r.split())
        print(f"{os.path.basename(f)[:28]:28s} {len(y) / sr:5.2f}s pk {peak:5.1f} rms {rms:5.1f} cer {c / len(r):5.1%} | {h[:100]}",
              flush=True)
    if cn:
        summary.append(f"{cond:12s} n={len(items):2d} CER {ce / cn:6.1%}  WER {we / wn:6.1%}")
        print(summary[-1], flush=True)
print("\n".join(["===== SUMMARY"] + summary + ["PEEK_DONE"]), flush=True)
