"""Why does Audio8's Vietnamese sound bad: the same Vietnamese lines through
  a8_dual    the Audio8 API (:29100), dual endpoint defaults (north -> its styled voice)
  a8_plain   the same, instruct=none (the plain voice)
  a8_<spk>   the Audio8 API, multilang endpoint, other Vietnamese voices
  queen_dual Queen TTS (:29000, read-only requests), dual endpoint defaults: the bar
  hf_voice   the unmodified HF checkpoint (model.generate), the same reference as a8_dual: engine vs model
  hf_noref   the HF checkpoint with no reference: the model's own Vietnamese
scored by whisper-large-v3-turbo (language vi): CER / syllable WER per line, transcripts printed.
Run on .30: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 venv/bin/python work/vi_eval.py [cond,...]"""
import io
import os
import re
import sys
import unicodedata

import numpy as np
import requests
import soundfile as sf
import torch

ROOT = "/home/ubuntu/truc_ai/Audio8"
MODEL = f"{ROOT}/models/Audio8-TTS-Realistic-0.6b"
SPK = f"{ROOT}/data/speakers/speakers_a8"
OUT = f"{ROOT}/work/vi_eval"
A8, QUEEN = "http://127.0.0.1:29100", "http://127.0.0.1:29000"
EXTRA_VOICES = ["ngan", "minh"]

LINES = [
    "Xin chào các bạn nhỏ! Hôm nay chúng mình sẽ cùng nhau học về các loài động vật sống trong rừng nhé.",
    "Con voi là loài vật trên cạn to lớn nhất. Nó có cái vòi rất dài để uống nước và tắm mát.",
    "Cậu giỏi quá! Thêm một lần nữa nhé, mình cùng đếm từ một đến mười nào.",
    "Mình là Pika, rất vui được làm quen với cậu. Cậu tên là gì vậy?",
    "Mùa thu lá vàng rơi đầy trên con đường nhỏ dẫn vào trường học.",
    "Bé hãy rửa tay thật sạch bằng xà phòng trước khi ăn cơm nhé.",
    "Ông mặt trời thức dậy, chiếu những tia nắng ấm áp xuống cánh đồng lúa.",
    "Chú mèo con nằm cuộn tròn ngủ ngon lành bên cạnh lò sưởi.",
    "Cậu có thích ăn kem không? Tớ thích nhất là kem sô cô la đấy.",
    "Hôm qua mưa to quá nên chúng tớ phải ở nhà đọc truyện cổ tích.",
    "Chiếc thuyền giấy nhỏ trôi bồng bềnh trên mặt nước trong veo.",
    "Bạn nhỏ ơi, đừng quên mặc áo ấm khi ra ngoài trời lạnh nhé.",
    "Những chú chim sẻ líu lo hót trên cành cây trước hiên nhà.",
    "Mẹ dặn tớ phải ngủ sớm để sáng mai dậy đi học đúng giờ.",
    "Trên bầu trời đêm có hàng nghìn ngôi sao lấp lánh như kim cương.",
    "Mình cùng hát một bài thật vui để chào đón ngày mới nào!",
]


def norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def api_wav(base, path, params):
    r = requests.post(base + path, json=dict(params, stream=False), timeout=300)
    r.raise_for_status()
    y, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return y, sr, r.headers.get("X-Speaker", "")


def main():
    conds = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        ["a8_dual", "a8_plain", "queen_dual", "hf_voice", "hf_noref"] + [f"a8_{v}" for v in EXTRA_VOICES]
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16,
                   device="cuda")

    def transcribe(y, sr):
        if y.ndim > 1:
            y = y.mean(1)
        return asr({"raw": y.astype(np.float32), "sampling_rate": sr},
                   generate_kwargs={"language": "vi", "task": "transcribe"})["text"].strip()

    hf = None
    voice = None
    summary = []
    for cond in conds:
        os.makedirs(f"{OUT}/{cond}", exist_ok=True)
        ce = cn = we = wn = 0
        print(f"===== {cond}", flush=True)
        for i, text in enumerate(LINES):
            spk = ""
            try:
                if cond == "a8_dual":
                    y, sr, spk = api_wav(A8, "/v2/speech/dual_vietnamese_english", {"text": text})
                    voice = voice or spk
                elif cond == "a8_plain":
                    y, sr, spk = api_wav(A8, "/v2/speech/dual_vietnamese_english", {"text": text, "instruct": "none"})
                elif cond == "queen_dual":
                    y, sr, spk = api_wav(QUEEN, "/v2/speech/dual_vietnamese_english", {"text": text})
                elif cond.startswith("a8_"):
                    y, sr, spk = api_wav(A8, "/v2/multilang", {"text": text, "speaker": cond[3:], "langs": "vi",
                                                              "instruct": "none"})
                else:
                    if hf is None:
                        from transformers import AutoModel, AutoProcessor
                        proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
                        model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                                          dtype=torch.bfloat16).eval().to("cuda")
                        hf = (proc, model)
                    proc, model = hf
                    ntext = requests.get(A8 + "/v2/normalize", params={"text": text}).json()["text"]
                    if cond == "hf_voice":
                        spk = voice or "pika-pitch-2_vi_normal_happy"
                        ref_text = open(f"{SPK}/{spk}.txt", encoding="utf-8").read().strip()
                        inputs = proc(text=[ntext], reference_audio=[f"{SPK}/{spk}.wav"], reference_text=[ref_text],
                                      return_tensors="pt")
                    else:
                        inputs = proc(text=[ntext], return_tensors="pt")
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    torch.manual_seed(i)
                    with torch.inference_mode():
                        out = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=50,
                                             do_sample=True, return_dict_in_generate=True)
                        wav, lens = model.decode_audio(out.codes)
                    sr = model.config.codec_sample_rate
                    y = wav[0, :int(lens[0])].float().cpu().numpy()
            except Exception as e:
                print(f"{i:02d} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            sf.write(f"{OUT}/{cond}/{i:02d}.wav", y, sr)
            hyp = transcribe(y, sr)
            r, h = norm(text), norm(hyp)
            c, w = edits(r, h), edits(r.split(), h.split())
            ce, cn, we, wn = ce + c, cn + len(r), we + w, wn + len(r.split())
            print(f"{i:02d} cer={c / len(r):5.1%} {len(y) / sr:5.2f}s {spk}\n   ref: {r}\n   hyp: {h}", flush=True)
        if cn:
            summary.append(f"{cond:12s} CER {ce / cn:6.1%}  WER {we / wn:6.1%}")
            print(summary[-1], flush=True)
    print("\n".join(["===== SUMMARY"] + summary + ["VI_EVAL_DONE"]), flush=True)


if __name__ == "__main__":
    main()
