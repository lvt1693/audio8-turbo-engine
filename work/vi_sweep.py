"""Can decoding or the voice make Audio8's Vietnamese better: the vi_eval lines (normalized as the API does) through a
FastEngine of its own (ticks run eagerly, so the sampling settings can change between conditions), TAKES takes each,
per condition (voice, semantic temperature, codebook temperature, top-k, top-p). Wavs -> work/vi_eval/<cond>/NN_tK.wav;
score them with vi_peek.py.
Run on .30: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 venv/bin/python work/vi_sweep.py"""
import os
import sys

ROOT = "/home/ubuntu/truc_ai/Audio8"
sys.path.insert(0, ROOT + "/serve")
sys.path.insert(0, ROOT + "/work")
os.environ.setdefault("TTS_GEMM_TUNE_CACHE", ROOT + "/data/gemm_tune.json")
os.environ.setdefault("TTS_VOC_TUNE_CACHE", ROOT + "/data/voc_tune.json")
os.environ.setdefault("TRITON_CACHE_DIR", ROOT + "/data/cache/triton")     # the API's (~/.triton: another Python's)

import requests  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from fast_engine import FastEngine  # noqa: E402
from tts_engine import TTSEngine, load_model, voice_item  # noqa: E402
from vi_eval import A8, LINES, OUT, SPK  # noqa: E402

TAKES = 2
PP2H = "pika-pitch-2_vi_normal_happy"
# name, voice, semantic temperature, codebook temperature, top_k, top_p
CONDS = [
    ("e_base", PP2H, 0.8, 0.8, 50, 0.95),
    ("e_t06", PP2H, 0.6, 0.6, 50, 0.95),
    ("e_t04", PP2H, 0.4, 0.4, 50, 0.95),
    ("e_s04", PP2H, 0.4, 0.8, 50, 0.95),
    ("e_k10", PP2H, 0.8, 0.8, 10, 0.95),
    ("e_pp2", "pika-pitch-2_vi_normal", 0.8, 0.8, 50, 0.95),
    ("e_p2", "pika-2_vi_normal", 0.8, 0.8, 50, 0.95),
    ("e_ngan", "ngan_vi_normal", 0.8, 0.8, 50, 0.95),
    ("e_minh", "minh_vi_normal", 0.8, 0.8, 50, 0.95),
    ("e_huyen", "huyen_vi_normal", 0.8, 0.8, 50, 0.95),
    ("e_mai", "mai_vi_normal", 0.8, 0.8, 50, 0.95),
    ("e_phuong", "phuong_vi_normal", 0.8, 0.8, 50, 0.95),
]

texts = [requests.get(A8 + "/v2/normalize", params={"text": t}).json()["text"] for t in LINES]
model, proc = load_model(ROOT + "/models/Audio8-TTS-Realistic-0.6b")
sr = model.config.codec_sample_rate
n = len(texts) * TAKES
eng = FastEngine(model, proc.tokenizer, dict(do_sample=True, temperature=0.8, top_k=50, top_p=0.95,
                                             max_new_tokens=1024), max_slots=n)
eng.capture(log=lambda s: print(s, flush=True))


def voice(name):
    p = torch.load(f"{SPK}/{name}.pt", map_location="cpu", weights_only=True)
    return voice_item(p["ref_code"].long(), TTSEngine._close_ref_text(p.get("ref_text", "")))


@torch.inference_mode()
def run(reqs, max_frames=700):
    k = len(reqs)
    eng.admit(list(range(k)), reqs)
    b = eng.bucket_for(k)
    auds = [[] for _ in range(k)]
    done, started = [False] * k, [False] * k
    for _ in range(max_frames + 2):
        eng._tick(b)
        toks, au = eng.tok_in[:k].cpu(), eng.audio[:k].float().cpu()
        for i in range(k):
            if done[i]:
                continue
            if started[i]:
                auds[i].append(au[i].numpy())
            started[i] = True
            done[i] = int(toks[i]) == eng.eos
        if all(done):
            break
    return [torch.cat([torch.from_numpy(x) for x in a]).numpy() if a else None for a in auds]


for name, spk, ts, tc, k, p in CONDS:
    eng.samp.t, eng.samp.k, eng.samp.p = ts, k, p
    eng.cb_samp.t, eng.cb_samp.k, eng.cb_samp.p = tc, min(k, eng.cbk), p
    it = voice(spk)
    torch.manual_seed(1234)
    reqs = [{"text": t, "item": it, "key": spk} for t in texts for _ in range(TAKES)]
    wavs = run(reqs)
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    for j, w in enumerate(wavs):
        if w is not None:
            sf.write(f"{OUT}/{name}/{j // TAKES:02d}_t{j % TAKES}.wav", w, sr)
    print(f"{name}: {spk} t_sem {ts} t_cb {tc} top_k {k} top_p {p} -> {sum(w is not None for w in wavs)} wavs, "
          f"ref {it.ref_code.shape[0] / 21.53:.1f}s", flush=True)
print("SWEEP_DONE", flush=True)
