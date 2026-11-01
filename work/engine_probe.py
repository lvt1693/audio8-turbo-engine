"""Engine probe (GPU 0 of .30): the Audio8 FastEngine against the HF checkpoint.
  1. prompt: our token / code rows == ArkttsModel._prepare_prompt's
  2. prefill logits vs HF forward (semantic + EOS)
  3. greedy generation: frames until the first difference vs HF generate(do_sample=False)
  4. streaming vocoder vs HF decode (fp32 codec as the reference): SNR
  5. tick time per bucket (CUDA graph replay)
  6. sampled lines -> wavs to listen to
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 venv/bin/python work/engine_probe.py [B]"""
import os
import sys
import time

ROOT = "/home/ubuntu/truc_ai/Audio8"
sys.path.insert(0, ROOT + "/serve")
os.environ.setdefault("TTS_GEMM_TUNE_CACHE", ROOT + "/data/gemm_tune.json")

import numpy as np
import soundfile as sf
import torch

from create_speakers import encode_refs
from fast_engine import FastEngine
from tts_engine import load_model, voice_item

MODEL = ROOT + "/models/Audio8-TTS-Realistic-0.6b"
REFS = ROOT + "/data/refs/speakers_06b_v3"
OUT = ROOT + "/work/out_probe"
os.makedirs(OUT, exist_ok=True)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 64
SKIP_PARITY = os.environ.get("SKIP_PARITY") == "1"

model, proc = load_model(MODEL)
sr = model.config.codec_sample_rate


def ref(name):
    d = REFS if os.path.exists(f"{REFS}/{name}.txt") else "/home/ubuntu/truc_ai/Qwen3-TTS/data/speakers/speakers_06b_v3"
    return open(f"{d}/{name}.txt", encoding="utf-8").read().strip(), f"{d}/{name}.wav"   # (read only)


def snr(a, r):
    n = min(len(a), len(r))
    a, r = a[:n], r[:n]
    return 10 * np.log10((r ** 2).sum() / max(((a - r) ** 2).sum(), 1e-20))


CASES = [
    ("pika-pitch-2_vi_normal_happy", "Xin chào các bạn nhỏ! Hôm nay chúng mình sẽ cùng nhau học về các loài động vật sống trong rừng nhé."),
    ("pika-2_vi_normal", "Cậu có biết không, con voi là loài vật trên cạn to lớn nhất đấy. Nó có cái vòi rất dài để uống nước."),
    ("pika-pitch-2_en_normal", "Hello there! Today we are going to learn some new words about animals. Are you ready?"),
]
vo = {}
for name, _ in CASES:
    rt, wav = ref(name)
    vo[name] = voice_item(encode_refs(model, [wav])[0], rt)
    print(f"voice {name}: {vo[name].ref_code.shape[0]} frames", flush=True)

# ---------------- HF greedy reference (before the engine touches the model) ----------------
hf = {}
if not SKIP_PARITY:
    for name, text in CASES[:2]:
        it = vo[name]
        inputs = proc(text=[text], reference_text=[it.ref_text], reference_codes=[it.ref_code.t()], return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.inference_mode():
            prompt, mask = model._prepare_prompt(None, None, inputs["prefix_input_ids"], inputs["prefix_attention_mask"],
                                                 inputs["suffix_input_ids"], inputs["suffix_attention_mask"],
                                                 inputs["reference_codes"], inputs["reference_code_lengths"])
            for L in list(model.layers) + list(model.fast_layers):     # the previous generate's caches
                L.attention.kv_cache = None
            lg = model(input_ids=prompt, attention_mask=mask).logits[0, -1].float()
            sel = torch.cat([lg[model.config.semantic_begin_id:model.config.semantic_end_id + 1],
                             lg[model.config.eos_token_id:model.config.eos_token_id + 1]])
            out = model.generate(**inputs, do_sample=False, max_new_tokens=400, return_dict_in_generate=True)
        hf[name] = dict(prompt=prompt[0].cpu(), logits=sel.cpu(), codes=out.codes[0, :, :int(out.code_lengths[0])].cpu())
        print(f"HF greedy {name}: {hf[name]['codes'].shape[1]} frames", flush=True)
    # codec reference decodes of the first line's codes: HF bf16 and fp32
    c = hf[CASES[0][0]]["codes"].cuda()
    with torch.inference_mode():
        w_bf16 = model.decode_audio(c[None])[0][0].float().cpu().numpy()
        codec32 = model.load_codec(dtype=torch.float32)
        w_fp32 = codec32.decode(c[None].clone())[0, 0].float().cpu().numpy()
        model.load_codec(dtype=torch.bfloat16)

# ---------------- engine ----------------
gp = dict(do_sample=False, temperature=0.8, top_k=50, top_p=0.95, max_new_tokens=1024)
t0 = time.perf_counter()
eng = FastEngine(model, proc.tokenizer, gp, max_slots=B)
eng.capture(log=lambda s: print(s, flush=True))
print(f"engine up in {time.perf_counter() - t0:.1f}s; VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def run(reqs, max_frames=600, prime=True):
    n = len(reqs)
    slots = list(range(n))
    if prime:
        eng.prime_speakers([(r["key"], r["item"]) for r in reqs])
    lens = eng.admit(slots, reqs)
    b = eng.bucket_for(n)
    frames, auds = [[] for _ in range(n)], [[] for _ in range(n)]
    done, started = [False] * n, [False] * n
    for _ in range(max_frames + 2):
        eng.replay(b)
        toks, fr, au = eng.tok_in[:n].cpu(), eng.frame_prev[:n].cpu(), eng.audio[:n].cpu()
        for i in range(n):
            if done[i]:
                continue
            if started[i]:
                auds[i].append(au[i].numpy())
            started[i] = True
            if int(toks[i]) == eng.eos:
                done[i] = True
                continue
            frames[i].append(fr[i])
        if all(done):
            break
    return lens, [torch.stack(f, 1) if f else torch.zeros(10, 0, dtype=torch.long) for f in frames], \
        [np.concatenate(a) if a else np.zeros(0, np.float32) for a in auds]


if not SKIP_PARITY:
    # 1 + 2: prompt rows and prefill logits
    for name, text in CASES[:2]:
        eng._parts_init() if eng._parts is None else None
        sem, cb, aud = eng._prompt_ids(vo[name])
        from fast_engine import _clean_text
        rest = torch.tensor(eng._enc(_clean_text(text)) + eng._parts.tail)
        ours = torch.cat([sem, rest])
        hp = hf[name]["prompt"]
        same_sem = ours.shape[0] == hp.shape[1] and bool((ours == hp[0]).all())
        ocb = torch.cat([cb, torch.zeros(len(rest), 10, dtype=torch.long)]).t()
        same_cb = same_sem and bool((ocb == hp[1:]).all())
        print(f"[prompt] {name}: len ours {ours.shape[0]} hf {hp.shape[1]} | semantic row equal {same_sem} | codes equal {same_cb}",
              flush=True)
        eng.admit([0], [{"text": text, "item": vo[name], "key": name}])
        ol = eng._head(eng.hid[:1])[0].cpu()
        hl = hf[name]["logits"]
        print(f"[logits] {name}: max|d| {float((ol - hl).abs().max()):.3f} (range {float(hl.max() - hl.min()):.1f}) "
              f"argmax ours {int(ol.argmax())} hf {int(hl.argmax())} top5 overlap "
              f"{len(set(ol.topk(5).indices.tolist()) & set(hl.topk(5).indices.tolist()))}/5", flush=True)
    # 3: greedy frames
    for name, text in CASES[:2]:
        _, fr, _ = run([{"text": text, "item": vo[name], "key": name}], max_frames=400, prime=False)
        o, h = fr[0], hf[name]["codes"]
        n = min(o.shape[1], h.shape[1])
        eq = (o[:, :n] == h[:, :n]).all(0)
        first = int((~eq).nonzero()[0]) if (~eq).any() else n
        sem_eq = (o[0, :n] == h[0, :n]).float().mean().item()
        print(f"[greedy] {name}: ours {o.shape[1]} frames, hf {h.shape[1]}; identical frames up to {first}; "
              f"codebook-0 agreement {sem_eq:.2%} over {n}", flush=True)
    # 4: vocoder
    c = hf[CASES[0][0]]["codes"].cuda()
    with torch.inference_mode():
        eng.voc.clear_many([0])
        w = torch.cat([eng.voc.step(c[:, t][None], 1)[0] for t in range(c.shape[1])]).float().cpu().numpy()
    print(f"[vocoder] streaming vs fp32 decode: SNR {snr(w, w_fp32):.1f} dB | HF bf16 vs fp32: {snr(w_bf16, w_fp32):.1f} dB",
          flush=True)
    sf.write(f"{OUT}/voc_stream.wav", w, sr)
    sf.write(f"{OUT}/voc_hf_fp32.wav", w_fp32, sr)

# 5: tick time per bucket
with torch.inference_mode():
    for b in eng.buckets:
        for _ in range(3):
            eng.replay(b)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(20):
            eng.replay(b)
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / 20
        print(f"[tick] b={b:4d}: {ms:6.2f} ms  RTF {ms / (1000 * eng.up / sr):.3f}", flush=True)

# 6: sampled lines, all three at once (the served settings)
eng.samp.do_sample = eng.cb_samp.do_sample = True
reqs = [{"text": t, "item": voice_item(vo[n].ref_code, vo[n].ref_text.rstrip(".") + "."), "key": n} for n, t in CASES]
for k in range(2):
    _, fr, au = run(reqs, max_frames=600)
    for (n, _), a, f in zip(CASES, au, fr):
        sf.write(f"{OUT}/{n}_s{k}.wav", a, sr)
        print(f"[sample] {n} take {k}: {f.shape[1]} frames, {len(a) / sr:.2f}s", flush=True)
print("PROBE_DONE", flush=True)
