"""Teacher-forced parity, engine stacks vs the HF checkpoint, on HF's own greedy frames:
  slow (packed prefill path), slow (decode path, one frame at a time), fast AR (each codebook position).
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 venv/bin/python work/tf_probe.py"""
import os
import sys

ROOT = "/home/ubuntu/truc_ai/Audio8"
sys.path.insert(0, ROOT + "/serve")
os.environ.setdefault("TTS_GEMM_TUNE_CACHE", ROOT + "/data/gemm_tune.json")

import torch

from create_speakers import encode_refs
from fast_engine import FastEngine, PrefillPlan, _to_dev
from tts_engine import load_model, voice_item

MODEL = ROOT + "/models/Audio8-TTS-Realistic-0.6b"
REFS = ROOT + "/data/refs/speakers_06b_v3"
NAME = "pika-pitch-2_vi_normal_happy"
TEXT = "Xin chào các bạn nhỏ! Hôm nay chúng mình sẽ cùng nhau học về các loài động vật sống trong rừng nhé."

model, proc = load_model(MODEL)
cfg = model.config
sb, se, eos = cfg.semantic_begin_id, cfg.semantic_end_id, cfg.eos_token_id
d = REFS if os.path.exists(f"{REFS}/{NAME}.txt") else "/home/ubuntu/truc_ai/Qwen3-TTS/data/speakers/speakers_06b_v3"
rt = open(f"{d}/{NAME}.txt", encoding="utf-8").read().strip()
it = voice_item(encode_refs(model, [f"{d}/{NAME}.wav"])[0], rt)
inputs = proc(text=[TEXT], reference_text=[rt], reference_codes=[it.ref_code.t()], return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}


def clear_caches():
    for L in list(model.layers) + list(model.fast_layers):
        L.attention.kv_cache = None


def sel(lg):
    return torch.cat([lg[..., sb:se + 1], lg[..., eos:eos + 1]], -1).float()


with torch.inference_mode():
    prompt, mask = model._prepare_prompt(None, None, inputs["prefix_input_ids"], inputs["prefix_attention_mask"],
                                         inputs["suffix_input_ids"], inputs["suffix_attention_mask"],
                                         inputs["reference_codes"], inputs["reference_code_lengths"])
    out = model.generate(**inputs, do_sample=False, max_new_tokens=60, return_dict_in_generate=True)
    codes = out.codes[0, :, :int(out.code_lengths[0])]                        # [10, T]
    T, L = codes.shape[1], prompt.shape[-1]
    cols = torch.cat([(codes[0] + sb)[None], codes], 0)                         # [11, T]
    full = torch.cat([prompt[0], cols], 1)[None]                                # [1, 11, L+T]
    clear_caches()
    o = model(input_ids=full, attention_mask=torch.ones(1, L + T, dtype=torch.long, device="cuda"))
    hf_sem = sel(o.logits[0, L - 1:L - 1 + T])                                  # predicts frame t's semantic
    nh = model.norm(o.hidden_states)[0, L - 1:L - 1 + T]                        # normed slow hidden per frame
    print(f"T={T} L={L}; HF forward argmax == generate's semantic: "
          f"{(hf_sem.argmax(-1) == codes[0]).float().mean():.2%}", flush=True)
    # HF fast AR, teacher-forced
    model._setup_generation_caches(1, 0, torch.bfloat16)
    hf_fast = []
    for t in range(T):
        model._fast_step(nh[t][None, None], 0)
        row = []
        for i in range(1, cfg.num_codebooks):
            row.append(model._fast_step(model.fast_embeddings(codes[i - 1, t])[None, None], i)[0].float())
        hf_fast.append(torch.stack(row))                                         # [9, 4096]
    hf_fast = torch.stack(hf_fast)                                               # [T, 9, 4096]
    hf_gen_ok = (hf_fast.argmax(-1) == codes[1:].t()).float().mean()
    print(f"HF fast teacher-forced argmax == generate's codebooks: {hf_gen_ok:.2%}", flush=True)
    clear_caches()

    eng = FastEngine(model, proc.tokenizer, dict(do_sample=False), max_slots=4)
    eng._parts_init()
    # slow, packed prefill path over the whole teacher-forced sequence
    sem_f, cb_f = full[0, 0].cpu(), full[0, 1:].t().cpu()
    aud = (sem_f >= sb) & (sem_f <= se)
    x = eng._prompt_embeds(sem_f, cb_f, aud)
    n = x.shape[0]
    plan = PrefillPlan([0], [n], [0], [0], eng.dev)
    h = eng.slow.forward(x.clone(), torch.arange(n, device="cuda", dtype=torch.int32),
                         torch.zeros(n, device="cuda", dtype=torch.int32), None, plan=plan)
    our_sem = eng._head(h[L - 1:L - 1 + T])
    print(f"[slow prefill path] argmax agree {(our_sem.argmax(-1) == hf_sem.argmax(-1)).float().mean():.2%}, "
          f"max|d| {float((our_sem - hf_sem).abs().max()):.3f}, mean|d| {float((our_sem - hf_sem).abs().mean()):.4f}; "
          f"hidden max|d| {float((h[L - 1:L - 1 + T].float() - nh.float()).abs().max()):.3f}", flush=True)
    # slow, decode path: prefill the prompt, then one frame a step
    eng.admit([1], [{"text": TEXT, "item": it, "key": NAME}])
    agree, md = [], 0.0
    for t in range(T - 1):
        pos = eng.cpos[1:2]
        hh = eng.slow.forward(eng._embed((codes[0, t] + sb).view(1), codes[:, t].view(1, -1)), pos, eng.ar[1:2],
                              pos + 1, rowmap=True)
        lg = eng._head(hh)[0]
        agree.append(int(lg.argmax()) == int(hf_sem[t + 1].argmax()))
        md = max(md, float((lg - hf_sem[t + 1]).abs().max()))
        eng.cpos[1:2] += 1
    print(f"[slow decode path] argmax agree {sum(agree) / len(agree):.2%}, max|d| {md:.3f}", flush=True)
    # fast AR, teacher-forced, from HF's normed hidden
    ag = torch.zeros(9)
    mx = torch.zeros(9)
    for t in range(T):
        xx = torch.stack([nh[t], eng.fast_emb[codes[0, t]]]).view(2, -1)
        hf2 = eng.fast.forward(xx.clone(), eng.f_pos0[:2], eng.f_slot0[:2], eng.f_len0[:2])
        lgs = [eng.mm(hf2[1:2], eng.fast_out).float()[0]]
        for i in range(2, cfg.num_codebooks):
            hi = eng.fast.forward(eng.fast_emb[codes[i - 1, t]].view(1, -1).clone(), eng.f_pos[i][:1], eng.ar[:1],
                                  eng.f_len[i][:1], rowmap=False)
            lgs.append(eng.mm(hi, eng.fast_out).float()[0])
        lg = torch.stack(lgs)
        ag += (lg.argmax(-1) == hf_fast[t].argmax(-1)).float().cpu()
        mx = torch.maximum(mx, (lg - hf_fast[t]).abs().amax(-1).cpu())
    print("[fast] argmax agree per codebook 1..9: " + " ".join(f"{a / T:.0%}" for a in ag.tolist()), flush=True)
    print("[fast] max|d| per codebook 1..9:      " + " ".join(f"{m:.2f}" for m in mx.tolist()), flush=True)
    # streaming vocoder (rows 1 and 2 side by side, row 2 restarted halfway from a snapshot) vs the fp32 decode
    codec32 = model.load_codec(dtype=torch.float32)
    ref = codec32.decode(codes[None].clone())[0, 0].float()
    model.load_codec(dtype=torch.bfloat16)
    v = eng.voc
    v.clear_many([1, 2])
    outs, half = [], T // 2
    for t in range(T):
        if t == half:
            snap = v.snapshot(1)
            v.clear_many([2])
            v.load_many([2], snap)
        w = v.step(codes[:, t].view(1, -1).expand(3, -1).contiguous(), 3)
        outs.append(w[1:3])
    w = torch.cat(outs, 1).float()
    snr = lambda a, r: float(10 * torch.log10((r ** 2).sum() / ((a - r) ** 2).sum().clamp(min=1e-20)))
    print(f"[vocoder] row 1 SNR {snr(w[0], ref):.1f} dB, row 2 (snapshot-restored halfway) "
          f"{snr(w[1, half * 2048:], ref[half * 2048:]):.1f} dB; rows 1 vs 2 max|d| first half "
          f"{float((w[0, :half * 2048] - w[1, :half * 2048]).abs().max()):.4f}, second half "
          f"{float((w[0, half * 2048:] - w[1, half * 2048:]).abs().max()):.4f}; len w {w.shape[1]} ref {ref.shape[0]}",
          flush=True)
print("TF_DONE")
