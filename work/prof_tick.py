"""Where a tick's time goes, per part, at a few batch sizes (CUDA graphs per part, GPU 0 of .30).
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 venv/bin/python work/prof_tick.py 32,64,128"""
import os
import sys

ROOT = "/home/ubuntu/truc_ai/Audio8"
sys.path.insert(0, ROOT + "/serve")
os.environ.setdefault("TTS_GEMM_TUNE_CACHE", ROOT + "/data/gemm_tune.json")

import torch
import torch.nn.functional as F

from fast_engine import FastEngine
from tts_engine import load_model

BS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32,64,128").split(",")]
model, proc = load_model(ROOT + "/models/Audio8-TTS-Realistic-0.6b")
eng = FastEngine(model, proc.tokenizer, dict(do_sample=True), max_slots=max(BS), buckets=BS)
with torch.inference_mode():
    eng._tune(BS + [2 * b for b in BS], print)


def gtime(fn, n=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


v = eng.voc
with torch.inference_mode():
    for b in BS:
        codes = torch.randint(0, 1024, (b, 10), device="cuda")
        x1024 = torch.randn(b, 1024, device="cuda", dtype=torch.bfloat16)
        x4 = torch.randn(b, 4, 1024, device="cuda", dtype=torch.bfloat16)
        r = {}
        r["tick"] = gtime(lambda: eng._tick(b))
        r["voc.step"] = gtime(lambda: v.step(codes, b))
        vpos = v.pos[:b].contiguous()          # (a strided view into the state arena)
        r["voc.post"] = gtime(lambda: v.post.forward(x1024.clone(), vpos, v.ar[:b], (vpos + 1).clamp(max=128),
                                                       rowmap=False))

        def ups():
            x = x1024.unsqueeze(1)
            for u in v.ups:
                T = x.shape[1]
                x = F.linear(x, u.tw, u.tb).view(b, 2 * T, -1)
                T = 2 * T
                rr = x
                xin = torch.cat([u.st[:b], x], 1)
                y = u.dwb + xin[:, 0:T] * u.dww[0]
                for j in range(1, u.k):
                    y = y + xin[:, j:j + T] * u.dww[j]
                y = F.layer_norm(y, (y.shape[-1],), u.ln_w, u.ln_b, u.ln_eps)
                y = F.linear(F.gelu(F.linear(y, u.w1, u.b1)), u.w2, u.b2)
                x = rr + u.gamma * y
            return x
        r["voc.ups"] = gtime(ups)
        cl = v.cl
        xs = [x4.to(torch.float16)]
        xs.append(cl.dec_in(xs[0], b))
        r["voc.dec_in"] = gtime(lambda: cl.dec_in(xs[0], b))
        for k, (tc, units) in enumerate(cl.blocks):
            xi = xs[-1]
            r[f"voc.blk{k}.trans"] = gtime(lambda: tc(xi, b))
            y = tc(xi, b)
            for j, (c1, c2, _fu) in enumerate(units):
                yy = y
                r[f"voc.blk{k}.ru{j}"] = gtime(lambda: c2(c1(yy, b), b, res=yy))
            xs.append(y)
        xl = xs[-1]
        r["voc.final"] = gtime(lambda: cl.final(xl, b))
        hid = eng.hid[:b]
        c0 = torch.randint(0, 4096, (b,), device="cuda")
        r["fast(9 passes+draws)"] = gtime(lambda: eng._fast_codes(hid, c0, b))
        xx = torch.randn(2 * b, 896, device="cuda", dtype=torch.bfloat16)
        r["fast.2tok pass"] = gtime(lambda: eng.fast.forward(xx.clone(), eng.f_pos0[:2 * b], eng.f_slot0[:2 * b],
                                                              eng.f_len0[:2 * b]))
        xb = torch.randn(b, 896, device="cuda", dtype=torch.bfloat16)
        r["fast.1tok pass"] = gtime(lambda: eng.fast.forward(xb.clone(), eng.f_pos[5][:b], eng.ar[:b], eng.f_len[5][:b],
                                                              rowmap=False))
        lg = torch.randn(b, 4096, device="cuda")
        r["cb draw"] = gtime(lambda: eng.cb_samp(lg))
        eng.cpos[:b] = 300
        r["slow step"] = gtime(lambda: eng.slow.forward(xb.clone(), eng.cpos[:b], eng.ar[:b], eng.cpos[:b] + 1,
                                                         nsplit=max(1, min(16, -(-168 // (b * 2)))), rowmap=False))
        r["head+sem draw"] = gtime(lambda: eng._sample_sem(eng._head(xb), b))
        print(f"\n=== b={b}: tick {r['tick']:.2f} ms (RTF {r['tick'] / 46.44:.3f})")
        for k, t in r.items():
            if k != "tick":
                print(f"  {k:22s} {t:7.3f} ms  {100 * t / r['tick']:5.1f}%")
print("PROF_DONE")
