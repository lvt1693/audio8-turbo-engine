"""Fused residual unit (fast_kernels.ru_fused) vs the two-conv path: exactness over two chunks (state carried) and a
tile-config sweep per channel count. Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 venv/bin/python
work/ru_sweep.py [B]"""
import itertools
import os
import sys

ROOT = "/home/ubuntu/truc_ai/Audio8"
sys.path.insert(0, ROOT + "/serve")

import torch

from fast_engine import StreamingVocoder, _FusedRU
from fast_kernels import ru_fused, snake_cat
from tts_engine import load_model

B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
model, _ = load_model(ROOT + "/models/Audio8-TTS-Realistic-0.6b")
with torch.inference_mode():
    codec = model.load_codec(dtype=torch.float32)
    voc = StreamingVocoder(codec, B, torch.device("cuda"))


def gtime(fn, n=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


T_OF = {768: 32, 384: 256, 192: 1024, 96: 2048}
with torch.inference_mode():
    for bi, (tc, units) in enumerate(voc.cl.blocks):
        for ui, (c1, c2, _) in enumerate(units):
            C = c1.Cin
            T = T_OF[C]
            if T < c1.ctx or C > 192 or (c1.dil != 1 and os.environ.get("ALL_DIL") != "1"):
                continue
            fu = _FusedRU(c1, c2, torch.device("cuda"))
            torch.manual_seed(0)
            xs = [(torch.randn(B, T, C, device="cuda") * 0.3).half() for _ in range(2)]
            st0 = (torch.randn(B, c1.ctx, C, device="cuda") * 0.3).half()
            # reference: two chunks through the two-conv path
            c1.state[:B].copy_(st0)
            ref = [c2(c1(x, B), B, res=x) for x in xs]
            ref_st = c1.state[:B].clone()
            c1.state[:B].copy_(st0)
            got = [fu(x, B) for x in xs]
            got_st = c1.state[:B].clone()
            err = max(float((g.float() - r.float()).abs().max()) for g, r in zip(got, ref))
            scale = max(float(r.float().abs().max()) for r in ref)
            st_err = float((got_st.float() - ref_st.float()).abs().max())
            t_ref = gtime(lambda: c2(c1(xs[0], B), B, res=xs[0]))
            best = None
            print(f"blk{bi} ru{ui} C={C} dil={c1.dil}: max|d| {err:.4f} (|ref| {scale:.2f}), state {st_err:.4f}; "
                  f"two-conv {t_ref:.3f} ms", flush=True)
            for bm, bk, bn2, nw, ns in itertools.product((16, 32, 64), (32, 64), (64,), (4, 8), (1, 2)):
                if bm * fu.CP > 16384 or bn2 > fu.CP:
                    continue
                out = torch.empty_like(xs[0])
                try:
                    t = gtime(lambda: ru_fused(xs[0], c1.state[:B], fu.a1, fu.ib1, fu.w1, fu.b1, fu.a2, fu.ib2, fu.w2,
                                               fu.b2, fu.k1, fu.dil, out, bm=bm, bk=bk, bn2=bn2, nw=nw, ns=ns))
                except Exception:
                    continue
                if best is None or t < best[0]:
                    best = (t, (bm, bk, bn2, nw, ns))
            print(f"blk{bi} ru{ui} C={C} T={T} dil={c1.dil}: max|d| {err:.4f} (|ref| {scale:.2f}), state {st_err:.4f}; "
                  f"two-conv {t_ref:.3f} ms, fused best {best[0]:.3f} ms {best[1]}", flush=True)
print("SWEEP_DONE")
