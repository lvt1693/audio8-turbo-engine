#!/usr/bin/env python3
"""
FastEngine -- continuous-batching serving engine for Audio8-TTS-Realistic-0.6b (DualAR, Fish Audio S2 style).

The checkpoint (models/Audio8-TTS-Realistic-0.6b, its own modeling_arktts*.py) generates one 44.1 kHz codec frame
(2048 samples, ~21.5 a second) per step:
  * slow AR: 24 Qwen2-shaped layers (896 wide, GQA 14/2, qkv bias) over the packed text / audio sequence; its
    normalized last hidden gives the semantic token (= codebook 0, logits on the tied embedding) and feeds ...
  * fast AR: 4 layers over 10 positions -- [slow hidden, codebook 0, ..., codebook 8] -> codebooks 1..9;
  * codec decoder: codes -> 8-layer window-128 transformer at the frame rate -> 2x(transposed conv + ConvNeXt) -> a
    DAC-style causal conv decoder (x8 x8 x4 x2) -> 2048 samples.

The HF code runs that as a Python loop of single-row-shaped ops (RTF ~2 on a 3090). This engine keeps the
checkpoint's weights and numerics and re-implements the hot path:
  * every transformer (slow, fast, codec) on fused weights + Triton kernels (fast_kernels.py): fused residual +
    RMSNorm, qkv bias + RoPE + KV store, split-K decode attention over each row's own length, fused silu*mul. The
    checkpoint's interleaved RoPE runs as rotate_half on q / k rows permuted per head at load (q.k unchanged).
  * the semantic head reads only the 4096 semantic rows + EOS of the 155776-row tied embedding: the model's
    semantic logits processor masks every other token anyway (37x fewer head FLOPs).
  * fast AR: positions 0 and 1 (slow hidden, codebook 0) are both known before it starts, so they go in one
    2-token pass: 9 passes a frame instead of 10.
  * a STATEFUL streaming codec decoder: the codec is fully causal (causal convs, causal transposed convs, a
    128-frame window transformer), so carrying each layer's left context (conv tails, a 128-slot KV ring) reproduces
    the whole-utterance decode exactly while doing one frame of work per frame. The decoder convs are channels-last
    fp16 implicit-GEMM Triton kernels with the snake activation fused into their input.
  * one CUDA graph per bucket for the WHOLE tick: codec frame (side stream, one frame behind) -> fast AR (9 codebook
    draws) -> slow AR step -> semantic draw with RAS -> state update. The CPU only replays and reads back one token,
    one frame of codes and 2048 audio samples per row.
  * continuous batching over fixed slots, packed batched prefill, and per-voice caching of the prompt prefix KV
    (system turn + reference transcript + reference codes): a request prefills only its own text.

Sampling is the checkpoint's (ArkttsLegacyTopKTopPLogitsProcessor + _sample + RAS), in fp32: top-k, top-p on the
untempered full softmax, then the temperature; the semantic token is redrawn at (ras_top_p, ras_temperature) when it
repeats one of the row's last ras_window_size tokens.
"""
from __future__ import annotations

import os
import re
import time
import zlib
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from fast_kernels import (DecodeAttention, GemmDispatch, PrefillPlan, bias_rope_store, conv_cl, prefill_attention,
                          rmsnorm, ru_fused, silu_mul, snake_cat)


def _to_dev(t, dev):
    """Host tensor -> device without stalling the stream: a pageable copy blocks the host until all queued GPU work
    (the in-flight tick) is done; a pinned non_blocking one doesn't."""
    return t.pin_memory().to(dev, non_blocking=True)


def _permute_qk(w, nq, nkv, hd):
    """Rows of a fused [q | k | v] projection (weight [N, K] or bias [N]) with every q / k head's dims reordered from
    the interleaved RoPE pairs (2i, 2i+1) to rotate_half halves (i, i + hd/2)."""
    p = torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)]).to(w.device)
    rest = w.shape[1:]
    q = w[:nq * hd].reshape(nq, hd, *rest)[:, p].reshape(nq * hd, *rest)
    k = w[nq * hd:(nq + nkv) * hd].reshape(nkv, hd, *rest)[:, p].reshape(nkv * hd, *rest)
    return torch.cat([q, k, w[(nq + nkv) * hd:]], 0).contiguous()


def _rope_tables(base, hd, n, dev):
    """cos / sin [n, hd] for rotate_half, from the checkpoint's table: fp32 phases rounded to bf16 (_precompute_rope)."""
    fr = 1.0 / (base ** (torch.arange(0, hd, 2, device=dev).float()[: hd // 2] / hd))
    ph = torch.outer(torch.arange(n, device=dev).float(), fr)
    c, s = ph.cos().to(torch.bfloat16), ph.sin().to(torch.bfloat16)
    return torch.cat([c, c], -1).contiguous(), torch.cat([s, s], -1).contiguous()


def _clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


class _Stack:
    """A stack of Arktts transformer blocks (slow AR, fast AR or the codec's window transformer) on fused weights +
    Triton kernels. forward(x, pos, slot, lens) processes R independent token rows: row r is at RoPE position pos[r],
    its K/V is stored into cache row slot[r] at column pos[r] % ring, and it attends columns [0, lens[r]) of that cache
    row. Returns the final-normed hidden. `x` is consumed (the residual stream is updated in place)."""

    def __init__(self, layers, norm, nq, nkv, hd, eps, rope_base, max_rows, cache_rows, cache_len, max_pos, dev, dt,
                 nsplit_max=16, mm=None, ring=None, layer_scale=False):
        self.mm = mm if mm is not None else (lambda x, w: torch.matmul(x, w.t()))
        self.nq, self.nkv, self.hd, self.eps = nq, nkv, hd, float(eps)
        self.layers = []
        for L in layers:
            a, f = L.attention, L.feed_forward
            wo, w2 = a.wo.weight.data, f.w2.weight.data
            if layer_scale:          # the codec's LayerScale folded into the output projections
                wo = (wo.float() * L.attention_layer_scale.gamma.data.float()[:, None]).to(dt)
                w2 = (w2.float() * L.ffn_layer_scale.gamma.data.float()[:, None]).to(dt)
            self.layers.append(SimpleNamespace(
                wqkv=_permute_qk(a.wqkv.weight.data, nq, nkv, hd).to(dt),
                bqkv=_permute_qk(a.wqkv.bias.data, nq, nkv, hd).to(dt) if a.wqkv.bias is not None else None,
                wo=wo.to(dt).contiguous(),
                w13=torch.cat([f.w1.weight.data, f.w3.weight.data], 0).to(dt).contiguous(),
                w2=w2.to(dt).contiguous(),
                ln1=L.attention_norm.weight.data.to(dt), ln2=L.ffn_norm.weight.data.to(dt)))
        self.norm = norm.weight.data.to(dt)
        self.cos, self.sin = _rope_tables(rope_base, hd, max_pos, dev)
        self.ring = int(ring) if ring else 1 << 30
        # all layers' caches in one tensor each ([layers, rows, nkv, len, hd]; kc[i] / vc[i] are views): a prompt
        # prefix goes into a row with one copy
        self.kc_all = torch.zeros(len(self.layers), cache_rows, nkv, cache_len, hd, dtype=dt, device=dev)
        self.vc_all = torch.zeros_like(self.kc_all)
        self.kc = list(self.kc_all.unbind(0))
        self.vc = list(self.vc_all.unbind(0))
        self.attn = DecodeAttention(max_rows, nq, nkv, hd, nsplit_max, dev)

    def forward(self, x, pos, slot, lens, nsplit=1, rowmap=True, plan=None):
        """plan: a fast_kernels.PrefillPlan -> packed causal prefill attention (lens unused)."""
        R = x.shape[0]
        nl = len(self.layers)
        h = rmsnorm(x, self.layers[0].ln1, self.eps)
        res = x
        q = torch.empty(R, self.nq, self.hd, dtype=x.dtype, device=x.device)
        o = torch.empty_like(q)
        for i, L in enumerate(self.layers):
            qkv = self.mm(h, L.wqkv)
            bias_rope_store(qkv, L.bqkv, self.cos, self.sin, pos, slot, q, self.kc[i], self.vc[i],
                            self.nq, self.nkv, self.hd, ring=self.ring)
            if plan is not None:
                prefill_attention(q, self.kc[i], self.vc[i], plan, o, self.nq // self.nkv)
            else:
                self.attn(q, self.kc[i], self.vc[i], lens, o, nsplit=nsplit, rowmap=slot if rowmap else None)
            a = self.mm(o.view(R, -1), L.wo)
            h, res = rmsnorm(a, L.ln2, self.eps, residual=res)
            m = self.mm(silu_mul(self.mm(h, L.w13)), L.w2)
            nxt = self.layers[i + 1].ln1 if i + 1 < nl else self.norm
            h, res = rmsnorm(m, nxt, self.eps, residual=res)
        return h


class _Sampler:
    """The checkpoint's sampling, graph-safe, in fp32. filt(): top-k (sorted) and the cumulative probability of each
    candidate under the UNTEMPERED full softmax; draw(): drops the candidates whose cumulative probability (itself
    included) passes p -- the best one always stays --, applies the temperature and draws (exponential race: argmax
    p/E, E~Exp(1), draws from p exactly). Same order as ArkttsLegacyTopKTopPLogitsProcessor + ArkttsModel._sample."""

    def __init__(self, do_sample, temperature, top_k, top_p, vocab):
        self.do_sample = bool(do_sample)
        self.t = max(float(temperature or 1.0), 1e-5)
        self.k = min(int(top_k) if top_k and int(top_k) > 0 else int(vocab), int(vocab))
        self.p = float(top_p) if top_p is not None else 1.0

    def filt(self, logits):
        v, idx = torch.topk(logits, self.k, dim=-1)
        cum = torch.exp(v - torch.logsumexp(logits, -1, keepdim=True)).cumsum(-1)
        return v, idx, cum

    @staticmethod
    def draw(v, idx, cum, p, t):
        if p < 1.0:
            drop = cum > p
            drop[:, 0] = False
            v = v.masked_fill(drop, float("-inf"))
        pr = torch.softmax(v / t, -1)
        c = (pr / torch.empty_like(pr).exponential_()).argmax(-1, keepdim=True)
        return idx.gather(-1, c).squeeze(-1)

    def __call__(self, logits):
        if not self.do_sample:
            return logits.argmax(-1)
        v, idx, cum = self.filt(logits)
        return self.draw(v, idx, cum, self.p, self.t)


# =============================================================================================
# Stateful streaming codec decoder
# =============================================================================================
class _CLConv:
    """Packed weights + state for one conv of the channels-last Triton decoder path. snake_mod: the ArkttsSnake1d in
    front of it (x + 1/(alpha+1e-9) * sin(alpha x)^2), fused into the conv's input copy."""

    def __init__(self, m, Bmax, dev, states, snake_mod=None, trans=False):
        f16 = torch.float16
        c = m.conv
        w = c.weight.detach().float()
        if trans:                                      # ConvTranspose1d [Cin, Cout, 2s], right crop s (causal)
            Cin, Cout, K = w.shape
            s = c.stride[0]
            assert K == 2 * s
            W = torch.zeros(s, 2, Cin, Cout, device=dev)
            for p in range(s):
                W[p, 0] = w[:, :, p + s]               # tap 0 reads the previous input frame
                W[p, 1] = w[:, :, p]                   # tap 1 reads the current one
            self.taps, self.dil, self.ph, self.ctx = 2, 1, s, 1
        else:                                          # causal Conv1d [Cout, Cin, K]
            Cout, Cin, K = w.shape
            assert c.stride[0] == 1 and c.groups == 1
            self.taps, self.dil, self.ph = K, c.dilation[0], 1
            self.ctx = (K - 1) * self.dil
            W = w.permute(2, 1, 0).unsqueeze(0)        # [1, K, Cin, Cout]
        cp = max(16, Cout)
        Wp = torch.zeros(*W.shape[:-1], cp, device=dev)
        Wp[..., :Cout] = W
        self.W = Wp.to(f16).contiguous()
        self.bias = c.bias.detach().to(f16).contiguous() if c.bias is not None else None
        self.Cin, self.Cout = Cin, Cout
        self.cfg = None                                # conv_cl tile config (StreamingVocoder.tune)
        self.a = self.ib = None
        if snake_mod is not None:
            al = snake_mod.alpha.detach().float().reshape(-1)
            self.a = al.contiguous()
            self.ib = (1.0 / (al + 1e-9)).contiguous()
        self.state = None
        if self.ctx > 0:
            self.state = torch.zeros(Bmax, self.ctx, Cin, dtype=f16, device=dev)
            states.append(self.state)

    def __call__(self, x, b, res=None):
        """x: [b, T, Cin] fp16 channels-last (pre-activation) -> [b, T*ph, Cout] fp16."""
        T = x.shape[1]
        if self.state is None and self.a is not None:     # no context: the snake goes into the conv's loads
            y = torch.empty(b, T * self.ph, self.Cout, dtype=torch.float16, device=x.device)
            conv_cl(x.contiguous(), self.W, self.bias, T, self.taps, self.dil, self.ph, y, res=res, cfg=self.cfg,
                    snake=(self.a, self.ib))
            return y
        xin = torch.empty(b, self.ctx + T, self.Cin, dtype=torch.float16, device=x.device)
        snake_cat(x, self.state[:b] if self.state is not None else None, self.a, self.ib, xin)
        y = torch.empty(b, T * self.ph, self.Cout, dtype=torch.float16, device=x.device)
        conv_cl(xin, self.W, self.bias, T, self.taps, self.dil, self.ph, y, res=res, cfg=self.cfg)
        if self.state is not None:
            self.state[:b].copy_(xin[:, T:])
        return y


class _FusedRU:
    """A residual unit (snake, causal conv k dil, snake, conv 1x1, + input) as ONE kernel (fast_kernels.ru_fused): the
    conv_k output stays in registers, the activations are applied on the loads. Uses c1's state (activated context) and
    refreshes it after. For chunks at least as long as the context (the decoder's later, high-rate blocks)."""

    # (BM, BK, BN2, warps, stages) per channel count, the best of a sweep on an RTX 3090 (work/ru_sweep.py)
    CFG = {96: (32, 32, 64, 8, 1), 192: (32, 32, 64, 4, 2), 384: (32, 32, 64, 8, 2)}

    def __init__(self, c1, c2, dev):
        assert c1.Cin == c1.Cout == c2.Cin == c2.Cout and c2.taps == 1 and c1.ph == c2.ph == 1
        C = c1.Cin
        CP = 1 << (C - 1).bit_length()
        f16, f32 = torch.float16, torch.float32

        def pad(v, n=CP):
            o = torch.zeros(n, dtype=f32, device=dev)
            if v is not None:
                o[:v.numel()] = v.float().reshape(-1)
            return o.contiguous()
        self.c1, self.C, self.CP, self.k1, self.dil = c1, C, CP, c1.taps, c1.dil
        W1 = torch.zeros(c1.taps, C, CP, dtype=f16, device=dev)
        W1[:, :, :C] = c1.W[0, :, :, :C]
        W2 = torch.zeros(CP, CP, dtype=f16, device=dev)
        W2[:C, :C] = c2.W[0, 0, :, :C]
        self.w1, self.w2 = W1.contiguous(), W2.contiguous()
        self.b1, self.b2 = pad(c1.bias), pad(c2.bias)
        self.a1, self.ib1 = c1.a, c1.ib
        self.a2, self.ib2 = pad(c2.a), pad(c2.ib)
        self.cfg = self.CFG.get(C, (32, 32, 64, 4, 2))

    def __call__(self, x, b):
        T = x.shape[1]
        out = torch.empty_like(x)
        bm, bk, bn2, nw, ns = self.cfg
        ru_fused(x, self.c1.state[:b], self.a1, self.ib1, self.w1, self.b1, self.a2, self.ib2, self.w2, self.b2,
                 self.k1, self.dil, out, bm=bm, bk=bk, bn2=bn2, nw=nw, ns=ns)
        ctx = self.c1.ctx
        snake_cat(x[:, T - ctx:], None, self.a1, self.ib1, self.c1.state[:b])   # the next chunk's context
        return out


class _CLDecoder:
    """ArkttsDecoder (conv_in, 4 x [snake, causal transposed conv, 3 residual units], snake, conv_out, tanh) in
    channels-last fp16 with the implicit-GEMM Triton conv (fp16 MMA, fp32 block accumulation); residual units on
    chunks at least as long as their context run fused (_FusedRU, TTS_VOC_FUSED_RU=0: never)."""

    def __init__(self, dec, Bmax, dev, states):
        mods = list(dec.model)
        assert len(mods) == 8 and isinstance(mods[-1], torch.nn.Tanh)
        # off by default: on an RTX 3090 at 128 rows the fused unit measured SLOWER than the two convs (C=192: 3.5 vs
        # 1.6 ms, C=96: 1.5 vs 1.2 ms; work/ru_sweep.py): a full-width [rows, C] tile forces small row tiles, so the
        # 7-tap weights are re-read from L2 twice as often. Kept for GPUs with more registers / shared memory.
        fuse = {int(c) for c in os.environ.get("TTS_VOC_FUSED_RU", "").split(",") if c.strip().isdigit()}
        self.dec_in = _CLConv(mods[0], Bmax, dev, states)
        self.blocks = []
        for blk in mods[1:5]:
            bm = list(blk.block)
            tc = _CLConv(bm[1], Bmax, dev, states, snake_mod=bm[0], trans=True)
            units = []
            for ru in bm[2:]:
                c1 = _CLConv(ru.block[1], Bmax, dev, states, snake_mod=ru.block[0])
                c2 = _CLConv(ru.block[3], Bmax, dev, states, snake_mod=ru.block[2])
                units.append((c1, c2, _FusedRU(c1, c2, dev) if c1.Cin in fuse else None))
            self.blocks.append((tc, units))
        self.final = _CLConv(mods[6], Bmax, dev, states, snake_mod=mods[5])

    def __call__(self, x, b):
        """x: [b, T, 1024] -> wav [b, T*up] float32 (tanh)."""
        x = self.dec_in(x.to(torch.float16).contiguous(), b)
        for tc, units in self.blocks:
            x = tc(x, b)
            for c1, c2, fu in units:
                x = fu(x, b) if fu is not None and x.shape[1] >= c1.ctx else c2(c1(x, b), b, res=x)
        y = self.final(x, b)                                    # [b, T*up, 1]
        return y[..., 0].float().tanh()


class StreamingVocoder:
    """Frame-at-a-time, batched, stateful ArkttsCodec.decode. Row state is independent per slot; load_many(slots,
    snapshot) primes rows (e.g. with a voice's reference)."""

    WINDOW = 128

    def __init__(self, codec, Bmax, dev, max_pos=8192, mm=None):
        """codec: the ArkttsCodec in float32 (tables and weights are taken from it)."""
        self.dev, self.Bmax = dev, Bmax
        self.dt = torch.bfloat16
        self.states: List[torch.Tensor] = []
        q = codec.quantizer
        self.up = 4 * 512                                   # 2 x2 upsample stages, then the decoder's x512

        def table(vq):            # codebook -> its 1x1 out_proj (bias included): [V, 1024]
            with torch.no_grad():
                return vq.out_proj(vq.codebook.weight.detach().float().t().unsqueeze(0))[0].t().contiguous()

        self.t_sem = table(q.semantic_quantizer.quantizers[0])                     # [4096, 1024]
        self.t_res = torch.stack([table(v) for v in q.quantizer.quantizers])        # [9, 1024, 1024]
        self.v_sem, self.v_res = self.t_sem.shape[0], self.t_res.shape[1]
        self.ar_res = torch.arange(self.t_res.shape[0], device=dev)
        pm = q.post_module
        assert isinstance(pm.input_proj, torch.nn.Identity) and isinstance(pm.output_proj, torch.nn.Identity)
        a0 = pm.layers[0].attention
        self.post = _Stack(pm.layers, pm.norm, a0.n_head, a0.n_local_heads, a0.head_dim, pm.layers[0].attention_norm.eps,
                           pm.rope_base, max_rows=Bmax, cache_rows=Bmax, cache_len=self.WINDOW, max_pos=max_pos,
                           dev=dev, dt=self.dt, nsplit_max=1, mm=mm, ring=self.WINDOW, layer_scale=True)
        for k, v in zip(self.post.kc, self.post.vc):
            self.states += [k, v]
        self.pos = torch.zeros(Bmax, dtype=torch.int32, device=dev)            # frames decoded per row
        self.max_pos = max_pos
        self.ar = torch.arange(Bmax, dtype=torch.int32, device=dev)
        # upsample stages ([transposed conv k=s=2, ConvNeXt] x2), channels-last: the transposed conv is a plain GEMM,
        # the ConvNeXt dwconv a 7-tap causal depthwise filter over a [B, 6, C] tail
        self.ups = []
        for blocks in q.upsample:
            tc, cnx = blocks[0], blocks[1]
            w = tc.conv.weight.detach()                                         # [Cin, Cout, 2]
            assert w.shape[-1] == tc.conv.stride[0] == 2
            dw = cnx.dwconv
            st = torch.zeros(Bmax, dw.kernel_size - 1, w.shape[1], dtype=self.dt, device=dev)
            self.states.append(st)
            self.ups.append(SimpleNamespace(
                tw=torch.cat([w[:, :, 0].t(), w[:, :, 1].t()], 0).to(self.dt).contiguous(),   # [2*Cout, Cin]
                tb=torch.cat([tc.conv.bias.detach(), tc.conv.bias.detach()]).to(self.dt).contiguous(),
                dww=dw.conv.weight.detach()[:, 0, :].t().to(self.dt).contiguous(), dwb=dw.conv.bias.detach().to(self.dt),
                st=st, k=dw.kernel_size,
                ln_w=cnx.norm.weight.detach().to(self.dt), ln_b=cnx.norm.bias.detach().to(self.dt), ln_eps=cnx.norm.eps,
                w1=cnx.pwconv1.weight.detach().to(self.dt), b1=cnx.pwconv1.bias.detach().to(self.dt),
                w2=cnx.pwconv2.weight.detach().to(self.dt), b2=cnx.pwconv2.bias.detach().to(self.dt),
                gamma=cnx.gamma.detach().to(self.dt)))
        self.cl = _CLDecoder(codec.decoder, Bmax, dev, self.states)
        self.states.append(self.pos)
        self._arena_init()

    def _arena_init(self):
        """Every state tensor becomes a view into ONE byte buffer [Bmax, bytes a row] (16-byte aligned pieces), so a
        row's whole state is cleared / loaded / snapshotted by one kernel instead of ~60 (an admission's cost)."""
        sizes = [s[0].numel() * s.element_size() for s in self.states]
        offs, tot = [], 0
        for p in sizes:
            offs.append(tot)
            tot += (p + 15) // 16 * 16
        self.arena = torch.zeros(self.Bmax, tot, dtype=torch.uint8, device=self.dev)
        new = {id(s): self.arena[:, o:o + p].view(s.dtype).view(s.shape) for s, o, p in zip(self.states, offs, sizes)}
        for c in [self.cl.dec_in, self.cl.final] + [u for tc, units in self.cl.blocks for u in
                                                     [tc] + [c for c1c2 in units for c in c1c2[:2]]]:
            if c.state is not None:
                c.state = new[id(c.state)]
        for u in self.ups:
            u.st = new[id(u.st)]
        self.post.kc = [new[id(k)] for k in self.post.kc]
        self.post.vc = [new[id(v)] for v in self.post.vc]
        self.pos = new[id(self.pos)]
        self.states = [new[id(s)] for s in self.states]

    # ---- conv tile tuning ----
    CONV_CANDIDATES = [(64, 128, 64, 4, 3), (128, 128, 32, 8, 3), (128, 64, 64, 4, 3), (128, 128, 64, 8, 2),
                       (64, 64, 64, 4, 3), (128, 64, 32, 4, 4), (64, 128, 32, 4, 4), (32, 128, 64, 4, 3),
                       (128, 32, 64, 4, 3), (64, 32, 64, 4, 3), (256, 64, 32, 8, 3), (128, 128, 32, 4, 3)]

    def convs(self):
        """(conv, its input frames a tick, whether it adds a residual) of every decoder conv, in order."""
        out, T = [(self.cl.dec_in, 4, False)], 4
        for tc, units in self.cl.blocks:
            out.append((tc, T, False))
            T *= tc.ph
            for c1, c2, _ in units:
                out += [(c1, T, False), (c2, T, True)]
        out.append((self.cl.final, T, False))
        return out

    @torch.inference_mode()
    def tune(self, b, log=print, cache_path=None):
        """Pick each decoder conv's conv_cl tile config at b rows (the largest bucket: where the time goes), against
        the heuristic default; cached per GPU name in cache_path (JSON)."""
        import json
        import triton
        gpu = torch.cuda.get_device_name()
        allc = {}
        if cache_path and os.path.exists(cache_path):
            try:
                allc = json.load(open(cache_path))
            except Exception:
                allc = {}
        disk = allc.setdefault(gpu, {})
        f16, n_new, gain = torch.float16, 0, []
        for c, T, has_res in self.convs():
            key = f"conv{c.Cin},{c.Cout},{c.taps},{c.dil},{c.ph},{b * T}"
            if key in disk:
                c.cfg = tuple(disk[key]) if disk[key] else None
                continue
            xin = torch.randn(b, c.ctx + T, c.Cin, device=self.dev, dtype=f16) * 0.5
            y = torch.empty(b, T * c.ph, c.Cout, device=self.dev, dtype=f16)
            res = torch.randn_like(y) if has_res else None
            run = lambda cfg: conv_cl(xin, c.W, c.bias, T, c.taps, c.dil, c.ph, y, res=res, cfg=cfg)
            best, t0 = None, GemmDispatch._time(lambda: run(None))
            best_t = t0
            for cfg in self.CONV_CANDIDATES:
                if cfg[1] > max(16, triton.next_power_of_2(c.Cout)) or cfg[2] > max(16, triton.next_power_of_2(c.Cin)):
                    continue
                try:
                    t = GemmDispatch._time(lambda: run(cfg))
                except Exception:
                    continue
                if t < best_t * 0.97:
                    best, best_t = cfg, t
            c.cfg = best
            disk[key] = list(best) if best else None
            n_new += 1
            gain.append((t0, best_t))
        if n_new:
            log(f"[voc] tuned {n_new} decoder convs at {b} rows: {sum(a for a, _ in gain):.2f} -> "
                f"{sum(t for _, t in gain):.2f} ms")
            if cache_path:
                tmp = cache_path + ".tmp"
                json.dump(allc, open(tmp, "w"))
                os.replace(tmp, cache_path)

    # ---- per-row state management (the arena: one row = every state of that row) ----
    def snapshot(self, row=0):
        return [self.arena[row:row + 1].clone()]

    def load_many(self, slots, snap):
        """Load one snapshot into several rows (one index_copy). The snapshot may live in host memory (pinned): it is
        copied over on the way."""
        idx = _to_dev(torch.tensor(slots, dtype=torch.long), self.dev)
        v = snap[0].to(self.dev, non_blocking=True)
        self.arena.index_copy_(0, idx, v.expand(len(slots), -1))

    def clear(self, slot):
        self.arena[slot:slot + 1].zero_()

    def clear_many(self, slots):
        self.arena.index_fill_(0, _to_dev(torch.tensor(slots, dtype=torch.long), self.dev), 0)

    # ---- one frame for rows [0, b) ----
    def step(self, codes, b):
        """codes: [b, 10] long (the LLM's codes; clamped here like ArkttsDownsampleQuantizer.decode) -> [b, 2048] f32."""
        c0 = codes[:, 0].clamp(0, self.v_sem - 1)
        cr = codes[:, 1:].clamp(0, self.v_res - 1)
        x = self.t_sem[c0] + self.t_res[self.ar_res, cr].sum(1)                 # [b, 1024] fp32
        pos = self.pos[:b]
        h = self.post.forward(x.to(self.dt), pos.clamp(max=self.max_pos - 1), self.ar[:b],
                              (pos + 1).clamp(max=self.WINDOW), nsplit=1, rowmap=False)   # [b, 1024]
        x = h.unsqueeze(1)                                                       # [b, 1, C] channels-last
        for u in self.ups:
            T = x.shape[1]
            x = F.linear(x, u.tw, u.tb).view(b, 2 * T, -1)                       # out[2j+p] = x[j] @ W[..., p]
            T = 2 * T
            r = x
            xin = torch.cat([u.st[:b], x], 1)                                    # [b, k-1+T, C]
            y = u.dwb + xin[:, 0:T] * u.dww[0]
            for j in range(1, u.k):
                y = y + xin[:, j:j + T] * u.dww[j]
            u.st[:b].copy_(xin[:, T:])
            y = F.layer_norm(y, (y.shape[-1],), u.ln_w, u.ln_b, u.ln_eps)
            y = F.linear(F.gelu(F.linear(y, u.w1, u.b1)), u.w2, u.b2)
            x = r + u.gamma * y
        wav = self.cl(x, b)                                                      # [b, 2048]
        self.pos[:b] += 1
        return wav


def ref_fp(item) -> int:
    """A fingerprint of a voice's reference codes: its vocoder snapshot is made from them (SnapStore)."""
    rc = getattr(item, "ref_code", None)
    return zlib.crc32(rc.detach().cpu().numpy().astype(np.int32).tobytes()) if rc is not None else 0


class SnapStore:
    """The voices' vocoder snapshots (prime_speakers): name -> ([state tensors], last ref frame).

    With a `path` every snapshot is also a file, <path>/<name>.<fingerprint of its reference codes>.snap, made once and
    kept across restarts (a voice rebuilt under its name has another fingerprint, so another file); host memory holds
    the `resident` ones (the pinned families) plus the `ram` most recently used others, and a miss is read back from
    disk (~2.5 MB, a few ms, on the voice's first line after a while). Without a path all of them stay in memory."""

    def __init__(self, path=None, ram=2048):
        self.path, self.ram = path, ram
        self.mem: "OrderedDict[str, tuple]" = OrderedDict()
        self.fp: Dict[str, int] = {}
        self.resident: set = set()
        self.loads = 0                       # read back from disk (a miss)
        if path:
            os.makedirs(path, exist_ok=True)

    def _file(self, key):
        return os.path.join(self.path, f"{key}.{self.fp[key]:08x}.snap")

    def set_fp(self, key, fp):
        if self.fp.get(key, fp) != fp:          # rebuilt from another reference: the one in memory is stale
            self.mem.pop(key, None)
        self.fp[key] = fp

    def __contains__(self, key):
        return key in self.mem or bool(self.path and key in self.fp and os.path.exists(self._file(key)))

    def get(self, key, default=None):
        v = self.mem.get(key)
        if v is not None:
            self.mem.move_to_end(key)
            return v
        if not self.path or key not in self.fp or not os.path.exists(self._file(key)):
            return default
        states, last = torch.load(self._file(key), map_location="cpu", weights_only=True)
        v = ([t.pin_memory() if torch.cuda.is_available() else t for t in states], last)
        self.loads += 1
        self._keep(key, v)
        return v

    def __setitem__(self, key, v):
        if self.path and key in self.fp:
            f = self._file(key)
            torch.save(([t.cpu() for t in v[0]], v[1].cpu()), f + ".part")
            os.replace(f + ".part", f)
        self._keep(key, v)

    def _keep(self, key, v):
        self.mem[key] = v
        self.mem.move_to_end(key)
        while self.path and len(self.mem) > self.ram + len(self.resident):
            old = next((k for k in self.mem if k not in self.resident), None)
            if old is None:
                break
            del self.mem[old]

    def values(self):
        return self.mem.values()

    def on_disk(self) -> int:
        return len(os.listdir(self.path)) if self.path else 0


# =============================================================================================
# Engine
# =============================================================================================
class FastEngine:
    def __init__(self, model, tokenizer, gen_params: dict, max_slots: int = 32, max_seq: int = 2048, buckets=None,
                 seq_from_env=True):
        """model: the HF ArkttsModel (bf16, on its GPU); tokenizer: its tokenizer (ArkttsProcessor.tokenizer).
        seq_from_env: TTS_MAX_SEQ, when set, overrides max_seq (capped at the checkpoint's 2048 positions)."""
        self.m, self.tokz = model, tokenizer
        c = self.cfg = model.config
        self.dev = model.device
        self.dt = torch.bfloat16
        self.B = int(max_slots)
        S = int(os.environ.get("TTS_MAX_SEQ", str(max_seq))) if seq_from_env else int(max_seq)
        self.S = min(S, int(c.max_seq_len))
        gp = dict(gen_params or {})
        self.gp = gp
        self.eos, self.sb, self.se = int(c.eos_token_id), int(c.semantic_begin_id), int(c.semantic_end_id)
        self.ncg, self.cbk = int(c.num_codebooks), int(c.codebook_size)
        self.nsem = self.se - self.sb + 1                    # 4096 semantic tokens; head row nsem = EOS
        self.samp = _Sampler(gp.get("do_sample", True), gp.get("temperature", 0.8), gp.get("top_k", 50),
                             gp.get("top_p", 0.95), self.nsem + 1)
        self.cb_samp = _Sampler(gp.get("do_sample", True), gp.get("temperature", 0.8), gp.get("top_k", 50),
                                gp.get("top_p", 0.95), self.cbk)
        self.ras_p, self.ras_t, self.ras_w = float(c.ras_top_p), max(float(c.ras_temperature), 1e-5), int(c.ras_window_size)
        self.max_new = int(gp.get("max_new_tokens", 2048))
        B, dev, dt, H = self.B, self.dev, self.dt, int(c.dim)
        self.H = H
        assert int(c.fast_dim) == H and bool(c.norm_fastlayer_input), "fast_project_in / unnormed fast input: not supported"

        # GEMMs go through a dispatcher that picks cuBLAS or the Triton linear per shape, tuned on this GPU in
        # capture(). TTS_TRITON_GEMM=0 -> always cuBLAS.
        self.mm = GemmDispatch()
        self.mm.enabled = os.environ.get("TTS_TRITON_GEMM", "1") not in ("0", "false", "no")
        # prefill packs every admitted prompt token as its own attention row -> cap per prefill call
        self.prefill_tokens = int(os.environ.get("TTS_PREFILL_TOKENS", "8192"))
        # cache row B is scratch: the prefill graphs' padding rows write there
        self.SCR = B
        self.slow = _Stack(model.layers, model.norm, c.n_head, c.n_local_heads, c.head_dim, c.norm_eps, c.rope_base,
                           max_rows=max(16 * B, self.prefill_tokens), cache_rows=B + 1, cache_len=self.S,
                           max_pos=self.S, dev=dev, dt=dt, mm=self.mm)
        self.fast = _Stack(model.fast_layers, model.fast_norm, c.fast_n_head, c.fast_n_local_heads, c.fast_head_dim,
                           c.norm_eps, c.rope_base, max_rows=max(2 * B, 16), cache_rows=B, cache_len=self.ncg,
                           max_pos=self.ncg, dev=dev, dt=dt, nsplit_max=1, mm=self.mm)
        self.emb = model.embeddings.weight.data                                   # [V, H] (tied head)
        self.cbemb = model.codebook_embeddings.weight.data                        # [10 * 4096, H]
        self.cb_off = (torch.arange(self.ncg, device=dev) * self.cbk).view(1, -1)
        # semantic head: the rows of the tied embedding the semantic logits processor keeps (4096 semantic + EOS),
        # padded to a multiple of 64 rows (the pad columns are sliced off)
        nh = self.nsem + 1
        self.nh = nh
        W = torch.zeros((nh + 63) // 64 * 64, H, dtype=dt, device=dev)
        W[:self.nsem] = self.emb[self.sb:self.se + 1]
        W[self.nsem] = self.emb[self.eos]
        self.head = W.contiguous()
        self.fast_emb = model.fast_embeddings.weight.data                         # [4096, H]
        self.fast_out = model.fast_output.weight.data                             # [4096, H]

        # --- vocoder: built from an fp32 copy of the codec, which then goes back to the model's dtype (encode) ---
        # (in inference mode: the codec's weights are inference tensors once it has run under it)
        with torch.inference_mode():
            codec = model.load_codec(device=dev, dtype=torch.float32)
            self.sr = int(codec.sample_rate)
            self.voc = StreamingVocoder(codec, B, dev, max_pos=int(os.environ.get("TTS_VOC_MAX_POS", "8192")),
                                        mm=self.mm)
            model.load_codec(device=dev, dtype=dt)
        self.up = self.voc.up
        self.fps = self.sr / self.up
        # codec frame of pure silence: replaces the (garbage) EOS frame in the tick so the vocoder of a row whose
        # segment just ended keeps a clean state for a code-switch continuation (see admit)
        with torch.inference_mode():
            codes, _ = model.encode_audio(torch.zeros(1, 1, self.sr, device=dev))
        self.sil_frame = codes[0, :, codes.shape[-1] // 2].long().clamp(min=0)
        # TTS_VOC_PRIME=0 (default): a row's vocoder starts clean, as the checkpoint's decode_audio (which decodes the
        # generated codes alone); its first tick decodes a silence frame, discarded. 1: from its voice's reference
        # decoded (as if the reference and the line were one recording), every voice primed at startup and its
        # snapshot kept on disk (SnapStore, ~2.4 MB a voice)
        self.voc_prime = os.environ.get("TTS_VOC_PRIME", "0").strip().lower() not in ("0", "false", "no", "off")
        self.prime_frames = max(2, int(os.environ.get("TTS_VOC_PRIME_FRAMES", "160")))

        # --- persistent per-slot state (graph inputs/outputs; fixed addresses) ---
        # (B + 1 rows: row B = SCR, the prefill graphs' padding)
        self.tok = torch.zeros(B + 1, dtype=torch.long, device=dev)              # next semantic token (or EOS)
        self.hid = torch.zeros(B + 1, H, dtype=dt, device=dev)                   # its slow hidden (normed)
        self.cpos = torch.zeros(B + 1, dtype=torch.int32, device=dev)
        self.prev = torch.zeros(B + 1, self.ras_w, dtype=torch.long, device=dev)  # RAS window
        self.nstep = torch.zeros(B + 1, dtype=torch.int32, device=dev)
        self.tok_in = torch.zeros(B, dtype=torch.long, device=dev)
        self.audio = torch.zeros(B, self.up, dtype=torch.float32, device=dev)
        self.frame_prev = torch.zeros(B, self.ncg, dtype=torch.long, device=dev)
        self.side = torch.cuda.Stream(device=dev)
        self.ar = torch.arange(B, dtype=torch.int32, device=dev)
        self.f_pos0 = torch.arange(2 * B, dtype=torch.int32, device=dev) % 2
        self.f_slot0 = torch.arange(2 * B, dtype=torch.int32, device=dev) // 2
        self.f_len0 = self.f_pos0 + 1
        self.f_pos = [torch.full((B,), i, dtype=torch.int32, device=dev) for i in range(self.ncg)]
        self.f_len = [p + 1 for p in self.f_pos]

        env_b = os.environ.get("TTS_BATCH_BUCKETS", "").strip()
        if env_b:
            buckets = [int(x) for x in env_b.split(",") if x.strip()]
        if not buckets:
            buckets = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
                       144, 160, 176, 192]
        self.buckets = sorted({b for b in buckets if 1 <= b <= B} | {B})
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = None
        self.nsplit = {}
        # prefill graphs (TTS_PREFILL_GRAPHS = token buckets, "0" = off): a prefill of up to PF_ROWS requests packed in
        # nt tokens replays one graph instead of ~500 kernel launches (the admission's CPU and the GPU's gaps)
        env_pf = os.environ.get("TTS_PREFILL_GRAPHS", "64,128,256,512,1024,2048").strip()
        self.pf_buckets = sorted({int(x) for x in env_pf.split(",") if x.strip().isdigit() and int(x) > 0}) \
            if env_pf not in ("", "0") else []
        self.pf_graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.pf_buckets:
            NT, NR = self.pf_buckets[-1], max(1, int(os.environ.get("TTS_PREFILL_GRAPH_ROWS", "64")))
            NB = NT // PrefillPlan.BM + NR
            self.pf_NT, self.pf_NR, self.pf_NB = NT, NR, NB
            self.pf_x = torch.zeros(NT, H, dtype=dt, device=dev)
            # int32 metadata, one buffer: pos, slot [NT] | q0, n, bslot, pos0 [NB] | last, st, cpos [NR]
            self.pf_meta = torch.zeros(2 * NT + 4 * NB + 3 * NR, dtype=torch.int32, device=dev)
            o = np.cumsum([0, NT, NT, NB, NB, NB, NB, NR, NR, NR])
            self.pf_v = SimpleNamespace(**{k: self.pf_meta[o[i]:o[i + 1]] for i, k in enumerate(
                ("pos", "slot", "q0", "n", "bslot", "pos0", "last", "st", "cpos"))})
            self.pf_off = o
        # prompt caches
        self._parts = None
        self._spk_cache: Dict[str, tuple] = {}      # voice -> (prefix semantic ids, prefix codes) on the GPU
        self._pkv: Dict[str, list] = {}             # voice -> prefix KV per slow layer
        self.voc_snaps = SnapStore()                # memory only; engine_worker gives it a disk cache
        # no guidance / director on this model (engine_worker calls these unconditionally)
        self.dir = None
        self.guide = False

    # ------------------------------------------------------------------ the tick
    def _fast_codes(self, hid, code0, b):
        """Fast AR: [slow hidden, codebook 0] (positions 0 and 1, one 2-token pass) -> codebook 1, then one pass per
        codebook -> codebooks 2..9. Returns [b, 10]."""
        x = torch.stack([hid, self.fast_emb[code0]], 1).view(2 * b, self.H)
        h = self.fast.forward(x, self.f_pos0[:2 * b], self.f_slot0[:2 * b], self.f_len0[:2 * b])
        c = self.cb_samp(self.mm(h.view(b, 2, self.H)[:, 1], self.fast_out).float())
        cs = [code0, c]
        for i in range(2, self.ncg):
            h = self.fast.forward(self.fast_emb[c], self.f_pos[i][:b], self.ar[:b], self.f_len[i][:b], rowmap=False)
            c = self.cb_samp(self.mm(h, self.fast_out).float())
            cs.append(c)
        return torch.stack(cs, 1)

    def _embed(self, sem, codes):
        """Slow AR input of audio positions: embed(semantic) + the 10 codebook embeddings (ArkttsModel._embed)."""
        return self.emb[sem] + self.cbemb[codes + self.cb_off].sum(1)

    def _head(self, h):
        """Semantic + EOS logits (fp32) of normed slow hiddens [n, H] -> [n, 4097]."""
        return self.mm(h, self.head)[:, :self.nh].float()

    def _to_token(self, j):
        return torch.where(j == self.nsem, self.eos, j + self.sb)

    def _sample_sem(self, logits, b):
        """The semantic token: the regular draw, redrawn at (ras_top_p, ras_temperature) when it repeats one of the
        row's last ras_window tokens (ArkttsModel._sample_semantic)."""
        if not self.samp.do_sample:
            return self._to_token(logits.argmax(-1))
        v, idx, cum = self.samp.filt(logits)
        nrm = self._to_token(self.samp.draw(v, idx, cum, self.samp.p, self.samp.t))
        high = self._to_token(self.samp.draw(v, idx, cum, self.ras_p, self.ras_t))
        rep = (self.prev[:b] == nrm[:, None]).any(1) & (nrm != self.eos)
        return torch.where(rep, high, nrm)

    def _tick(self, b):
        """One generation step for rows [0, b). The vocoder decodes the PREVIOUS tick's frame on a side stream,
        overlapping the (latency-bound) fast AR and the slow step of this tick, so self.audio after tick t holds the
        audio of frame t-1 (a row's first tick yields the audio of its voice's last reference frame, which the caller
        discards)."""
        t_in = self.tok[:b]
        self.tok_in[:b].copy_(t_in)
        main = torch.cuda.current_stream()
        self.side.wait_stream(main)
        with torch.cuda.stream(self.side):
            self.audio[:b].copy_(self.voc.step(self.frame_prev[:b], b))
        code0 = (t_in - self.sb).clamp(0, self.cbk - 1)
        codes = self._fast_codes(self.hid[:b], code0, b)                        # [b, 10]
        # ---- slow AR on the frame: the next semantic token ----
        pos = self.cpos[:b]
        h = self.slow.forward(self._embed(t_in, codes), pos, self.ar[:b], pos + 1,
                              nsplit=self.nsplit.get(b, 1), rowmap=False)
        logits = self._head(h)
        # RAS window: the token this tick consumed joins it -- not a row's first token (the checkpoint's generate
        # starts its window after it)
        upd = torch.cat([self.prev[:b, 1:], t_in.unsqueeze(1)], 1)
        self.prev[:b].copy_(torch.where((self.nstep[:b] > 0).unsqueeze(1), upd, self.prev[:b]))
        # EOS sticks: a row that has ended keeps consuming EOS, so its frames stay the silence frame until the slot is
        # admitted again -- a code-switch continuation (keep_voc) can wait a few ticks for its admission without
        # garbage frames running through the vocoder state it keeps
        nt = torch.where(t_in == self.eos, t_in, self._sample_sem(logits, b))
        self.tok[:b].copy_(nt)
        self.hid[:b].copy_(h)
        self.cpos[:b].copy_((pos + 1).clamp(max=self.S - 1))
        self.nstep[:b] += 1
        frame = torch.where((t_in == self.eos).unsqueeze(1), self.sil_frame, codes)
        main.wait_stream(self.side)          # vocoder done reading frame_prev
        self.frame_prev[:b].copy_(frame)

    @torch.inference_mode()
    def capture(self, log=print):
        """Warm up (Triton JIT, cuBLAS handles) and capture one CUDA graph per bucket."""
        t0 = time.perf_counter()
        for b in self.buckets:
            self.nsplit[b] = max(1, min(16, -(-2 * 84 // (b * self.slow.nkv))))
        self._tune(list(self.buckets) + [2 * b for b in self.buckets], log)
        here = os.path.dirname(os.path.abspath(__file__))       # TTS_VOC_TUNE_CACHE relative = from serve/
        self.voc.tune(max(self.buckets), log, os.path.normpath(os.path.join(
            here, os.environ.get("TTS_VOC_TUNE_CACHE", "").strip() or "../data/voc_tune.json")))
        self.pool = torch.cuda.graph_pool_handle()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        order = sorted(self.buckets, reverse=True)          # largest first so the shared pool is sized once
        with torch.cuda.stream(s):
            for b in order:
                for _ in range(2):
                    self._tick(b)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        for b in order:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._tick(b)
            self.graphs[b] = g
        torch.cuda.synchronize()
        if self.pf_buckets:                                  # prefill graphs, all-padding while warming / capturing
            self.pf_x.zero_()
            self.pf_meta.copy_(torch.from_numpy(self._pf_meta_host(self.pf_NT, [], [], [])).to(self.dev))
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for nt in reversed(self.pf_buckets):
                    for _ in range(2):
                        self._pf_body(nt)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            for nt in reversed(self.pf_buckets):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    self._pf_body(nt)
                self.pf_graphs[nt] = g
            torch.cuda.synchronize()
        # the warmup/capture ran garbage through every slot; start clean
        for st in (self.tok, self.cpos, self.hid, self.prev, self.nstep):
            st.zero_()
        self.voc.clear_many(list(range(self.B)))
        log(f"[fast] captured {len(self.graphs)} tick graphs {self.buckets} + {len(self.pf_graphs)} prefill graphs "
            f"{self.pf_buckets} in {time.perf_counter()-t0:.1f}s")

    def _pf_body(self, nt):
        """The prefill graph of nt packed tokens (static pf_x / pf_meta): the slow AR over them (each request's K/V into
        its slot, after its voice's prefix), then each request's first token, hidden and position. Padding rows / blocks
        / requests point at the scratch row SCR."""
        v, nb = self.pf_v, nt // PrefillPlan.BM + self.pf_NR
        plan = SimpleNamespace(q0=v.q0[:nb], n=v.n[:nb], slot=v.bslot[:nb], pos0=v.pos0[:nb], nblk=nb,
                               BM=PrefillPlan.BM)
        h = self.slow.forward(self.pf_x[:nt], v.pos[:nt], v.slot[:nt], None, plan=plan)
        st = v.st.long()
        hl = h[v.last.long()]
        t0 = self._to_token(self.samp(self._head(hl)))       # the first token: no RAS window yet
        self.tok.index_copy_(0, st, t0)
        self.hid.index_copy_(0, st, hl)
        self.cpos.index_copy_(0, st, v.cpos)
        self.prev.index_fill_(0, st, 0)
        self.nstep.index_fill_(0, st, 0)

    def _pf_meta_host(self, nt, slots, ns, starts):
        """pf_meta's contents (host int32) for requests of ns tokens at positions starts.. in slots, packed in order."""
        BM, SCR, o = PrefillPlan.BM, self.SCR, self.pf_off
        m = np.zeros(int(o[-1]), np.int32)
        pos, slot, q0, n, bslot, p0, last, st, cpos = (m[o[i]:o[i + 1]] for i in range(9))
        slot[:], bslot[:], st[:] = SCR, SCR, SCR
        r0 = bi = 0
        for k, (s, cnt, a) in enumerate(zip(slots, ns, starts)):
            pos[r0:r0 + cnt] = np.arange(a, a + cnt)
            slot[r0:r0 + cnt] = s
            for off in range(0, cnt, BM):
                q0[bi], n[bi], bslot[bi], p0[bi] = r0 + off, min(BM, cnt - off), s, a + off
                bi += 1
            last[k], st[k], cpos[k] = r0 + cnt - 1, s, a + cnt
            r0 += cnt
        assert r0 <= nt and bi <= nt // BM + self.pf_NR and len(slots) <= self.pf_NR
        return m

    def _tune(self, Ms, log):
        """Pick cuBLAS or the Triton linear per GEMM shape of this engine's stacks (cached on disk)."""
        if not self.mm.enabled:
            return
        S0, F0, P0 = self.slow.layers[0], self.fast.layers[0], self.voc.post.layers[0]
        ws = [S0.wqkv, S0.wo, S0.w13, S0.w2, self.head, F0.wqkv, F0.wo, F0.w13, F0.w2, self.fast_out,
              P0.wqkv, P0.wo, P0.w13, P0.w2]
        here = os.path.dirname(os.path.abspath(__file__))       # TTS_GEMM_TUNE_CACHE relative = from serve/
        self.mm.tune(ws, Ms, log=log, cache_path=os.path.normpath(os.path.join(
            here, os.environ.get("TTS_GEMM_TUNE_CACHE", "").strip() or ".gemm_tune.json")))
        # Triton specializes integer args on divisibility by 16: JIT both variants of the prefill-sized (large-M)
        # kernels now, not on the first real burst (~1-2 s each).
        for w in ws:
            for M in (512, 513):
                self.mm(torch.randn(M, w.shape[1], device=self.dev, dtype=self.dt), w)
        torch.cuda.synchronize()

    def bucket_for(self, n_rows):
        for b in self.buckets:
            if b >= n_rows:
                return b
        return self.buckets[-1]

    @torch.inference_mode()
    def replay(self, b, bd=0):
        self.graphs[b].replay()

    # ---- guidance (Qwen3-TTS engine only): the worker's calls are no-ops here ----
    def dir_bucket(self):
        return 0

    def pairs_free(self):
        return 0

    def clear_groups(self, slots):
        return None

    def guide_groups(self, groups):
        if groups:
            raise NotImplementedError("guided groups (TTS_INSTRUCT_MODE=guide) are not supported by the Audio8 engine")

    def guide_cross(self, entries):
        return {}

    def attach_director(self, *a, **k):
        raise NotImplementedError("a guidance director (TTS_GUIDE_DIRECTOR) is not supported by the Audio8 engine")

    # ------------------------------------------------------------------ prompts
    def _enc(self, s):
        return self.tokz.encode(s, add_special_tokens=False)

    def _parts_init(self):
        """The prompt's fixed pieces, tokenized one by one as ArkttsProcessor._prompt_segments does."""
        e = self._enc
        self._parts = SimpleNamespace(
            ref_head=e("<|im_start|>system\n") + e("convert the provided text to speech reference to the following:\n\nText:\n"),
            speech=e("\n\nSpeech:\n"),
            user=e("<|im_end|>\n") + e("<|im_start|>user\n"),
            tail=e("<|im_end|>\n") + e("<|im_start|>assistant\n<|voice|>"),
            noref=e("<|im_start|>system\n") + e("convert the provided text to speech") + e("<|im_end|>\n")
            + e("<|im_start|>user\n"))

    @staticmethod
    def _ref_text(text):
        t = _clean_text(text)
        return t if re.search(r"<\|speaker:\d+\|>", t) else f"<|speaker:0|>{t}"

    def _prompt_ids(self, item):
        """The part of the prompt before the target text: (semantic ids [L] long, codes [L, 10] long, is_audio [L]
        bool) on the host. With a reference: system turn with the reference transcript, its codes, then the user turn's
        opening; without one: the plain system turn."""
        P = self._parts
        rc = getattr(item, "ref_code", None) if item is not None else None
        if rc is None or not getattr(item, "ref_text", ""):
            ids = torch.tensor(P.noref, dtype=torch.long)
            return ids, torch.zeros(len(ids), self.ncg, dtype=torch.long), torch.zeros(len(ids), dtype=torch.bool)
        head = P.ref_head + self._enc(self._ref_text(item.ref_text)) + P.speech
        codes = rc.long().cpu()
        T = codes.shape[0]
        sem = torch.cat([torch.tensor(head, dtype=torch.long), codes[:, 0] + self.sb, torch.tensor(P.user, dtype=torch.long)])
        cb = torch.zeros(len(sem), self.ncg, dtype=torch.long)
        cb[len(head):len(head) + T] = codes
        aud = torch.zeros(len(sem), dtype=torch.bool)
        aud[len(head):len(head) + T] = True
        return sem, cb, aud

    def _prompt_embeds(self, sem, cb, aud):
        """Slow AR input embeds [L, H] of prompt pieces (ArkttsModel._embed: codebook sum on audio positions only)."""
        sem, cb, aud = (_to_dev(t, self.dev) for t in (sem, cb, aud))
        e = self.emb[sem]
        return torch.where(aud.unsqueeze(1), e + self.cbemb[cb + self.cb_off].sum(1), e)

    def _speaker_prefix(self, key, item):
        """The voice's text-independent prompt prefix as embeds [Lp, H], cached per voice."""
        hit = self._spk_cache.get(key)
        if hit is None:
            hit = self._prompt_embeds(*self._prompt_ids(item))
            self._spk_cache[key] = hit
        return hit

    @torch.inference_mode()
    def build_prompts(self, reqs):
        """reqs: list of {text, item, key[, ctx_text, ctx_codes]}. Returns per request (pkey, prefix, rest): the slow
        AR input embeds split into the voice's text-independent prefix ([Lp, H], KV-cached per pkey = voice) and the
        rest: the user turn's text + the assistant turn's opening.

        A code-switch continuation (ctx_text / ctx_codes: what the line's previous segment(s) said, and their codes)
        continues a generation: the user turn reads ctx_text + text and the assistant turn already holds ctx_codes, as
        if the model had just spoken them, so it goes on with the rest of the text in the same breath. Its voice's
        prefix stays the cached one (the reference is not touched), so a continuation prefills only its own tokens."""
        if self._parts is None:
            self._parts_init()
        P = self._parts
        sems, cbs, auds = [], [], []
        for r in reqs:
            ctx_t = (r.get("ctx_text") or "").strip()
            ids = torch.tensor(self._enc(_clean_text((ctx_t + " " if ctx_t else "") + r["text"]) or ".") + P.tail,
                               dtype=torch.long)
            cb = torch.zeros(len(ids), self.ncg, dtype=torch.long)
            aud = torch.zeros(len(ids), dtype=torch.bool)
            if r.get("ctx_codes") is not None:
                c = torch.as_tensor(r["ctx_codes"]).long().reshape(-1, self.ncg)
                ids = torch.cat([ids, c[:, 0] + self.sb])
                cb = torch.cat([cb, c])
                aud = torch.cat([aud, torch.ones(len(c), dtype=torch.bool)])
            sems.append(ids), cbs.append(cb), auds.append(aud)
        lens = [len(t) for t in sems]
        rest_e = torch.split(self._prompt_embeds(torch.cat(sems), torch.cat(cbs), torch.cat(auds)), lens)
        return [(r["key"], self._speaker_prefix(r["key"], r["item"]), rest_e[i]) for i, r in enumerate(reqs)]

    # ------------------------------------------------------------------ admission
    def prime_speaker(self, key, item):
        """Decode the voice's reference codes once through the streaming vocoder and keep the state after all but the
        LAST reference frame, plus that last frame: an admitted row loads the state and gets the last ref frame as its
        "previous frame", so the tick's one-frame-behind vocoder decodes it on the row's first tick and lands exactly
        on the fully primed state (no cold-start onset, no extra step). Kept in host memory (SnapStore)."""
        self.prime_speakers([(key, item)])

    @torch.inference_mode()
    def prime_speakers(self, pairs):
        """prime_speaker for many (key, item) pairs: up to B references go through the vocoder side by side (its rows
        are independent), each row snapshotted after its own second-to-last frame. Uses rows [0, B): only while no
        request is running (startup / an idle engine). TTS_VOC_PRIME=0: nothing to do (rows start clean)."""
        if not self.voc_prime:
            return
        todo, seen = [], set()
        for key, item in pairs:
            if self.voc_snaps.path:              # its file is named by the reference it comes from
                self.voc_snaps.set_fp(key, ref_fp(item))
            if key not in self.voc_snaps and key not in seen:
                seen.add(key)
                todo.append((key, item))
        B = self.voc.Bmax
        for g in range(0, len(todo), B):
            grp = todo[g:g + B]
            n = len(grp)
            codes = []
            for _, item in grp:
                rc = getattr(item, "ref_code", None)
                if rc is not None:      # the reference's last TTS_VOC_PRIME_FRAMES frames: the window attention sees
                    rc = rc[-self.prime_frames:]   # 128 back, the convs far less
                codes.append(rc.to(self.dev).long() if rc is not None and rc.shape[0] > 0
                             else torch.zeros(0, self.ncg, dtype=torch.long, device=self.dev))   # [T, 10]
            lens = [int(c.shape[0]) for c in codes]
            T = max(lens)
            pad = torch.zeros(n, max(T, 1), self.ncg, dtype=torch.long, device=self.dev)
            for r, c in enumerate(codes):
                pad[r, :lens[r]] = c
            self.voc.clear_many(list(range(n)))
            snaps = [self.voc.snapshot(r) if lens[r] <= 1 else None for r in range(n)]
            for t in range(T - 1):
                self.voc.step(pad[:, t], n)
                for r in range(n):
                    if lens[r] - 2 == t:
                        snaps[r] = self.voc.snapshot(r)
            for r, (key, _) in enumerate(grp):
                last = codes[r][-1].clone() if lens[r] else self.sil_frame.clone()
                self.voc_snaps[key] = ([t.cpu().pin_memory() for t in snaps[r]], last.cpu())
            self.voc.clear_many(list(range(n)))

    def evict_voices(self, keys) -> int:
        """Drop the GPU prompt state of these voices (prefix KV, prompt embeds): rebuilt on their next use. Rows already
        admitted keep their own KV copy, so this is always safe. Returns the entries dropped."""
        ks, n = set(keys), 0
        for d in (self._pkv, self._spk_cache):
            for k in [k for k in d if k in ks]:
                del d[k]
                n += 1
        return n

    def voice_state_bytes(self) -> int:
        """GPU bytes held by the prompt-state caches (prefix KV + prompt embeds) of all voices."""
        n = sum(t.numel() * t.element_size() for kv in self._pkv.values() for t in kv)
        return n + sum(t.numel() * t.element_size() for t in self._spk_cache.values())

    @torch.inference_mode()
    def admit(self, slots: List[int], reqs: List[dict]):
        """Batched packed prefill of reqs into the given free slots (chunked to TTS_PREFILL_TOKENS tokens per forward).
        Returns each prompt's length, or -1 for a prompt that can't fit the KV cache (that slot is left untouched)."""
        parts = self.build_prompts(reqs)
        lens = [(p.shape[0] if p is not None else 0) + r.shape[0] for _, p, r in parts]
        limit = min(self.S - 32, self.prefill_tokens)
        ok = [i for i, L in enumerate(lens) if L <= limit]
        # chunks a prefill graph takes (<= its tokens and requests), else TTS_PREFILL_TOKENS
        cap_tok = min(self.prefill_tokens, self.pf_NT) if self.pf_buckets else self.prefill_tokens
        cap_req = self.pf_NR if self.pf_buckets else 1 << 30
        chunk, tot = [], 0
        for i in ok:
            n = parts[i][2].shape[0]
            if chunk and (tot + n > cap_tok or len(chunk) >= cap_req):
                self._prefill([slots[k] for k in chunk], [reqs[k] for k in chunk], [parts[k] for k in chunk])
                chunk, tot = [], 0
            chunk.append(i)
            tot += n
        if chunk:
            self._prefill([slots[k] for k in chunk], [reqs[k] for k in chunk], [parts[k] for k in chunk])
        return [L if L <= limit else -1 for L in lens]

    def _prefix_kv(self, pkey, prefix, scratch_slot):
        """KV of a voice's prompt prefix for every slow layer, computed once (into a scratch cache row) and kept:
        [layers] of (k, v) each [nkv, Lp, hd]."""
        hit = self._pkv.get(pkey)
        if hit is not None:
            return hit
        Lp = prefix.shape[0]
        dev = self.dev
        pos = torch.arange(Lp, device=dev, dtype=torch.int32)
        plan = PrefillPlan([0], [Lp], [scratch_slot], [0], dev)
        self.slow.forward(prefix.clone(), pos, torch.full((Lp,), scratch_slot, device=dev, dtype=torch.int32), None,
                          plan=plan)
        kv = (self.slow.kc_all[:, scratch_slot, :, :Lp].clone(), self.slow.vc_all[:, scratch_slot, :, :Lp].clone())
        self._pkv[pkey] = kv
        return kv

    def _prefill(self, slots, reqs, parts):
        dev = self.dev
        # 1) cached prefix KV -> each row's cache (one index_copy per layer per distinct prefix)
        groups: Dict[str, List[int]] = {}
        for s, (pkey, prefix, _) in zip(slots, parts):
            if pkey is not None:
                groups.setdefault(pkey, []).append(s)
        p0 = {}
        for pkey, gslots in groups.items():
            prefix = next(p for (k, p, _) in parts if k == pkey)
            k, v = self._prefix_kv(pkey, prefix, gslots[0])                 # [layers, nkv, Lp, hd]
            Lp = prefix.shape[0]
            idx = _to_dev(torch.tensor(gslots, dtype=torch.long), dev)
            self.slow.kc_all[:, idx, :, :Lp] = k.unsqueeze(1)
            self.slow.vc_all[:, idx, :, :Lp] = v.unsqueeze(1)
            for s in gslots:
                p0[s] = Lp
        # 2) packed causal prefill of the rest, positions continuing after the prefix: a prefill graph when one fits
        rests = [r for _, _, r in parts]
        ns = [int(r.shape[0]) for r in rests]
        starts = [p0.get(s, 0) for s in slots]
        NT = sum(ns)
        nt = next((b for b in self.pf_buckets if b >= NT), None) if self.pf_graphs and len(slots) <= self.pf_NR \
            else None
        if nt is not None:
            self.pf_x[:NT].copy_(torch.cat(rests, 0))
            self.pf_meta.copy_(_to_dev(torch.from_numpy(self._pf_meta_host(nt, slots, ns, starts)), dev))
            self.pf_graphs[nt].replay()
        else:
            x = torch.cat(rests, 0).contiguous()
            pos = _to_dev(torch.cat([torch.arange(a, a + n, dtype=torch.int32) for a, n in zip(starts, ns)]), dev)
            slot_t = _to_dev(torch.cat([torch.full((n,), s, dtype=torch.int32) for n, s in zip(ns, slots)]), dev)
            r0s = np.concatenate([[0], np.cumsum(ns)[:-1]]).tolist()
            plan = PrefillPlan(r0s, ns, slots, starts, dev)
            h = self.slow.forward(x, pos, slot_t, None, plan=plan)
            last = _to_dev(torch.tensor(np.cumsum(ns) - 1), dev)
            hl = h[last]                                                   # [k, H] normed
            j = self.samp(self._head(hl))                                  # the first token: no RAS window yet
            t0 = self._to_token(j)
            st = _to_dev(torch.tensor(slots, dtype=torch.long), dev)
            self.tok[st] = t0
            self.hid[st] = hl
            self.cpos[st] = _to_dev(torch.tensor([a + n for a, n in zip(starts, ns)], dtype=torch.int32), dev)
            self.prev.index_fill_(0, st, 0)
            self.nstep.index_fill_(0, st, 0)
        # 3) vocoder state: the voice's primed snapshot (one index_copy per state tensor). A code-switch continuation
        #    (keep_voc) is admitted into the slot its previous segment just left and keeps that slot's vocoder state,
        #    so the waveform runs on without a seam; its first tick decodes a silence frame (discarded like the ref's
        #    last frame normally is).
        vgroups: Dict[str, List[int]] = {}
        keep = []
        for s, r in zip(slots, reqs):
            if r.get("keep_voc"):
                keep.append(s)
            else:
                vgroups.setdefault(r["key"] if self.voc_prime else None, []).append(s)
        if keep:
            self.frame_prev.index_copy_(0, _to_dev(torch.tensor(keep, dtype=torch.long), dev),
                                        self.sil_frame.unsqueeze(0).expand(len(keep), -1))
        for key, gslots in vgroups.items():
            snap = self.voc_snaps.get(key) if key is not None else None
            idx = _to_dev(torch.tensor(gslots, dtype=torch.long), dev)
            if snap is None:
                self.voc.clear_many(gslots)
                self.frame_prev.index_copy_(0, idx, self.sil_frame.unsqueeze(0).expand(len(gslots), -1))
            else:
                self.voc.load_many(gslots, snap[0])
                self.frame_prev.index_copy_(0, idx, snap[1].to(dev).long().unsqueeze(0).expand(len(gslots), -1))
