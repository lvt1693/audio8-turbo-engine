#!/usr/bin/env python3
"""
Triton kernels for the fast engine (serve/fast_engine.py).

All kernels are CUDA-graph safe (no host syncs, no data-dependent launch shapes) and reproduce
the HF module numerics step-for-step where it matters (bf16 rounding points of RMSNorm, RoPE and
the gated MLP), so the custom engine tracks the reference model closely.

    rmsnorm(x, w, eps, residual=None)      HF Qwen3 RMSNorm, optionally fused with the residual add
    qk_norm_rope_store(...)                per-head RMSNorm on q/k + RoPE + write k/v into a cache
    bias_rope_store(...)                   qkv bias + RoPE (fp32, rounded once) + write k/v into a cache
    decode_attention(q, kc, vc, lens, ...) split-K flash-decoding over per-row cache lengths (GQA)
    silu_mul(gu)                           silu(gate) * up on a fused [.., 2I] projection
    prefill_attention(...)                 packed causal flash-attention over KV stored in cache rows
    linear_triton / GemmDispatch           bf16 GEMM with small-M tiles, chosen per shape vs cuBLAS
    conv_cl / snake_cat                    channels-last implicit-GEMM causal (transposed) conv + SnakeBeta
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------------------------
# RMSNorm (+ fused residual add)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _rmsnorm_kernel(X, R, W, Y, RO, sx, sr, sy, sro, N, eps,
                    HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(X + row * sx + offs, mask=m, other=0.0)
    if HAS_RES:
        r = tl.load(R + row * sr + offs, mask=m, other=0.0)
        x = (x.to(tl.float32) + r.to(tl.float32)).to(Y.dtype.element_ty)   # bf16 residual add
        tl.store(RO + row * sro + offs, x, mask=m)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / N
    xn = (xf * tl.math.rsqrt(var + eps)).to(Y.dtype.element_ty)          # HF: .to(input_dtype)
    w = tl.load(W + offs, mask=m, other=0.0)
    y = (w.to(tl.float32) * xn.to(tl.float32)).to(Y.dtype.element_ty)   # HF: weight * x
    tl.store(Y + row * sy + offs, y, mask=m)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, residual: torch.Tensor = None,
            out: torch.Tensor = None, res_out: torch.Tensor = None):
    """x: [..., N]. With residual: h = x + residual (stored to res_out, default in-place into
    residual) and returns (rmsnorm(h), h). Without: returns rmsnorm(x)."""
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    rows = x2.shape[0]
    y = out if out is not None else torch.empty_like(x)
    y2 = y.reshape(-1, N)
    BLOCK = triton.next_power_of_2(N)
    nw = 4 if BLOCK <= 1024 else 8
    if residual is not None:
        r2 = residual.reshape(-1, N)
        ro = res_out if res_out is not None else residual
        ro2 = ro.reshape(-1, N)
        _rmsnorm_kernel[(rows,)](x2, r2, w, y2, ro2, x2.stride(0), r2.stride(0), y2.stride(0),
                                 ro2.stride(0), N, eps, HAS_RES=True, BLOCK=BLOCK, num_warps=nw)
        return y, ro
    _rmsnorm_kernel[(rows,)](x2, x2, w, y2, y2, x2.stride(0), 0, y2.stride(0), 0, N, eps,
                             HAS_RES=False, BLOCK=BLOCK, num_warps=nw)
    return y


# ----------------------------------------------------------------------------------------------
# per-head q/k RMSNorm + RoPE + KV-cache store
# ----------------------------------------------------------------------------------------------
@triton.jit
def _qk_rope_store_kernel(QKV, QN, KN, COS, SIN, POS, SLOT, QO, KC, VC,
                          s_qkv, s_qo_r, s_qo_h, s_kc_b, s_kc_h, s_kc_s, s_cs,
                          eps, RING,
                          NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                          HAS_NORM: tl.constexpr):
    r = tl.program_id(0)             # token row
    h = tl.program_id(1)             # 0..NQ-1 q heads, NQ..NQ+NKV-1 k heads, then v heads
    HALF: tl.constexpr = HD // 2
    d = tl.arange(0, HD)
    dp = (d + HALF) % HD             # rotate_half partner index
    base = QKV + r * s_qkv
    pos = tl.load(POS + r)
    slot = tl.load(SLOT + r).to(tl.int64)
    sidx = pos % RING
    if h >= NQ + NKV:                # value head: plain copy into the cache
        hv = h - NQ - NKV
        v = tl.load(base + (NQ + NKV) * HD + hv * HD + d)
        tl.store(VC + slot * s_kc_b + hv * s_kc_h + sidx * s_kc_s + d, v)
    else:
        if h < NQ:
            src = base + h * HD
            W = QN
        else:
            src = base + NQ * HD + (h - NQ) * HD
            W = KN
        x = tl.load(src + d).to(tl.float32)
        xp = tl.load(src + dp).to(tl.float32)
        if HAS_NORM:
            rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / HD + eps)
            w = tl.load(W + d).to(tl.float32)
            wp = tl.load(W + dp).to(tl.float32)
            y = (w * (x * rstd).to(QO.dtype.element_ty).to(tl.float32)).to(QO.dtype.element_ty).to(tl.float32)
            yp = (wp * (xp * rstd).to(QO.dtype.element_ty).to(tl.float32)).to(QO.dtype.element_ty).to(tl.float32)
        else:
            y = x
            yp = xp
        c = tl.load(COS + pos * s_cs + d).to(tl.float32)
        s = tl.load(SIN + pos * s_cs + d).to(tl.float32)
        sign = tl.where(d < HALF, -1.0, 1.0)
        a = (y * c).to(QO.dtype.element_ty).to(tl.float32)
        b = (sign * yp * s).to(QO.dtype.element_ty).to(tl.float32)
        o = (a + b).to(QO.dtype.element_ty)
        if h < NQ:
            tl.store(QO + r * s_qo_r + h * s_qo_h + d, o)
        else:
            hk = h - NQ
            tl.store(KC + slot * s_kc_b + hk * s_kc_h + sidx * s_kc_s + d, o)


def qk_norm_rope_store(qkv, qn, kn, cos_tab, sin_tab, pos, slot, qo, kc, vc, nq, nkv, hd,
                       eps=1e-6, ring=1 << 30, has_norm=True):
    """qkv: [R, (nq+2nkv)*hd] fused projection rows. pos/slot: [R] int (rope position, cache row).
    Writes q (normed, rotated) to qo [R, nq, hd]; k (normed, rotated) and v into kc/vc
    [Bslots, nkv, S, hd] at [slot, :, pos % ring]."""
    R = qkv.shape[0]
    _qk_rope_store_kernel[(R, nq + 2 * nkv)](
        qkv, qn if qn is not None else qkv, kn if kn is not None else qkv, cos_tab, sin_tab, pos, slot,
        qo, kc, vc, qkv.stride(0), qo.stride(0), qo.stride(1),
        kc.stride(0), kc.stride(1), kc.stride(2), cos_tab.stride(0), eps, ring,
        NQ=nq, NKV=nkv, HD=hd, HAS_NORM=has_norm, num_warps=1 if hd <= 64 else 2)
    return qo


@triton.jit
def _bias_rope_store_kernel(QKV, BIAS, COS, SIN, POS, SLOT, QO, KC, VC,
                            s_qkv, s_qo_r, s_qo_h, s_kc_b, s_kc_h, s_kc_s, s_cs, RING,
                            NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Arktts attention input: (qkv + bias) -> RoPE on q / k in fp32, rounded once (the model's _apply_rope:
    x.float() * rope, then .to(bf16)). The interleaved-pair RoPE of the checkpoint runs as rotate_half: the
    q / k rows of the fused weight are permuted per head at load (A8 engine), which leaves q.k unchanged."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = HD // 2
    d = tl.arange(0, HD)
    dp = (d + HALF) % HD
    base = QKV + r * s_qkv
    pos = tl.load(POS + r)
    slot = tl.load(SLOT + r).to(tl.int64)
    sidx = pos % RING
    if h >= NQ + NKV:
        hv = h - NQ - NKV
        col = (NQ + NKV) * HD + hv * HD
        v = tl.load(base + col + d).to(tl.float32)
        if HAS_BIAS:
            v += tl.load(BIAS + col + d).to(tl.float32)
        tl.store(VC + slot * s_kc_b + hv * s_kc_h + sidx * s_kc_s + d, v.to(VC.dtype.element_ty))
    else:
        col = h * HD                     # q heads, then k heads: contiguous in the fused row
        x = tl.load(base + col + d).to(tl.float32)
        xp = tl.load(base + col + dp).to(tl.float32)
        if HAS_BIAS:
            x += tl.load(BIAS + col + d).to(tl.float32)
            xp += tl.load(BIAS + col + dp).to(tl.float32)
        c = tl.load(COS + pos * s_cs + d).to(tl.float32)
        s = tl.load(SIN + pos * s_cs + d).to(tl.float32)
        sign = tl.where(d < HALF, -1.0, 1.0)
        o = (x * c + sign * xp * s).to(QO.dtype.element_ty)
        if h < NQ:
            tl.store(QO + r * s_qo_r + h * s_qo_h + d, o)
        else:
            tl.store(KC + slot * s_kc_b + (h - NQ) * s_kc_h + sidx * s_kc_s + d, o)


def bias_rope_store(qkv, bias, cos_tab, sin_tab, pos, slot, qo, kc, vc, nq, nkv, hd, ring=1 << 30):
    """qkv: [R, (nq+2nkv)*hd]; bias: [(nq+2nkv)*hd] or None. Writes rotated q to qo [R, nq, hd] and k / v
    into kc / vc [Bslots, nkv, S, hd] at [slot, :, pos % ring]."""
    R = qkv.shape[0]
    _bias_rope_store_kernel[(R, nq + 2 * nkv)](
        qkv, bias if bias is not None else qkv, cos_tab, sin_tab, pos, slot, qo, kc, vc,
        qkv.stride(0), qo.stride(0), qo.stride(1), kc.stride(0), kc.stride(1), kc.stride(2), cos_tab.stride(0), ring,
        NQ=nq, NKV=nkv, HD=hd, HAS_BIAS=bias is not None, num_warps=1 if hd <= 64 else 2)
    return qo


# ----------------------------------------------------------------------------------------------
# split-K decode attention (one query token per row, GQA)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _dec_attn_split_kernel(Q, KC, VC, LENS, ROWMAP, ACC, ML, scale,
                           s_qb, s_qh, s_kb, s_kh, s_ks, s_ab, s_ah, s_as, s_mb, s_mh, s_ms,
                           GROUP: tl.constexpr, GP: tl.constexpr, HD: tl.constexpr,
                           BN: tl.constexpr, NSPLIT: tl.constexpr, HAS_MAP: tl.constexpr,
                           PREC: tl.constexpr):
    b = tl.program_id(0)
    kh = tl.program_id(1)
    sp = tl.program_id(2)
    L = tl.load(LENS + b)
    if HAS_MAP:
        crow = tl.load(ROWMAP + b)
    else:
        crow = b
    per = tl.cdiv(tl.cdiv(L, NSPLIT), BN) * BN
    lo = sp * per
    hi = tl.minimum(lo + per, L)
    g = tl.arange(0, GP)
    gm = g < GROUP
    d = tl.arange(0, HD)
    q = tl.load(Q + b * s_qb + (kh * GROUP + g)[:, None] * s_qh + d[None, :], mask=gm[:, None], other=0.0)
    m_i = tl.full([GP], float("-inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, HD], tl.float32)
    kbase = KC + crow.to(tl.int64) * s_kb + kh * s_kh
    vbase = VC + crow.to(tl.int64) * s_kb + kh * s_kh
    for start in range(lo, hi, BN):
        n = start + tl.arange(0, BN)
        nm = n < hi
        k = tl.load(kbase + n[:, None] * s_ks + d[None, :], mask=nm[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=PREC) * scale   # [GP, BN] fp32
        s = tl.where(nm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vbase + n[:, None] * s_ks + d[None, :], mask=nm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PREC)
        m_i = m_new
    hq = kh * GROUP + g
    tl.store(ACC + b * s_ab + hq[:, None] * s_ah + sp * s_as + d[None, :], acc, mask=gm[:, None])
    tl.store(ML + b * s_mb + hq * s_mh + sp * s_ms, m_i, mask=gm)
    tl.store(ML + b * s_mb + hq * s_mh + sp * s_ms + 1, l_i, mask=gm)


@triton.jit
def _dec_attn_combine_kernel(ACC, ML, O, s_ab, s_ah, s_as, s_mb, s_mh, s_ms, s_ob, s_oh,
                             HD: tl.constexpr, NSPLIT: tl.constexpr, NSP: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.arange(0, NSP)
    spm = sp < NSPLIT
    d = tl.arange(0, HD)
    m = tl.load(ML + b * s_mb + h * s_mh + sp * s_ms, mask=spm, other=float("-inf"))
    l = tl.load(ML + b * s_mb + h * s_mh + sp * s_ms + 1, mask=spm, other=0.0)
    M = tl.max(m, axis=0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp(m - M))
    acc = tl.load(ACC + b * s_ab + h * s_ah + sp[:, None] * s_as + d[None, :], mask=spm[:, None], other=0.0)
    num = tl.sum(acc * w[:, None], axis=0)
    den = tl.sum(l * w, axis=0)
    tl.store(O + b * s_ob + h * s_oh + d, (num / den).to(O.dtype.element_ty))


class DecodeAttention:
    """Workspace-owning split-K decode attention. The scratch buffers are flat and sized in
    (query row x split) units, so one instance serves both decode (few rows, many splits) and
    packed prefill (many rows, one split) with fixed addresses for graph capture."""

    def __init__(self, cap_row_splits, nq, nkv, hd, nsplit_max, device, scale=None):
        self.nq, self.nkv, self.hd = nq, nkv, hd
        self.group = nq // nkv
        self.gp = max(16, triton.next_power_of_2(self.group))
        self.nsplit_max = int(nsplit_max)
        self.cap = int(cap_row_splits)
        self.scale = scale if scale is not None else hd ** -0.5
        self.acc_flat = torch.empty(self.cap * nq * hd, dtype=torch.float32, device=device)
        self.ml_flat = torch.empty(self.cap * nq * 2, dtype=torch.float32, device=device)

    def __call__(self, q, kc, vc, lens, out, nsplit=1, bn=64, rowmap=None):
        """q: [b, nq, hd]; kc/vc: [rows, nkv, S, hd]; query row i reads cache row rowmap[i]
        (identity when rowmap is None); lens: [b] int32 (keys [0, lens) attended); out: [b, nq, hd]."""
        b = q.shape[0]
        nsplit = max(1, min(int(nsplit), self.nsplit_max))
        assert b * nsplit <= self.cap, (b, nsplit, self.cap)
        acc = self.acc_flat[:b * self.nq * nsplit * self.hd].view(b, self.nq, nsplit, self.hd)
        ml = self.ml_flat[:b * self.nq * nsplit * 2].view(b, self.nq, nsplit, 2)
        prec = "ieee" if q.dtype == torch.float32 else "tf32"
        _dec_attn_split_kernel[(b, self.nkv, nsplit)](
            q, kc, vc, lens, rowmap if rowmap is not None else lens, acc, ml, self.scale,
            q.stride(0), q.stride(1), kc.stride(0), kc.stride(1), kc.stride(2),
            acc.stride(0), acc.stride(1), acc.stride(2), ml.stride(0), ml.stride(1), ml.stride(2),
            GROUP=self.group, GP=self.gp, HD=self.hd, BN=bn, NSPLIT=nsplit,
            HAS_MAP=rowmap is not None, PREC=prec, num_warps=4)
        _dec_attn_combine_kernel[(b, self.nq)](
            acc, ml, out, acc.stride(0), acc.stride(1), acc.stride(2),
            ml.stride(0), ml.stride(1), ml.stride(2), out.stride(0), out.stride(1),
            HD=self.hd, NSPLIT=nsplit, NSP=max(2, triton.next_power_of_2(nsplit)), num_warps=2)
        return out


# ----------------------------------------------------------------------------------------------
# packed causal prefill attention (flash-attention forward over KV already stored in cache rows)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _prefill_attn_kernel(Q, KC, VC, O, BQ0, BN_, BSLOT, BPOS0, scale,
                         s_qr, s_qh, s_kb, s_kh, s_ks, s_or, s_oh,
                         GROUP: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    blk = tl.program_id(0)
    h = tl.program_id(1)
    q0 = tl.load(BQ0 + blk)                    # first packed query row of this block
    nq = tl.load(BN_ + blk)                    # valid rows in the block (<= BM)
    slot = tl.load(BSLOT + blk).to(tl.int64)   # cache row of the request
    p0 = tl.load(BPOS0 + blk)                  # position of the block's first row
    kh = h // GROUP
    m = tl.arange(0, BM)
    d = tl.arange(0, HD)
    mm = m < nq
    q = tl.load(Q + (q0 + m)[:, None] * s_qr + h * s_qh + d[None, :], mask=mm[:, None], other=0.0)
    qpos = p0 + m
    hi = p0 + nq
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)
    kb = KC + slot * s_kb + kh * s_kh
    vb = VC + slot * s_kb + kh * s_kh
    for start in range(0, hi, BN):
        n = start + tl.arange(0, BN)
        nm = n < hi
        k = tl.load(kb + n[:, None] * s_ks + d[None, :], mask=nm[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where((n[None, :] <= qpos[:, None]) & nm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vb + n[:, None] * s_ks + d[None, :], mask=nm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(O + (q0 + m)[:, None] * s_or + h * s_oh + d[None, :], out.to(O.dtype.element_ty), mask=mm[:, None])


class PrefillPlan:
    """Query-block schedule for a packed prefill: request i's rows [r0_i, r0_i+n_i) sit at
    positions [p0_i, p0_i+n_i) of cache row slot_i (keys [0, p0_i) are an already-stored prefix)."""
    BM = 64

    def __init__(self, r0s, ns, slots, p0s, device):
        q0, cnt, sl, ps = [], [], [], []
        for r0, n, s, p0 in zip(r0s, ns, slots, p0s):
            for off in range(0, n, self.BM):
                q0.append(r0 + off); cnt.append(min(self.BM, n - off)); sl.append(s); ps.append(p0 + off)
        t = torch.tensor([q0, cnt, sl, ps], dtype=torch.int32).pin_memory().to(device, non_blocking=True)
        self.q0, self.n, self.slot, self.pos0 = t[0], t[1], t[2], t[3]
        self.nblk = len(q0)


def prefill_attention(q, kc, vc, plan: PrefillPlan, out, group, scale=None):
    """q/out: [R, nq, hd] packed rows; kc/vc: [rows, nkv, S, hd]."""
    nq, hd = q.shape[1], q.shape[2]
    scale = scale if scale is not None else hd ** -0.5
    _prefill_attn_kernel[(plan.nblk, nq)](
        q, kc, vc, out, plan.q0, plan.n, plan.slot, plan.pos0, scale,
        q.stride(0), q.stride(1), kc.stride(0), kc.stride(1), kc.stride(2), out.stride(0), out.stride(1),
        GROUP=group, HD=hd, BM=plan.BM, BN=64, num_warps=4, num_stages=2)
    return out


# ----------------------------------------------------------------------------------------------
# linear: y = x @ W^T, bf16 in/out, bf16 MMA with fp32 accumulation, small-M tile shapes
# ----------------------------------------------------------------------------------------------
@triton.jit
def _linear_kernel(X, W, Y, M, N, K, sxm, swn, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   GM: tl.constexpr):
    """bf16 MMA with fp32 accumulation (same numerics class as cuBLAS). The win over cuBLAS is the
    tile shape for the small decode M (cuBLAS picks 128-row tiles at M=80). An earlier fp16 variant
    (bf16 -> fp16 in registers, fp16-accumulated BK slices) overflowed on the large down_proj
    inputs of the 0.6B talker -- inf logits, garbage tokens, no EOS -- so it is not used."""
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    gid = pid // (GM * npn)
    fm = gid * GM
    gs = min(npm - fm, GM)
    pm = fm + (pid % gs)
    pn = (pid % (GM * npn)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    xp = X + rm[:, None].to(tl.int64) * sxm + rk[None, :]
    wp = W + rn[None, :].to(tl.int64) * swn + rk[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(xp, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(wp, mask=(rn[None, :] < N) & (rk[:, None] + k < K), other=0.0)
        acc += tl.dot(a, b)
        xp += BK
        wp += BK
    tl.store(Y + rm[:, None].to(tl.int64) * sym + rn[None, :], acc.to(Y.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def linear_triton(x, w, cfg, out=None):
    """x: [M, K] (row-contiguous), w: [N, K] (nn.Linear layout) -> [M, N]. cfg=(BM,BN,BK,warps,stages)."""
    M, K = x.shape
    N = w.shape[0]
    y = out if out is not None else torch.empty(M, N, dtype=x.dtype, device=x.device)
    BM, BN, BK, nw, ns = cfg
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _linear_kernel[grid](x, w, y, M, N, K, x.stride(0), w.stride(0), y.stride(0), BM=BM, BN=BN, BK=BK, GM=8,
                         num_warps=nw, num_stages=ns)
    return y


# the configs that won an offline sweep on RTX 3090 over the talker/predictor decode shapes
LIN_CANDIDATES_SMALL = [(16, 64, 128, 4, 4), (16, 64, 64, 4, 3), (32, 64, 128, 8, 4), (32, 128, 64, 8, 4),
                        (64, 64, 128, 8, 3), (64, 64, 64, 8, 3), (64, 128, 64, 8, 4)]
LIN_CANDIDATES_LARGE = [(128, 128, 32, 4, 4), (128, 128, 64, 8, 3), (128, 256, 32, 8, 3), (64, 256, 32, 8, 3)]


class GemmDispatch:
    """Per-(M, N, K) choice between cuBLAS and a Triton config, tuned on the running GPU at startup
    (outside graph capture). M values not tuned fall back to cuBLAS for small M, and to the tuned
    large-M config (if it won) for prefill-sized M."""

    def __init__(self):
        self.small = {}   # (M, N, K) -> cfg or None (None = cuBLAS)
        self.large = {}   # (N, K) -> cfg or None
        self.enabled = True

    @staticmethod
    def _time(fn, n=20):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                fn()
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); g.replay(); e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / (2 * n)

    @torch.inference_mode()
    def tune(self, weights, Ms, large_m=4096, log=print, cache_path=None):
        """weights: iterable of [N, K] tensors (distinct shapes are tuned once). Results are cached
        in cache_path (JSON, keyed by GPU name) so later startups skip the measurement."""
        import json
        shapes = {}
        for w in weights:
            shapes.setdefault((w.shape[0], w.shape[1]), w)
        gpu = torch.cuda.get_device_name()
        disk = {}
        if cache_path and os.path.exists(cache_path):
            try:
                disk = json.load(open(cache_path)).get(gpu, {})
            except Exception:
                disk = {}
        todo_small = [(M, N, K) for (N, K) in shapes for M in sorted(set(Ms)) if f"s{M},{N},{K}" not in disk]
        todo_large = [(N, K) for (N, K) in shapes if f"l{N},{K}" not in disk]
        for k, v in disk.items():
            key = tuple(int(x) for x in k[1:].split(","))
            (self.small if k[0] == "s" else self.large)[key] = tuple(v) if v else None
        if not todo_small and not todo_large:
            log(f"[gemm] loaded {len(disk)} tuned entries from {cache_path}")
            return
        wins = 0
        for (N, K), w in shapes.items():
            for M in sorted(set(Ms)):
                if (M, N, K) not in todo_small:
                    continue
                x = torch.randn(M, K, device=w.device, dtype=w.dtype)
                best, best_t = None, self._time(lambda: torch.matmul(x, w.t()))
                for cfg in LIN_CANDIDATES_SMALL:
                    if cfg[0] > 2 * max(16, triton.next_power_of_2(M)):
                        continue
                    try:
                        t = self._time(lambda: linear_triton(x, w, cfg))
                    except Exception:
                        continue
                    if t < best_t * 0.97:
                        best, best_t = cfg, t
                self.small[(M, N, K)] = best
                wins += best is not None
            if (N, K) not in todo_large:
                continue
            x = torch.randn(large_m, K, device=w.device, dtype=w.dtype)
            best, best_t = None, self._time(lambda: torch.matmul(x, w.t()), n=5)
            for cfg in LIN_CANDIDATES_LARGE:
                try:
                    t = self._time(lambda: linear_triton(x, w, cfg), n=5)
                except Exception:
                    continue
                if t < best_t * 0.97:
                    best, best_t = cfg, t
            self.large[(N, K)] = best
        log(f"[gemm] tuned {len(todo_small)} small + {len(todo_large)} large shapes: triton wins {wins} small, "
            f"{sum(v is not None for v in self.large.values())}/{len(self.large)} large")
        if cache_path:
            try:
                allc = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
            except Exception:
                allc = {}
            ent = allc.setdefault(gpu, {})
            for (M, N, K), v in self.small.items():
                ent[f"s{M},{N},{K}"] = list(v) if v else None
            for (N, K), v in self.large.items():
                ent[f"l{N},{K}"] = list(v) if v else None
            tmp = cache_path + ".tmp"
            json.dump(allc, open(tmp, "w"))
            os.replace(tmp, cache_path)

    def __call__(self, x, w):
        if not self.enabled:
            return torch.matmul(x, w.t())
        M, K = x.shape
        N = w.shape[0]
        cfg = self.small.get((M, N, K), "miss")
        if cfg == "miss":
            cfg = self.large.get((N, K)) if M > 256 else None
        if cfg is None or x.stride(1) != 1:
            return torch.matmul(x, w.t())
        return linear_triton(x, w, cfg)


# ----------------------------------------------------------------------------------------------
# gated MLP activation
# ----------------------------------------------------------------------------------------------
@triton.jit
def _silu_mul_kernel(GU, Y, s_gu, s_y, I, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < I
    g = tl.load(GU + r * s_gu + c, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + r * s_gu + I + c, mask=m, other=0.0).to(tl.float32)
    gs = (g / (1.0 + tl.exp(-g))).to(Y.dtype.element_ty).to(tl.float32)   # HF: act(gate) in bf16
    tl.store(Y + r * s_y + c, (gs * u).to(Y.dtype.element_ty), mask=m)


def silu_mul(gu: torch.Tensor, out: torch.Tensor = None):
    I = gu.shape[-1] // 2
    g2 = gu.reshape(-1, 2 * I)
    y = out if out is not None else torch.empty(*gu.shape[:-1], I, dtype=gu.dtype, device=gu.device)
    y2 = y.reshape(-1, I)
    BLOCK = 1024
    _silu_mul_kernel[(g2.shape[0], triton.cdiv(I, BLOCK))](g2, y2, g2.stride(0), y2.stride(0), I,
                                                          BLOCK=BLOCK, num_warps=4)
    return y


# ----------------------------------------------------------------------------------------------
# channels-last causal conv / transposed conv as implicit GEMM (vocoder decoder)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _conv_cl_kernel(X, W, BIAS, RES, Y, SA, SIB, M, TR, Cin, Cout, s_xb, s_xt, s_yb, s_yt, s_rb, s_rt,
                    s_wp, s_wt, s_wc,
                    TAPS: tl.constexpr, DIL: tl.constexpr, PH: tl.constexpr,
                    HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr, F16ACC: tl.constexpr, HAS_SNAKE: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Rows r = (b, t), t in [0, TR). Output row t*PH + p (p = phase, program axis 2):
         Y[b, t*PH+p, :] = bias + sum_j X[b, t + j*DIL, :] @ W[p, j]   (+ RES[b, t*PH+p, :])
    X is the context-padded input (state ++ new samples), W is [PH, TAPS, Cin, Cout]. HAS_SNAKE (a context-free
    conv only: every row of X is raw): the snake x + SIB sin(SA x)^2 is applied to X on the load."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    p = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mmask = rm < M
    nmask = rn < Cout
    b = (rm // TR).to(tl.int64)
    t = rm % TR
    acc = tl.zeros((BM, BN), tl.float32)
    for j in range(TAPS):
        xrow = X + b * s_xb + (t + j * DIL) * s_xt
        wbase = W + p * s_wp + j * s_wt
        for k0 in range(0, Cin, BK):
            kk = k0 + rk
            a = tl.load(xrow[:, None] + kk[None, :], mask=mmask[:, None] & (kk[None, :] < Cin), other=0.0)
            if HAS_SNAKE:
                af = a.to(tl.float32)
                sa = tl.load(SA + kk, mask=kk < Cin, other=0.0)
                sib = tl.load(SIB + kk, mask=kk < Cin, other=0.0)
                sv = tl.sin(af * sa[None, :])
                a = (af + sib[None, :] * sv * sv).to(tl.float16)
            w = tl.load(wbase + kk[:, None] * s_wc + rn[None, :], mask=(kk[:, None] < Cin) & nmask[None, :], other=0.0)
            if F16ACC:
                acc += tl.dot(a, w, out_dtype=tl.float16).to(tl.float32)
            else:
                acc += tl.dot(a, w)
    if HAS_BIAS:
        acc += tl.load(BIAS + rn, mask=nmask, other=0.0).to(tl.float32)[None, :]
    trow = t * PH + p
    if HAS_RES:
        r = tl.load(RES + b[:, None] * s_rb + trow[:, None] * s_rt + rn[None, :],
                    mask=mmask[:, None] & nmask[None, :], other=0.0)
        acc += r.to(tl.float32)
    tl.store(Y + b[:, None] * s_yb + trow[:, None] * s_yt + rn[None, :], acc.to(Y.dtype.element_ty),
             mask=mmask[:, None] & nmask[None, :])


def conv_cl(xin, w, bias, TR, taps, dil, ph, out, res=None, f16acc=True, cfg=None, snake=None):
    """xin: [B, Tin, Cin] contiguous fp16; w: [ph, taps, Cin, Cout_pad]; out: [B, TR*ph, >=Cout].
    cfg: (BM, BN, BK, warps, stages), tuned per layer (StreamingVocoder.tune); default a heuristic.
    snake: (alpha, 1/(alpha+eps)) fp32 [Cin] applied to xin on the load (context-free convs only)."""
    B, _, Cin = xin.shape
    Cout = out.shape[-1]
    M = B * TR
    if cfg is not None:
        BM, BN, BK, nw, ns = cfg
    else:
        BN = min(128, max(16, triton.next_power_of_2(Cout)))
        BM = 64 if M >= 4096 else 32
        BK = 64 if Cin >= 64 else max(16, triton.next_power_of_2(Cin))
        nw, ns = 4, 3
    grid = (triton.cdiv(M, BM), triton.cdiv(Cout, BN), ph)
    _conv_cl_kernel[grid](xin, w, bias if bias is not None else w, res if res is not None else out, out,
                          snake[0] if snake is not None else w, snake[1] if snake is not None else w,
                          M, TR, Cin, Cout, xin.stride(0), xin.stride(1), out.stride(0), out.stride(1),
                          res.stride(0) if res is not None else 0, res.stride(1) if res is not None else 0,
                          w.stride(0), w.stride(1), w.stride(2),
                          TAPS=taps, DIL=dil, PH=ph, HAS_BIAS=bias is not None, HAS_RES=res is not None,
                          F16ACC=f16acc, HAS_SNAKE=snake is not None, BM=BM, BN=BN, BK=BK, num_warps=nw,
                          num_stages=ns)
    return out


@triton.jit
def _ru_fused_kernel(X, ST, A1, IB1, W1, B1, A2, IB2, W2, B2, Y, T, C,
                     s_xb, s_xt, s_sb, s_st, s_yb, s_yt, s_w1j,
                     K1: tl.constexpr, DIL: tl.constexpr, CTX: tl.constexpr, CP: tl.constexpr,
                     BM: tl.constexpr, BK: tl.constexpr, BN2: tl.constexpr, F16ACC: tl.constexpr):
    """One whole DAC residual unit, channels-last, for rows [t0, t0+BM) of batch row b:
         Y = X + conv1x1(snake2(conv_k(snake1([ST | X]), dil) + B1)) + B2
    ST holds the CTX = (K1-1)*DIL activated input rows before X (the causal context). The conv_k output never leaves
    registers: [BM, CP] (CP = C padded to a power of 2; padded weights / biases / alphas are 0)."""
    pid_m = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    rm = pid_m * BM + tl.arange(0, BM)
    mm = rm < T
    cn = tl.arange(0, CP)
    acc = tl.zeros((BM, CP), tl.float32)
    for j in tl.static_range(K1):
        r = rm + (j - (K1 - 1)) * DIL                  # input row; < 0: the context
        in_st = r < 0
        for k0 in range(0, C, BK):
            kk = k0 + tl.arange(0, BK)
            km = kk < C
            xs = tl.load(ST + b * s_sb + (r + CTX)[:, None] * s_st + kk[None, :],
                         mask=(in_st & mm)[:, None] & km[None, :], other=0.0).to(tl.float32)
            xr = tl.load(X + b * s_xb + r[:, None] * s_xt + kk[None, :],
                         mask=((~in_st) & mm)[:, None] & km[None, :], other=0.0).to(tl.float32)
            a = tl.load(A1 + kk, mask=km, other=0.0)
            ib = tl.load(IB1 + kk, mask=km, other=0.0)
            sv = tl.sin(xr * a[None, :])
            xr = xr + ib[None, :] * sv * sv
            xin = tl.where(in_st[:, None], xs, xr).to(tl.float16)
            w = tl.load(W1 + j * s_w1j + kk[:, None] * CP + cn[None, :], mask=km[:, None], other=0.0)
            if F16ACC:
                acc += tl.dot(xin, w, out_dtype=tl.float16).to(tl.float32)
            else:
                acc += tl.dot(xin, w)
    h = acc + tl.load(B1 + cn)[None, :]
    a2 = tl.load(A2 + cn)
    ib2 = tl.load(IB2 + cn)
    s2 = tl.sin(h * a2[None, :])
    hf = (h + ib2[None, :] * s2 * s2).to(tl.float16)
    for n0 in tl.static_range(0, CP, BN2):
        rn = n0 + tl.arange(0, BN2)
        nm = rn < C
        w2 = tl.load(W2 + cn[:, None] * CP + rn[None, :])
        if F16ACC:
            o = tl.dot(hf, w2, out_dtype=tl.float16).to(tl.float32)
        else:
            o = tl.dot(hf, w2)
        o += tl.load(B2 + rn)[None, :]
        res = tl.load(X + b * s_xb + rm[:, None] * s_xt + rn[None, :], mask=mm[:, None] & nm[None, :], other=0.0)
        o += res.to(tl.float32)
        tl.store(Y + b * s_yb + rm[:, None] * s_yt + rn[None, :], o.to(Y.dtype.element_ty), mask=mm[:, None] & nm[None, :])


def ru_fused(x, state, a1, ib1, w1, b1, a2, ib2, w2, b2, k1, dil, out, bm=64, bk=32, bn2=64, nw=4, ns=2,
             f16acc=True):
    """x: [B, T, C] fp16 (raw RU input, T >= the context); state: [>=B, CTX, C] activated context (read, NOT
    updated: the caller refreshes it after); w1: [K1, C, CP], w2: [CP, CP] fp16, the rest padded to CP. -> out."""
    B, T, C = x.shape
    CP = w2.shape[0]
    CTX = (k1 - 1) * dil
    grid = (triton.cdiv(T, bm), B)
    _ru_fused_kernel[grid](x, state, a1, ib1, w1, b1, a2, ib2, w2, b2, out, T, C,
                           x.stride(0), x.stride(1), state.stride(0), state.stride(1), out.stride(0), out.stride(1),
                           w1.stride(0), K1=k1, DIL=dil, CTX=CTX, CP=CP, BM=bm, BK=bk, BN2=min(bn2, CP),
                           F16ACC=f16acc, num_warps=nw, num_stages=ns)
    return out


@triton.jit
def _snake_cat_kernel(X, ST, A, IB, Y, T, C, CTX, s_xb, s_xt, s_sb, s_st, s_yb, s_yt,
                      HAS_SNAKE: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """Y[b] = concat(ST[b] (CTX rows, already activated), snake(X[b]) (T rows)); channels-last."""
    b = tl.program_id(0).to(tl.int64)
    tt = tl.program_id(1) * BT + tl.arange(0, BT)          # output row in [0, CTX+T)
    cc = tl.program_id(2) * BC + tl.arange(0, BC)
    cm = cc < C
    in_state = tt < CTX
    rm = tt < CTX + T
    s = tl.load(ST + b * s_sb + tt[:, None] * s_st + cc[None, :], mask=(in_state & rm)[:, None] & cm[None, :],
                other=0.0).to(tl.float32)
    x = tl.load(X + b * s_xb + (tt - CTX)[:, None] * s_xt + cc[None, :],
                mask=((~in_state) & rm)[:, None] & cm[None, :], other=0.0).to(tl.float32)
    if HAS_SNAKE:
        a = tl.load(A + cc, mask=cm, other=0.0).to(tl.float32)
        ib = tl.load(IB + cc, mask=cm, other=0.0).to(tl.float32)
        sv = tl.sin(x * a[None, :])
        x = x + ib[None, :] * sv * sv
    y = tl.where(in_state[:, None], s, x)
    tl.store(Y + b * s_yb + tt[:, None] * s_yt + cc[None, :], y.to(Y.dtype.element_ty), mask=rm[:, None] & cm[None, :])


def snake_cat(x, state, a, ib, out):
    """x: [B, T, C] (raw), state: [>=B, CTX, C] (activated context) -> out [B, CTX+T, C]."""
    B, T, C = x.shape
    CTX = state.shape[1] if state is not None else 0
    BC = min(128, triton.next_power_of_2(C))
    BT = 32
    grid = (B, triton.cdiv(CTX + T, BT), triton.cdiv(C, BC))
    _snake_cat_kernel[grid](x, state if state is not None else x, a if a is not None else x,
                            ib if ib is not None else x, out, T, C, CTX, x.stride(0), x.stride(1),
                            state.stride(0) if state is not None else 0, state.stride(1) if state is not None else 0,
                            out.stride(0), out.stride(1), HAS_SNAKE=a is not None, BT=BT, BC=BC, num_warps=4)
    return out
