#!/usr/bin/env python3
"""
Precompute the voices for the Audio8 engine: every <name>.wav (+ <name>.txt, its exact transcript) of a source folder
is encoded once with the checkpoint's 44.1 kHz codec into <out>/<name>.pt = {ref_code [T, 10] int16, ref_text}, so
the API never encodes reference audio at request time. The .txt is copied next to it and the .wav hard-linked (or
copied) so the folder is self-contained. A voice without a transcript is skipped: the model clones from reference
audio + its transcript only.

References are encoded in batches (the codec is causal: zeros after a shorter reference don't change its frames).

Usage:
    python create_speakers.py --src ../data/refs/speakers_06b_v3 --out ../data/speakers/speakers_a8
    python create_speakers.py --add /path/ref.wav --name pika_bac --ref-text "..."     # one voice into --out
"""
import argparse
import glob
import math
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch


def load_audio(path, sr):
    """Mono float32 at `sr` (the processor's _load_audio: channel mean, torchaudio resample)."""
    import soundfile as sf
    a, r = sf.read(str(path), dtype="float32", always_2d=True)
    a = torch.from_numpy(np.ascontiguousarray(a.mean(axis=1)))
    if int(r) != sr:
        from torchaudio.functional import resample
        a = resample(a, int(r), sr)
    return a.contiguous()


@torch.inference_mode()
def encode_refs(model, refs, max_batch_s=240.0, max_items=32):
    """refs: paths or 1-D float tensors at the codec rate -> list of LongTensor [T, 10] (cpu), in order."""
    sr = int(model.config.codec_sample_rate)
    fl = int(model.config.codec_frame_size)
    auds = [r if torch.is_tensor(r) else load_audio(r, sr) for r in refs]
    order = sorted(range(len(auds)), key=lambda i: auds[i].numel())
    out = [None] * len(auds)
    dev = model.device
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and j - i < max_items:
            n = auds[order[j]].numel()
            if j > i and n * (j - i + 1) > max_batch_s * sr:
                break
            j += 1
        grp = order[i:j]
        L = max(auds[k].numel() for k in grp)
        L = math.ceil(L / fl) * fl
        x = torch.zeros(len(grp), 1, L, dtype=torch.float32)
        lens = torch.tensor([auds[k].numel() for k in grp], dtype=torch.long)
        for r, k in enumerate(grp):
            x[r, 0, :auds[k].numel()] = auds[k]
        codes, clens = model.encode_audio(x.to(dev), lens.to(dev))        # [n, 10, T], [n]
        for r, k in enumerate(grp):
            out[k] = codes[r, :, :int(clens[r])].t().contiguous().long().cpu()
        i = j
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--model", default=os.path.join(root, "models", "Audio8-TTS-Realistic-0.6b"))
    ap.add_argument("--src", default=None, help="folder of <name>.wav + <name>.txt (default: --out itself)")
    ap.add_argument("--out", default=os.path.join(root, "data", "speakers", "speakers_a8"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true", help="rebuild even if .pt exists")
    ap.add_argument("--only", default=None, help="comma-separated name prefixes to build (default: all)")
    ap.add_argument("--add", default=None, help="path to a reference wav to import")
    ap.add_argument("--name", default=None, help="speaker name for --add")
    ap.add_argument("--ref-text", default=None, help="reference transcript for --add")
    args = ap.parse_args()

    src = args.src or args.out
    os.makedirs(args.out, exist_ok=True)
    if args.add:
        if not args.name or not args.ref_text:
            print("ERROR: --name and --ref-text are required with --add", file=sys.stderr)
            return 2
        shutil.copyfile(args.add, os.path.join(args.out, args.name + ".wav"))
        with open(os.path.join(args.out, args.name + ".txt"), "w", encoding="utf-8") as f:
            f.write(args.ref_text.strip())
        src = args.out
        print(f"[add] imported {args.add} -> {args.out}/{args.name}.wav")

    prefixes = [p.strip() for p in args.only.split(",")] if args.only else None
    todo = []
    for wav in sorted(glob.glob(os.path.join(src, "*.wav"))):
        name = os.path.splitext(os.path.basename(wav))[0]
        if args.add and name != args.name:
            continue
        if prefixes and not any(name.startswith(p) for p in prefixes):
            continue
        txt = os.path.join(src, name + ".txt")
        if not os.path.exists(txt):
            print(f"[skip] {name}: no transcript")
            continue
        if os.path.exists(os.path.join(args.out, name + ".pt")) and not args.force:
            continue
        with open(txt, encoding="utf-8") as f:
            todo.append((name, wav, f.read().strip()))
    if not todo:
        print(f"nothing to build in {args.out}")
        return 0

    from tts_engine import load_model
    print(f"loading {args.model} ...", flush=True)
    model, _ = load_model(args.model, args.device)
    t0 = time.perf_counter()
    ok = fail = 0
    step = 512
    for g in range(0, len(todo), step):
        grp = todo[g:g + step]
        try:
            codes = encode_refs(model, [w for _, w, _ in grp])
        except Exception:
            traceback.print_exc()
            fail += len(grp)
            continue
        for (name, wav, text), c in zip(grp, codes):
            try:
                out = os.path.join(args.out, name + ".pt")
                torch.save({"name": name, "ref_text": text, "ref_code": c.to(torch.int16),
                            "sample_rate": int(model.config.codec_sample_rate), "format": "audio8-voice-v1"},
                           out + ".tmp")
                os.replace(out + ".tmp", out)
                if os.path.abspath(src) != os.path.abspath(args.out):
                    shutil.copyfile(os.path.join(src, name + ".txt"), os.path.join(args.out, name + ".txt"))
                    dst = os.path.join(args.out, name + ".wav")
                    if not os.path.exists(dst):
                        try:
                            os.link(wav, dst)
                        except OSError:
                            shutil.copyfile(wav, dst)
                ok += 1
            except Exception:
                traceback.print_exc()
                fail += 1
        print(f"[speakers] {ok + fail}/{len(todo)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"Done: {ok} built, {fail} failed in {time.perf_counter() - t0:.0f}s -> {args.out}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
