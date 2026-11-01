#!/usr/bin/env python3
"""
Configuration via environment / .env files for the TTS web UI and API.

One settings file, <repo>/.env (every setting; .env.example, hand-edited, never rewritten by a script), loaded WITHOUT
overriding variables already set in the real environment:

    real environment (run_profile.sh's profile / KEY=VALUE for one start)  >  <repo>/.env  >  built-in defaults

Within the file the last line of a key wins (as run_profile.sh and docker compose read it). Docker passes the same
.env as env_file: there it is the real environment.
"""
from __future__ import annotations
import os
from typing import Any, Dict

SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SERVE_DIR)
DEFAULT_MODEL = os.path.join(ROOT_DIR, "models", "Audio8-TTS-Realistic-0.6b")
DEFAULT_SPEAKERS = os.path.join(ROOT_DIR, "data", "speakers", "speakers_a8")


def resolve(path: str) -> str:
    """A path setting: absolute as is, relative to serve/ (where the API runs and the profiles' paths start)."""
    return path if not path or os.path.isabs(path) else os.path.normpath(os.path.join(SERVE_DIR, path))


def _load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    found: Dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                # quoted value: take what's inside the quotes (preserve '#' inside)
                if v[:1] in ('"', "'"):
                    q = v[0]
                    end = v.find(q, 1)
                    v = v[1:end] if end != -1 else v[1:]
                else:
                    # strip inline comment (e.g.  KEY=true   # note)
                    v = v.split("#", 1)[0].strip()
                if k:
                    found[k] = v                # the last line of a key wins
    except Exception as e:
        print(f"[config] warning: could not read {path}: {e}", flush=True)
    for k, v in found.items():
        os.environ.setdefault(k, v)             # never over the real environment / an earlier file


_load_env_file(os.path.join(ROOT_DIR, ".env"))
if os.path.exists(os.path.join(SERVE_DIR, ".env")):     # the old layout's generated profile (before 2026-10-22)
    print(f"[config] warning: {os.path.join(SERVE_DIR, '.env')} is not read: settings go in {ROOT_DIR}/.env", flush=True)


def _b(key: str, default: bool) -> bool:
    return str(os.getenv(key, str(default))).strip().lower() in ("1", "true", "yes", "on")


def _f(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return float(default)


def _i(key: str, default: int) -> int:
    try:
        return int(float(os.getenv(key, default)))
    except (TypeError, ValueError):
        return int(default)


def _s(key: str, default: str) -> str:
    return os.getenv(key, default)


def gen_params() -> Dict[str, Any]:
    """Decoding parameters (the model card's: temperature 0.8, top-p 0.95, top-k 50; one set for the semantic token and
    the codebooks, as the checkpoint's generate; RAS comes from the checkpoint's config)."""
    return dict(
        do_sample=_b("TTS_DO_SAMPLE", True),
        temperature=_f("TTS_TEMPERATURE", 0.8),
        top_k=_i("TTS_TOP_K", 50),
        top_p=_f("TTS_TOP_P", 0.95),
        max_new_tokens=_i("TTS_MAX_NEW_TOKENS", 2048),
    )


def server_cfg() -> Dict[str, Any]:
    """Server / engine defaults (CLI flags still override these)."""
    return dict(
        # the 0.6B by default (user 2026-10-20: "mặc định là 0.6b cho đỡ phải chỉnh"): its weights in <repo>/models, its
        # voices in <repo>/data/speakers
        model=_s("TTS_MODEL_PATH", DEFAULT_MODEL),
        device=_s("TTS_DEVICE", "cuda:0"),
        speakers_dir=_s("TTS_SPEAKERS_DIR", DEFAULT_SPEAKERS),
        # physical GPU ids, one engine worker each (e.g. "1,2"); empty -> one worker on TTS_DEVICE
        gpus=[g.strip() for g in _s("TTS_GPUS", "").split(",") if g.strip()],
        batch_size=_i("TTS_BATCH_SIZE", 16),      # slots (max concurrent streams) per worker
        attn=_s("TTS_ATTN", "sdpa"),
        host=_s("TTS_HOST", "0.0.0.0"),
        port=_i("TTS_PORT", 29100),
        ui_port=_i("TTS_UI_PORT", 29101),
        out_sr=_i("TTS_OUT_SR", 16000),
        frames_per_chunk=_i("TTS_FRAMES_PER_CHUNK", 6),   # streamed chunk size (after the ramp)
        max_text_chars=_i("TTS_MAX_TEXT_CHARS", 600),
    )
