#!/usr/bin/env python3
"""
Production TTS API (FastAPI) for Audio8-TTS-Realistic-0.6b voice clone (the Queen TTS API), served by the fast engine.

Endpoints:
    GET|POST /v2/speech/dual_vietnamese_english   # Vietnamese + English in one voice family
             (aliases, same behaviour: /v2/speech/vietnamese, /v2/speech/english)
        ?text=Từ apple nghĩa là quả táo. Now repeat after me!   # tags optional, languages detected
        &speaker=pika-2          # a voice family (GET /v2/speakers?group=true); default TTS_DUAL_SPEAKER;
                                 # simp_en_normal / simp_en_slow_3: pika-pitch-2, English normal / slow
        &accent=north|south      # default north (Vietnamese numbers / units read north or south)
        &speed=normal|slow       # English spans, default normal
        &speed_vi=normal|slow    # Vietnamese spans, default normal
        &ctx=prev|full|auto|none # default prev
        &instruct=happy          # a preset key of GET /v2/instructs, or any instruction text (instructs.py)
        &stream=false|true       # `lang` ignored; an unknown `speaker` reads with the default (X-Speaker-Note); TTS_DUAL_FALLBACK
                                 # (default pika-2) reads the languages the family has no voice in; headers X-Voices, X-Langs

    GET /v2/normalize?text=...&accent=north|south   # what the TTS will read (vi_normalize)

    GET|POST /v2/multilang       # any language, tags optional: every span's language detected
        ?text=Tiếng Pháp là merci. 日本語では「ありがとう」。 [en_slow] Thank you! Giỏi lắm
        &speaker=pika-1&speed=...&accent=...&ctx=...&stream=...&langs=vi,en,fr (Latin choices)
                                 # (multilang.py); tags [vi] [en] [zh] [ja] [ko] [fr] [de] [es] [pt] [ru] [it]
                                 # (+ _speed / _accent): a tag opens its language, its speed / accent hold
                                 # for that language in the tagged part; each span gets the best voice
                                 # {speaker}_{lang}[_speed][_accent]; ctx=prev|full|auto|none (default prev);
                                 # headers X-Voices, X-Langs (the voice and language tag of each span)
    (/v2/speech/codeswitch and /v2/auto_vi_en were removed on 2026-10-21, unused: /v2/multilang reads both
    their texts, tagged or not)
    GET /v2/multilang/detect?text=...&langs=...   # the spans /v2/multilang would use
    GET /v2/sfx                  # the non-verbal sounds: <giggle> / <giggle:hihi> in any text of the
                                 # speaker-family endpoints, read by {family}_sfx_{sound} (sfx.py); a family
                                 # without that sound skips it (header X-Sfx-Note)

    Every speech endpoint above (and bot/tts, /v2/preview) takes `instruct`: a preset key ("happy",
    "whisper", "storyteller", ... GET /v2/instructs) or free instruction text, for the whole line (needs the
    instruct model, TTS_INSTRUCT=1; header X-Instruct = the preset key, "custom" or "none").
    GET /v2/instructs            # the instruction presets by group (emotion, delivery, persona)

    Every speech endpoint also applies the pronunciation lexicon (serve/lexicon.json, lexicon.py): a word
    respelled for the language of the span it is in (Pika -> pi ka in Vietnamese, Pee-kah in English, ピカ in
    Japanese: [en] Hi, I'm Pika! -> [en] Hi, I'm Pee-kah!, no switch of voice), or "A -> b read in language L".
    GET /v2/lexicon              # its entries

    GET /v2/languages            # tag, name, preview sample of every language
    GET /v2/preview?speaker=anlee&lang=ja&speed=&accent=&text=&instruct=   # voice preview, WAV, cached; a style reads its own line
    GET /ui/                     # the web UI (serve/ui), when present

    POST /personalized-ai-coach/api/v1/bot/tts   # robot contract, returns an audio_url (bot_tts.py)
        {"text": "Wow! Mình cùng khám phá hang động nhé. See you!", "speaker": "pika-2", "speed": "slow",
         "speed_vi": "normal", "accent": "north", "ctx": "prev", "instruct": "excited", "use_cache": true}
                                 # like dual_vietnamese_english: one voice family (`speaker`, default
                                 # TTS_DUAL_SPEAKER), vi / en detected (tags optional)

    stream=false -> 200 audio/wav            (16 kHz, mono, PCM16)
    stream=true  -> 200 audio/pcm; rate=16000; channels=1   (raw little-endian s16, chunked)

This process holds no model: generation runs in one engine-worker process per GPU (see
engine_worker.fast_main); this event loop only relays bytes.

Robustness:
  - input validation -> 400; unknown speaker -> 404; generation error -> 500 (json or aborted stream)
  - a client that disconnects mid-stream is cancelled on the worker (its slot is freed)
  - /health and /v2/speakers for ops.

Run:
    python production_api.py --speakers-dir speakers --host 0.0.0.0 --port 29100
"""
import argparse
import asyncio
import gzip
import logging
import json
import os
import sys
import time

# Expandable CUDA segments keep reserved-but-unused memory down. Set before the engine workers are
# spawned so they inherit it before importing torch. Real env overrides.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# A finer GIL switch interval keeps the asyncio reader threads responsive. Env-overridable.
sys.setswitchinterval(float(os.environ.get("TTS_GIL_SWITCH_S", "0.0005")))

from typing import Optional, Union

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
import uvicorn

import queue as _queue
from collections import OrderedDict

import numpy as np
import multiprocessing as mp
import threading
from types import SimpleNamespace

import config    # first: loads .env into os.environ, which bot_tts / instructs read at import
import auth
import bot_tts
import instructs
import level
import lexicon
import chorus
import effects
import elongate
import music
import observability as obs   # Datadog: metric events + a request line per synthesis (stdout JSON, Spark's contract)
import sfx
import timing
from engine_worker import fast_main, post_main, READY, CHAIN
from vi_normalize import normalize_vi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tts-api")

_CFG = config.server_cfg()
ENGINE: "EngineClient" = None
OUT_SR = _CFG["out_sr"]
MAX_TEXT_CHARS = _CFG["max_text_chars"]

app = FastAPI(title="Audio8 TTS API", description="Audio8-TTS-Realistic-0.6b serving (the Queen TTS API): voices, languages, emotions (`instruct`). Web UI at /ui/")


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _wav_bytes(pcm: bytes, sr: int) -> bytes:
    import io
    import soundfile as sf
    y = np.frombuffer(pcm, dtype=np.int16) if pcm else np.zeros(0, np.int16)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.get("/health")
async def health(request: Request):
    """The status; with a valid API key (or the admin session) also the voices and limits (auth.py)."""
    ok = {"status": "ok" if ENGINE is not None else "loading"}
    if not getattr(request.state, "auth_ok", False):
        return ok
    names = ENGINE.list_speakers() if ENGINE else []
    return {**ok, "speakers": len(_speaker_view(names)["families"]), "voices": len(names),
            "out_sample_rate": OUT_SR, "max_text_chars": MAX_TEXT_CHARS}


_VIEW: dict = {"names": None, "res": None}


def _speaker_view(names) -> dict:
    """The speakers as a client picks them: family names, and per family what each language offers (speed / accent
    variants, styles, sounds) -- never the internal voice names ({family}_{lang}_{speed}_{accent}_{style}: 19k of
    them; user 2026-10-11 "chỉ trả ra các speaker name thôi"). Built once per voice list (fixed for the run)."""
    if _VIEW["names"] is names:
        return _VIEW["res"]
    g = bot_tts.group_voices(names)
    fams = {}
    for f, langs in g["families"].items():
        fams[f] = {lang: sorted({v[len(f) + len(lang) + 2:] or "default" for v in vs}) for lang, vs in langs.items()}
    _VIEW.update(names=names, res={"families": fams, "styles": g["styles"], "style_variants": g["style_variants"],
                                   "sfx": g["sfx"]})
    return _VIEW["res"]


_SLOW_NEW = {"path": None, "mtime": None, "families": []}


def _slow_new() -> list:
    """The families whose slow voices are all the new ones (user 2026-10-13, P4: ElevenLabs slow reference at tempo
    0.8 + the output at 0.8, tempo.py; "trên UI những voice nào đã dc ok bản mới rồi thì đánh màu xanh nhạt"): the
    file <speakers dir>.slow_p4.json {"families": [...]} (work/p4_status.py writes it), read again when it changes."""
    p = _SLOW_NEW["path"]
    try:
        m = os.path.getmtime(p) if p else None
    except OSError:
        m = None
    if m is None:
        return []
    if m != _SLOW_NEW["mtime"]:
        try:
            with open(p, encoding="utf-8") as f:
                _SLOW_NEW.update(mtime=m, families=list(json.load(f).get("families", [])))
        except (OSError, ValueError):
            _SLOW_NEW.update(mtime=m, families=[])
    return _SLOW_NEW["families"]


_SPEAKERS_JSON: dict = {"view": None, "slow": None, "raw": b"", "gz": b""}


@app.get("/v2/speakers")
async def speakers(request: Request,
                   group: bool = Query(False, description="also what each speaker offers: languages with their "
                                                          "speed / accent variants, styles, sounds")):
    """The speaker names (what `speaker=` takes). group=true: {"count", "families": {speaker: {lang: [variant, ...]}},
    "styles": {speaker: [style, ...]}, "style_variants": {speaker: {lang: {variant: [style, ...]}}}, "sfx": {speaker:
    [sound, ...]}, "slow_new": [speaker, ...]}; a variant is the speed / accent the language has ("normal_north",
    "slow"; "default" = one voice); style_variants: the styles each of those has (a style is read only in the
    languages and variants listed); slow_new: the speakers whose slow voices are all the new, slower ones.
    group=true is gzipped for a client that accepts it (~250 KB of JSON, the /ui/ page loads it once)."""
    v = _speaker_view(ENGINE.list_speakers() if ENGINE else [])
    if not group:
        return {"speakers": list(v["families"])}
    slow = [f for f in _slow_new() if f in v["families"]]
    if _SPEAKERS_JSON["view"] is not v or _SPEAKERS_JSON["slow"] != slow:     # built once per voice list
        raw = json.dumps({"count": len(v["families"]), **v, "slow_new": slow}, ensure_ascii=False,
                         separators=(",", ":")).encode()
        _SPEAKERS_JSON.update(view=v, slow=slow, raw=raw, gz=gzip.compress(raw, 6))
    if "gzip" in request.headers.get("accept-encoding", "").lower():
        return Response(_SPEAKERS_JSON["gz"], media_type="application/json",
                        headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})
    return Response(_SPEAKERS_JSON["raw"], media_type="application/json", headers={"Vary": "Accept-Encoding"})


@app.get("/v2/sfx")
async def sounds():
    """The non-verbal sounds a text may hold (<giggle>, <giggle:hihi>; sfx.py): key, name, the text it reads by
    default, and the voice families that have a voice for each ({family}_sfx_{key})."""
    names = ENGINE.list_speakers() if ENGINE else []
    return {"sounds": sfx.catalog(), "families": bot_tts.group_voices(names)["sfx"]}


@app.get("/v2/effects")
async def effects_list():
    """What a text may do to its audio (effects.py): effects as a tag part ([radio], [yuna_cave], [en_echo_sad], held
    to the next tag), and the inline marks <pause:800ms>, <slow>...</slow> / <slow:0.6>, <em>...</em>."""
    return {"effects": effects.FX,
            "marks": {"<pause>": "silence: <pause> 0.5 s, <pause:800ms>, <pause:1.5s> (at most 5 s)",
                      "<slow>...</slow>": f"the words slowed, the voice and its context kept: tempo "
                                          f"{effects.SLOW_WORD_TEMPO:g} (<slow:0.6> for another, 0.4-1)",
                      "<em>...</em>": "the words stressed: longer, higher, a little louder"}}


@app.get("/v2/lexicon")
async def lexicon_entries():
    """The pronunciation lexicon every speech endpoint applies: [{"from", "to", "lang"}] (lexicon.json,
    TTS_LEXICON_FILE; re-read when the file changes). `to` a dict {lang: respelling}: the word stays in the
    language of its span, respelled for it (a language not listed reads it as written); `to` a string: read in
    `lang` (null = the span's own)."""
    try:
        return {"file": os.path.basename(lexicon.PATH), "entries": lexicon.entries()}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"lexicon: {e}")


@app.get("/v2/instructs")
async def instruct_presets():
    """The `instruct` presets (key, Vietnamese label, instruction text) by group, and whether this
    server's model follows instructions at all."""
    return instructs.catalog()


_INSTRUCT_DOC = ("a preset key of GET /v2/instructs (happy, sad, whisper, storyteller, ...) or any instruction "
                 "text, e.g. 'Speak like a pirate'; default none")


def _instruct(instruct):
    """`instruct` -> (header label, instruction text for the engine or None, style for the voice choice or
    None: TTS_INSTRUCT_MODE=voices, instructs.for_engine); 400 when it can't be used."""
    try:
        key, text = instructs.resolve(instruct)
        text, style = instructs.for_engine(key, text)
    except instructs.InstructError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return key or "none", text, style


# /v2/speech/dual_vietnamese_english (and bot/tts) speak with the voice family `speaker` names, this one when it
# names none (TTS_DUAL_SPEAKER; TTS_DUAL_FALLBACK, default pika-2, for languages the family lacks)
DUAL_SPEAKER = bot_tts.DUAL_SPEAKER


class DualRequest(BaseModel):
    """POST body of /v2/speech/dual_vietnamese_english: the GET parameters as JSON."""
    text: Optional[str] = None
    speaker: Optional[str] = None       # a voice family; default DUAL_SPEAKER; simp_en_normal / simp_en_slow_3 too
    lang: Optional[str] = None          # accepted, ignored (every span's language is detected)
    accent: Optional[str] = None
    lora: Optional[str] = None          # north | south: the accent, before `accent` (user 2026-10-20)
    speed: Optional[str] = None         # the English spans (default normal)
    speed_vi: Optional[str] = None      # the Vietnamese spans (default normal)
    ctx: Optional[str] = None
    instruct: Optional[str] = None
    stream: Optional[bool] = False
    trace_id: Optional[str] = None      # for the logs only (observability.py, as streaming-spark): GET ?trace_id=
    robot_id: Optional[str] = None      # also header x-robot-id


@app.get("/v2/speech/dual_vietnamese_english")
@app.get("/v2/speech/vietnamese")           # aliases (user, 2026-10-05): the same endpoint
@app.get("/v2/speech/english")
async def synth(
    request: Request,
    text: str = Query(..., description="Vietnamese and English mixed freely; tags ([vi] [en] [en_normal] ...) optional"),
    speaker: Optional[str] = Query(None, description=f"voice family (GET /v2/speakers?group=true); default {DUAL_SPEAKER}"),
    lang: Optional[str] = Query(None, description="ignored: every span's language is detected"),
    accent: Optional[str] = Query(None, description="north (default) | south: how Vietnamese numbers / units are read"),
    lora: Optional[str] = Query(None, description="north | south: the accent (before `accent`)"),
    speed: Optional[str] = Query(None, description="English spans: normal (default) | slow; simp_en_normal / "
                                                   "simp_en_slow_3 as speaker say it too"),
    speed_vi: Optional[str] = Query(None, description="Vietnamese spans: normal (default) | slow"),
    ctx: Optional[str] = Query(None, description="prev (default) | full | auto | none"),
    instruct: Optional[str] = Query(None, description=_INSTRUCT_DOC),
    stream: str = Query("false"),
):
    return await _dual(request, text, speaker, accent, speed, speed_vi, ctx, _truthy(stream), instruct, lora)


@app.post("/v2/speech/dual_vietnamese_english")
@app.post("/v2/speech/vietnamese")
@app.post("/v2/speech/english")
async def synth_post(payload: DualRequest, request: Request):
    obs.stash(request, payload)
    return await _dual(request, payload.text, payload.speaker, payload.accent, payload.speed, payload.speed_vi,
                       payload.ctx, bool(payload.stream), payload.instruct, payload.lora)


async def _dual(request, text, speaker, accent, speed, speed_vi, ctx, do_stream, instruct=None, lora=None):
    """Vietnamese / English (tags optional, the languages detected as in /v2/multilang, Latin choices
    vi / en), voiced by the family `speaker` (bot_tts.dual_family: a voice name reads with its family, none or
    an unknown one with DUAL_SPEAKER, header X-Speaker-Note; TTS_DUAL_LOCK: the accent's family): English spans at
    `speed` (default normal; a simp_en_normal / simp_en_slow_3 speaker's own wins), Vietnamese spans at `speed_vi`
    (default normal; user 2026-10-22), Vietnamese read with `lora` / `accent` (default north), each span
    continuing from the previous one's audio (`ctx`, default prev), `instruct` default TTS_DUAL_INSTRUCT_DEFAULT.
    A tag's own speed wins for its span."""
    accent = bot_tts.dual_accent(lora, accent)
    family, note = bot_tts.dual_family(speaker, ENGINE.list_speakers() if ENGINE is not None else [], accent)
    if note:
        log.warning("dual %s", note)
    en_speed = bot_tts.simp_speaker(speaker)[1] or speed or "normal"
    resp = await _codeswitch(request, text, family, "normal", accent, ctx or "prev", do_stream, langs="vi,en",
                             lang_speed={"en": en_speed, "vi": speed_vi or "normal"}, label="dual",
                             fallback=bot_tts.DUAL_FALLBACK, instruct=bot_tts.dual_instruct(instruct, speaker))
    if note:
        resp.headers["X-Speaker-Note"] = note
    return resp


_VI_LANGS = {"vietnamese", "vi", "vie", "vn", "tiếng việt"}


@app.get("/v2/normalize")
async def normalize(text: str = Query(...), lang: str = Query("vi"), accent: str = Query("north")):
    """What the TTS will read for this text (Vietnamese normalization), for checking/debugging."""
    if (lang or "").strip().lower() not in _VI_LANGS | {"auto"}:
        return {"text": text, "normalized": False}
    return {"text": normalize_vi(text, accent=accent if accent in ("north", "south") else "north"),
            "normalized": True}


def _held(marks):
    """The drawn-out words of a text (elongate.collapse) as the detect endpoint shows them: the word read, held
    for how long."""
    return [{"word": m["word"], "seconds": round(min(elongate.ELONG_MAX_S, m["extra"] * elongate.ELONG_S), 2)}
            for m in marks]


class MultilangRequest(BaseModel):
    """POST body of /v2/multilang: the GET parameters as JSON; `langs` "vi,en,fr" or a list."""
    text: Optional[str] = None
    speaker: Optional[str] = None
    speed: Optional[str] = None
    accent: Optional[str] = None
    vi_accent: Optional[str] = None
    ctx: Optional[str] = None
    instruct: Optional[str] = None
    stream: Optional[bool] = False
    langs: Optional[Union[str, list]] = None
    trace_id: Optional[str] = None      # for the logs only (observability.py)
    robot_id: Optional[str] = None


_MULTI_TEXT = ("text in any language the model speaks (" + " ".join(bot_tts.LANGS) + "), tags optional: "
               "'Tiếng Pháp là merci. 日本語では「ありがとう」。 [en_slow] Thank you! Giỏi lắm'")


@app.get("/v2/multilang")
async def synth_multilang(
    request: Request,
    text: str = Query(..., description=_MULTI_TEXT),
    speaker: Optional[str] = Query(None, description="voice family: each span gets {speaker}_{lang}[_speed][_accent]"),
    speed: Optional[str] = Query(None, description="normal | slow (fast = normal); a tag's own speed wins for its span"),
    accent: Optional[str] = Query(None, description="north | south, for the Vietnamese spans"),
    vi_accent: Optional[str] = Query(None, description="alias of accent"),
    ctx: Optional[str] = Query(None, description="auto | prev | full | none"),
    instruct: Optional[str] = Query(None, description=_INSTRUCT_DOC),
    stream: str = Query("false"),
    langs: Optional[str] = Query(None, description="Latin-script languages to choose among, e.g. 'vi,en,fr' "
                                                   "(default all); scripts (zh ja ko ru) are always detected"),
):
    return await _codeswitch(request, text, speaker, speed, accent or vi_accent, ctx, _truthy(stream),
                             langs=langs, instruct=instruct)


@app.post("/v2/multilang")
async def synth_multilang_post(payload: MultilangRequest, request: Request):
    obs.stash(request, payload)
    return await _codeswitch(request, payload.text, payload.speaker, payload.speed,
                             payload.accent or payload.vi_accent, payload.ctx, bool(payload.stream),
                             langs=payload.langs, instruct=payload.instruct)


@app.get("/v2/multilang/timed")
async def synth_multilang_timed(
    request: Request,
    text: str = Query(..., description=_MULTI_TEXT),
    speaker: Optional[str] = Query(None, description="voice family"),
    speed: Optional[str] = Query(None, description="normal | slow"),
    accent: Optional[str] = Query(None, description="north | south"),
    ctx: Optional[str] = Query(None, description="auto | prev | full | none"),
    instruct: Optional[str] = Query(None, description=_INSTRUCT_DOC),
    stream: str = Query("false", description="true: NDJSON events (audio chunks, segment, words, visemes, end)"),
    langs: Optional[str] = Query(None),
    sync: str = Query("false", description="stream: true -> each segment's words + visemes before its audio "
                                           "(in sync for a player; first audio later)"),
):
    """/v2/multilang with word timings and mouth shapes (timing.py): stream=false -> JSON {"sample_rate",
    "duration", "audio_wav_base64", "segments", "words", "visemes"}; stream=true -> NDJSON events as they come
    (sync=true: a segment's timings first, then its audio)."""
    return await _codeswitch(request, text, speaker, speed, accent, ctx, _truthy(stream),
                             langs=langs, instruct=instruct, timed=True, sync=_truthy(sync))


class TimedRequest(MultilangRequest):
    """POST body of /v2/multilang/timed: + `sync` (stream: each segment's timings before its audio)."""
    sync: Optional[bool] = False


@app.post("/v2/multilang/timed")
async def synth_multilang_timed_post(payload: TimedRequest, request: Request):
    obs.stash(request, payload)
    return await _codeswitch(request, payload.text, payload.speaker, payload.speed,
                             payload.accent or payload.vi_accent, payload.ctx, bool(payload.stream),
                             langs=payload.langs, instruct=payload.instruct, timed=True, sync=bool(payload.sync))


async def _read_aligned(text, speaker, instruct, chars=False):
    """`text` read once (multilang, normal speed, `instruct`'s style) and its words found in the audio:
    (float32 audio at OUT_SR, [{"word", "start", "end"[, "chars"]}])."""
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    ins_key, ins_text, style = _instruct(instruct)
    try:
        segs = bot_tts.build_plan(text, speaker, "normal", "north", "prev", ENGINE.list_speakers(), MAX_TEXT_CHARS,
                                  multi=True, style=style, notes=[])
    except bot_tts.BotTtsError as e:
        raise HTTPException(status_code=e.status, detail=e.msg)
    rid, q = ENGINE.submit_plan(segs, "prev", ins_text, marks=True)
    pcm, words = [], []
    async for ev in _timed_events(rid, q, segs, chars=chars):
        if ev["type"] == "audio":
            pcm.append(ev["pcm"])
        elif ev["type"] == "words":
            words += ev["items"]
        elif ev["type"] == "error":
            raise HTTPException(status_code=500, detail=ev["detail"])
    y = np.frombuffer(b"".join(pcm), np.int16).astype(np.float32) / 32768.0
    return y, sorted(words, key=lambda w: w["start"])


def _wav_float(y):
    return Response(content=_wav_bytes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes(), OUT_SR),
                    media_type="audio/wav", headers={"X-Audio-Duration": f"{len(y) / OUT_SR:.3f}"})


class SingRequest(BaseModel):
    """POST /v2/sing: a tune (GET /v2/sing/tunes) and/or your lyrics + notes."""
    tune: Optional[str] = None          # twinkle | happy_birthday | kia_con_buom_vang
    lyrics: Optional[str] = None        # syllables: "Twin-kle twin-kle", "_" holds one more note
    notes: Optional[str] = None         # "C4/1 D4/0.5 R/1 ..." (one per syllable, rests free)
    bpm: Optional[float] = None
    key: Optional[Union[str, int]] = "auto"   # semitones to move the tune, or auto (to the voice)
    speaker: Optional[str] = None
    instruct: Optional[str] = "happy"
    accompany: bool = True


class ChantRequest(BaseModel):
    """POST /v2/chant: the text's words on the beat over a drum loop."""
    text: str
    bpm: float = 100
    speaker: Optional[str] = None
    instruct: Optional[str] = "excited"
    beat: bool = True
    count_in: bool = True


@app.get("/v2/sing/tunes")
async def sing_tunes():
    """The tunes /v2/sing knows: name, tempo, notes, the lyrics they come with."""
    return music.TUNES


def _music_synth(request, text, speaker, y_fn):
    """A whole-file music line (sing / chant) logged as a synthesis (observability.Synth): y_fn() -> float audio."""
    syn = obs.Synth(request, speaker=speaker or bot_tts.DEFAULT_SPEAKER, langs=["multi"], stream=False, lora=None,
                    input_text=text, normalized_text=[text], sr=OUT_SR)

    async def run():
        try:
            y = await y_fn()
        except Exception as e:
            syn.done("error", error=str(getattr(e, "detail", e)), error_type=type(e).__name__)
            raise
        syn.audio(2 * len(y))
        syn.done()
        return _wav_float(y)
    return run()


@app.post("/v2/sing")
async def sing(payload: SingRequest, request: Request):
    """Pika sings (music.py): the lyrics read, each syllable stretched and tuned to its note (PSOLA, an autotuned
    robot), a music box under it. A WAV."""
    tune = music.TUNES.get(payload.tune) if payload.tune else None
    if payload.tune and tune is None:
        raise HTTPException(status_code=400, detail=f"unknown tune {payload.tune!r}; tunes: {sorted(music.TUNES)}")
    lyrics = payload.lyrics or (tune or {}).get("lyrics")
    spec = payload.notes or (tune or {}).get("notes")
    if not lyrics or not spec:
        raise HTTPException(status_code=400, detail="give a tune, or lyrics and notes")
    try:
        notes = music.parse_notes(spec)
        music.score(lyrics, notes)
        key = "auto" if str(payload.key).lower() == "auto" else int(payload.key)
    except (music.MusicError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    bpm = float(payload.bpm or (tune or {}).get("bpm") or 100)
    if not 40 <= bpm <= 240:
        raise HTTPException(status_code=400, detail="bpm: 40 to 240")
    read = " ".join(w.replace("-", "") for w in lyrics.split() if w.strip("_.,!?;:"))

    async def song():
        y, words = await _read_aligned(read, payload.speaker, payload.instruct, chars=True)
        return await asyncio.to_thread(music.sing, y, OUT_SR, words, lyrics, notes, bpm, key, payload.accompany)
    return await _music_synth(request, lyrics, payload.speaker, song)


@app.post("/v2/chant")
async def chant(payload: ChantRequest, request: Request):
    """Pika chants (music.py): the text's words put on the beat (bpm), a drum loop with a count-in. A WAV."""
    if not 40 <= payload.bpm <= 240:
        raise HTTPException(status_code=400, detail="bpm: 40 to 240")

    async def beat():
        y, words = await _read_aligned(payload.text, payload.speaker, payload.instruct)
        if not words:
            raise HTTPException(status_code=422, detail="no words found in the reading (a language without Latin letters?)")
        return await asyncio.to_thread(music.chant, y, OUT_SR, words, payload.bpm, payload.beat, payload.count_in)
    return await _music_synth(request, payload.text, payload.speaker, beat)


@app.get("/v2/multilang/detect")
async def detect_multilang(text: str = Query(...), langs: Optional[str] = Query(None)):
    """How /v2/multilang splits a text: [{"lang", "attrs", "text"}] (after the pronunciation lexicon: what is
    read) and the same as tagged text; held: the drawn-out words ("vàooooo": "vào" held, elongate.py)."""
    text, held = elongate.collapse(bot_tts.en_normalize.expand_abbrev(text), bot_tts._TAG)
    try:
        spans = lexicon.apply(bot_tts.multi_spans(text, bot_tts.parse_langs(langs), _families()))
    except bot_tts.BotTtsError as e:
        raise HTTPException(status_code=e.status, detail=e.msg)

    def tag(lang, a):
        parts = ([a["speaker"]] if a.get("speaker") else []) + ([lang] if lang else []) + \
            [a[k] for k in ("speed", "accent") if k in a] + ([a["style"] or "plain"] if "style" in a else [])
        return "[" + "_".join(parts) + "]"

    def tagged(s):
        if s["lang"] == "sfx":
            return f"<{s['attrs']['sound']}:{s['text']}>"
        if s["lang"] == "chorus":
            return "+".join(tag(m.get("lang"), m) for m in s["attrs"]["chorus"]) + " " + s["text"] + " [end]"
        return tag(s["lang"], s["attrs"]) + " " + s["text"]
    return {"spans": spans, "tagged": " ".join(tagged(s) for s in spans), "held": _held(held)}


def _families() -> set:
    """The speaker names a tag may use ([yuna_vi_sad]): the voice families."""
    return bot_tts.families_of(ENGINE.list_speakers()) if ENGINE else set()


async def _codeswitch(request, text, speaker, speed, accent, ctx, do_stream, langs=None,
                      lang_speed=None, label=None, fallback=None, instruct=None, timed: bool = False,
                      sync: bool = False):
    """A multi-language line (/v2/multilang, /v2/multilang/timed, the dual endpoints) voiced as one code-switch
    chain: a WAV, or a raw PCM stream with stream=true. Every language is detected (multilang.py), tags optional:
    a tag ([vi] [en] [zh] [ja] [ko] [fr] [de] [es] [pt] [ru] [it], 3-letter codes too, with speed / accent
    attributes: [en_normal], [vi_south]) opens its language and its attributes hold for that language's spans in
    the tagged part, the languages after it are still detected; `langs` limits the Latin-script choices.
    Voices the bot/tts way: `speaker` (default BOT_TTS_DEFAULT_SPEAKER) + `speed` + `accent` pick each span's voice
    {speaker}_{lang}[_speed][_accent], a tag's own attributes winning; Vietnamese and English spans are
    normalized. lang_speed: a speed per language tag over `speed`; fallback: the family for languages `speaker`
    has no voice in (bot_tts.build_plan); instruct: a preset key or instruction text for the whole line
    (instructs.py). `ctx`: prev (default, BOT_TTS_CTX; the previous span's audio) | full (all previous spans) |
    auto (prev, but none across a north/south change and after a short span) | none."""
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    if not (instruct or "").strip():                   # a speaker name's own style (simp_en_slow_3: excited)
        instruct = bot_tts.speaker_instruct(speaker) or instruct
    ins_key, ins_text, style = _instruct(instruct)
    ctx = (ctx or bot_tts.CTX_POLICY).strip().lower()
    if ctx not in ("auto", "prev", "full", "none"):
        raise HTTPException(status_code=400, detail="`ctx` must be auto, prev, full or none")
    notes = []
    t_plan = time.perf_counter()
    try:   # full = every previous span as context (planned like prev: no span starts without one)
        segs = bot_tts.build_plan(text, speaker, speed, accent, "prev" if ctx == "full" else ctx,
                                  ENGINE.list_speakers(), MAX_TEXT_CHARS, multi=True, langs=langs,
                                  lang_speed=lang_speed, fallback=fallback, style=style, notes=notes)
    except bot_tts.BotTtsError as e:
        raise HTTPException(status_code=e.status, detail=e.msg)
    t0 = time.perf_counter()
    rid, q = ENGINE.submit_plan(segs, "full" if ctx == "full" else "prev", ins_text, marks=timed)
    voices = ",".join(s["speaker"] + ("/ctx=none" if s.get("ctx") else "") for s in segs)
    syn = obs.Synth(request, speaker=speaker or "", langs=[s["tag"] for s in segs], stream=do_stream, lora=accent,
                    input_text=text, normalized_text=[s["text"] for s in segs], sr=OUT_SR,
                    norm_ms=(t0 - t_plan) * 1000.0, segments=len(segs), voices=voices, instruct=ins_key)
    label = (label or "multilang") + ":" + voices
    if timed:
        resp = await _respond_timed(request, rid, q, segs, do_stream, t0,
                                    label + f" instruct={ins_key} timed" + (" sync" if sync else ""), sync=sync, syn=syn)
    else:
        resp = await _respond(request, rid, q, do_stream, t0, label + f" instruct={ins_key}", len((text or "").strip()),
                              syn=syn)
    resp.headers["X-Voices"] = voices
    resp.headers["X-Langs"] = ",".join(s["tag"] for s in segs)
    resp.headers["X-Instruct"] = ins_key
    if notes:
        resp.headers["X-Sfx-Note"] = "; ".join(notes)
    return resp


bot_tts.register(app, lambda: ENGINE, lambda: OUT_SR, MAX_TEXT_CHARS)   # POST .../bot/tts (+ audio link)


# ------------------------------------------------------------------ languages + voice preview (web UI)
LANGUAGES = [   # tag, name, a short sample line for the voice preview
    ("vi", "Tiếng Việt", "Xin chào, tớ là Pika. Mình cùng học nhé!"),
    ("en", "English", "Hi, I'm Pika. Let's learn together!"),
    ("zh", "中文", "你好，我是皮卡。我们一起学习吧！"),
    ("ja", "日本語", "こんにちは、ピカだよ。一緒に勉強しよう！"),
    ("ko", "한국어", "안녕, 나는 피카야. 같이 공부하자!"),
    ("fr", "Français", "Salut, je suis Pika. Apprenons ensemble !"),
    ("de", "Deutsch", "Hallo, ich bin Pika. Lass uns zusammen lernen!"),
    ("es", "Español", "¡Hola, soy Pika! ¡Aprendamos juntos!"),
    ("pt", "Português", "Olá, eu sou o Pika. Vamos aprender juntos!"),
    ("ru", "Русский", "Привет, я Пика. Давай учиться вместе!"),
    ("it", "Italiano", "Ciao, sono Pika. Impariamo insieme!"),
]
_PREVIEW_TEXT = {t: s for t, _, s in LANGUAGES}
_PREVIEW_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()     # (voice, text) -> wav
_PREVIEW_MAX = 512


@app.get("/v2/languages")
async def languages():
    """The languages the model speaks: tag, name, the voice-preview sample line."""
    return {"languages": [{"tag": t, "name": n, "sample": s} for t, n, s in LANGUAGES]}


@app.get("/v2/preview")
async def preview(request: Request, speaker: Optional[str] = Query(None, description="voice family"),
                  lang: str = Query("vi", description="language tag"),
                  speed: Optional[str] = Query(None), accent: Optional[str] = Query(None),
                  text: Optional[str] = Query(None, max_length=200,
                                              description="default: the style's line (vi / en), else the language's sample"),
                  instruct: Optional[str] = Query(None, description=_INSTRUCT_DOC)):
    """A short sample of a voice family in one language (the web UI's voice preview): the voice the
    tags would pick ({speaker}_{lang}[_speed][_accent]) reading `text`, WAV. Without `text`, a styled preview
    reads a line that fits the style (instructs.SAMPLES), else the language's sample. Cached per (voice, text,
    instruct), so a preview clicked again costs no GPU."""
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    ins_key, ins_text, style = _instruct(instruct)
    tag = bot_tts.LANG_ALIASES.get(lang.strip().lower(), lang.strip().lower())
    if tag not in bot_tts.LANGS:
        raise HTTPException(status_code=400, detail=f"unknown lang {lang!r}; languages: {list(bot_tts.LANGS)}")
    line = text or instructs.sample(ins_key, tag) or _PREVIEW_TEXT[tag]
    try:
        segs = bot_tts.build_plan(f"[{tag}] {line.strip()}", speaker, speed, accent, "none",
                                  ENGINE.list_speakers(), MAX_TEXT_CHARS, style=style)
    except bot_tts.BotTtsError as e:
        raise HTTPException(status_code=e.status, detail=e.msg)
    voice = segs[0]["speaker"]
    key = (voice, tuple(s["text"] for s in segs), ins_text)
    wav, cached = _PREVIEW_CACHE.get(key), True
    syn = obs.Synth(request, speaker=speaker or "", langs=[tag], stream=False, lora=accent, input_text=line,
                    normalized_text=[s["text"] for s in segs], sr=OUT_SR, segments=len(segs), voices=voice,
                    instruct=ins_key)
    if wav is None:
        cached = False
        rid, q = ENGINE.submit_plan(segs, "none", ins_text)
        try:
            pcm = await _collect(rid, q)
        except Exception as e:
            syn.done("error", error=str(getattr(e, "detail", e)), error_type=type(e).__name__)
            raise
        wav = _wav_bytes(pcm, OUT_SR)
        _PREVIEW_CACHE[key] = wav
        while len(_PREVIEW_CACHE) > _PREVIEW_MAX:
            _PREVIEW_CACHE.popitem(last=False)
    _PREVIEW_CACHE.move_to_end(key)
    syn.extra["cache"] = "hit" if cached else "miss"
    syn.audio(max(0, len(wav) - 44))
    syn.done()
    return Response(content=wav, media_type="audio/wav",
                    headers={"X-Voices": voice, "X-Langs": tag, "X-Instruct": ins_key,
                             "X-Cache": "hit" if cached else "miss", "Cache-Control": "no-store"})


# the web UI (serve/ui, a static page that calls the endpoints above): /ui
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
ADMIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "admin")
from fastapi.staticfiles import StaticFiles    # noqa: E402


class _UiFiles(StaticFiles):
    """The web pages' files (/ui, /admin), revalidated on every load (ETag -> 304: without a Cache-Control the
    browsers kept an older page for hours after a deploy, 2026-10-11), never framed by another site."""
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "same-origin"
        return resp


AGENT_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.md")


@app.get("/agent.md", include_in_schema=False)
async def agent_md(request: Request):
    """How to use the API, written for AI agents (no key needed to read it): serve/agent.md with {BASE} = this
    server's public URL (HAProxy sets X-Forwarded-Proto) and the live limits."""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    base = f"{proto}://{request.headers.get('host') or request.url.netloc}"
    text = open(AGENT_MD, encoding="utf-8").read()
    for k, v in {"{BASE}": base, "{MAX_TEXT_CHARS}": str(MAX_TEXT_CHARS), "{OUT_SR}": str(OUT_SR),
                 "{DEFAULT_SPEAKER}": bot_tts.DEFAULT_SPEAKER, "{DUAL_SPEAKER}": bot_tts.DUAL_SPEAKER,
                 "{INSTRUCT_DEFAULT}": os.environ.get("TTS_INSTRUCT_DEFAULT", "none") or "none"}.items():
        text = text.replace(k, v)
    return Response(content=text, media_type="text/markdown; charset=utf-8", headers={"Cache-Control": "no-cache"})


@app.get("/robots.txt", include_in_schema=False)
async def robots():
    return Response(content="User-agent: *\nDisallow: /admin/\nDisallow: /docs\nDisallow: /v2/\nAllow: /agent.md\n",
                    media_type="text/plain")


if os.path.isdir(UI_DIR):
    from fastapi.responses import RedirectResponse
    app.mount("/ui", _UiFiles(directory=UI_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/ui/")


def _is_err(p):
    return isinstance(p, tuple) and len(p) == 2 and p[0] == "__error__"


async def _collect(rid, q) -> bytes:
    """All PCM of a submitted request (raises 500 on an engine error)."""
    parts = []
    try:
        while True:
            payload = await q.get()
            if payload is None:
                break
            if _is_err(payload):
                raise HTTPException(status_code=500, detail=payload[1])
            parts.append(payload)
    finally:
        ENGINE.release(rid)
    return b"".join(parts)


async def _respond(request, rid, q, do_stream, t0, speaker, nchars, syn=None):
    """Relay a submitted request's PCM chunks as a WAV (stream=false) or a raw PCM stream. syn (observability.Synth):
    its TTFA, audio and outcome, logged once the request ends."""
    if not do_stream:
        try:
            pcm = await _collect(rid, q)
        except Exception as e:
            if syn:
                syn.done("error", error=str(getattr(e, "detail", e)), error_type=type(e).__name__)
            raise
        dur = len(pcm) / 2 / OUT_SR if OUT_SR else 0.0
        log.info("wav speaker=%s chars=%d audio=%.2fs in %.2fs", speaker, nchars, dur, time.perf_counter() - t0)
        if syn:
            syn.audio(len(pcm))
            syn.done()
        return Response(content=_wav_bytes(pcm, OUT_SR), media_type="audio/wav",
                        headers={"X-Audio-Duration": f"{dur:.3f}", "X-Sample-Rate": str(OUT_SR)})

    media_type = f"audio/pcm; rate={OUT_SR}; channels=1"

    async def body():
        first = True
        try:
            while True:
                payload = await q.get()
                if payload is None:
                    break
                if _is_err(payload):
                    log.error("stream engine error speaker=%s: %s", speaker, payload[1])
                    if syn:
                        syn.done("error", error=str(payload[1]), error_type="EngineError")
                    break
                if first:
                    log.info("stream first-packet speaker=%s in %.0fms", speaker, (time.perf_counter() - t0) * 1000)
                    first = False
                if payload:
                    if syn:
                        syn.audio(len(payload))
                    yield payload
                if await request.is_disconnected():
                    if syn:
                        syn.done("cancelled", error="client_disconnected")
                    break
        except (GeneratorExit, asyncio.CancelledError):    # the server closed the stream: the client went away
            if syn:
                syn.done("cancelled", error="client_disconnected")
            raise
        except Exception as e:
            if syn:
                syn.done("error", error=str(e), error_type=type(e).__name__)
            raise
        finally:
            ENGINE.release(rid)
            if syn:
                syn.done()        # the end: no-op if already logged
    return StreamingResponse(body(), media_type=media_type, headers={"Cache-Control": "no-store"})


async def _timed_events(rid, q, segs, chars=False, sync=False):
    """The events of a request submitted with marks (timing.py): {"type": "audio", "t", "pcm"} as it comes;
    {"type": "segment", "segment", "kind" (speech | sound | chorus), "text", "lang", "start", "end", "amplitude"
    (mouth opening per 40 ms)} at each segment's end; then, once aligned (a worker thread, so the audio is not held
    up), {"type": "words", "segment", "items": [{"word", "start", "end"}]} and {"type": "visemes", "segment",
    "items": [{"v", "start", "end"}]}; {"type": "end", "duration"} last ({"type": "error", "detail"} on a failure).
    Times are seconds from the start of the line; chars: each word with its letters' times too (music.py).
    sync: each segment's audio is held until its timings are out, so a segment arrives as segment, words, visemes,
    then its audio -- a player never has sound it has no mouth for (first audio later by the first segment's
    generation + alignment; the engine keeps generating meanwhile, segments are sent in order)."""
    sr, out, al = OUT_SR, asyncio.Queue(), timing.aligner()

    async def align(k, seg, audio, start):
        try:
            words = await asyncio.to_thread(al.align, audio, sr, seg.get("text", ""), chars)
            for w in words:
                w["start"], w["end"] = round(w["start"] + start, 3), round(w["end"] + start, 3)
                if "chars" in w:
                    w["chars"] = [(round(a + start, 3), round(b + start, 3)) for a, b in w["chars"]]
            vis = await asyncio.to_thread(timing.visemes, words, seg.get("tag") or "vi") if words else []
        except Exception as e:
            log.warning("timing of segment %s failed: %s", k, e)
            words, vis = [], []
        return [{"type": "words", "segment": k, "items": words}, {"type": "visemes", "segment": k, "items": vis}]

    async def put_timing(task):
        for ev in await task:
            out.put_nowait(ev)

    ready = asyncio.Queue()          # sync: (segment event, align task or None, its audio events) in segment order

    async def emit_in_order():
        while (item := await ready.get()) is not None:
            seg_ev, task, held = item
            if seg_ev:
                out.put_nowait(seg_ev)
            if task:
                await put_timing(task)
            for ev in held:
                out.put_nowait(ev)

    async def pump():
        pos, seg_start, buf, k, tasks, held = 0, 0, bytearray(), 0, [], []
        emitter = asyncio.create_task(emit_in_order()) if sync else None
        try:
            while True:
                p = await q.get()
                if p is None:
                    break
                if _is_err(p):
                    if sync:            # what was generated before the error still goes out first
                        ready.put_nowait((None, None, held))
                        ready.put_nowait(None)
                        await emitter
                        emitter, held = None, []
                    out.put_nowait({"type": "error", "detail": p[1]})
                    break
                if isinstance(p, tuple) and p and p[0] == "__seg__":
                    seg = segs[k] if k < len(segs) else {}
                    audio = np.frombuffer(bytes(buf), np.int16).astype(np.float32) / 32768.0
                    start = seg_start / sr
                    kind = "chorus" if seg.get("chorus") else "sound" if seg.get("sfx") else "speech"
                    seg_ev = {"type": "segment", "segment": k, "kind": kind, "text": seg.get("text", ""),
                              "lang": seg.get("tag"), "start": round(start, 3),
                              "end": round(start + len(audio) / sr, 3), "amplitude": timing.amplitude(audio, sr)}
                    task = asyncio.create_task(align(k, seg, audio, start)) if kind == "speech" else None
                    if sync:
                        ready.put_nowait((seg_ev, task, held))
                        held = []
                    else:
                        out.put_nowait(seg_ev)
                        if task:
                            tasks.append(asyncio.create_task(put_timing(task)))
                    seg_start += len(audio)
                    buf.clear()
                    k += 1
                    continue
                if p:
                    ev = {"type": "audio", "t": round(pos / sr, 3), "pcm": p}
                    if sync:
                        held.append(ev)
                    else:
                        out.put_nowait(ev)
                    buf += p
                    pos += len(p) // 2
            if emitter:
                ready.put_nowait((None, None, held))       # audio after the last segment mark (none expected)
                ready.put_nowait(None)
                await emitter
            if tasks:
                await asyncio.gather(*tasks)
            out.put_nowait({"type": "end", "duration": round(pos / sr, 3)})
        finally:
            if emitter and not emitter.done():
                emitter.cancel()
            ENGINE.release(rid)
            out.put_nowait(None)

    task = asyncio.create_task(pump())
    try:
        while True:
            ev = await out.get()
            if ev is None:
                break
            yield ev
    finally:
        task.cancel()


async def _respond_timed(request, rid, q, segs, do_stream, t0, label, sync=False, syn=None):
    """/v2/multilang/timed: stream=true -> NDJSON events (_timed_events; audio as base64 16-bit PCM in "pcm16";
    sync: each segment's timings before its audio); stream=false -> JSON {"sample_rate", "duration",
    "audio_wav_base64", "segments", "words", "visemes"}. syn (observability.Synth): logged once it ends."""
    import base64
    if do_stream:
        async def body():
            first = True
            try:
                async for ev in _timed_events(rid, q, segs, sync=sync):
                    if ev["type"] == "audio":
                        if first:
                            log.info("timed stream first-packet %s in %.0fms", label, (time.perf_counter() - t0) * 1000)
                            first = False
                        if syn:
                            syn.audio(len(ev["pcm"]))
                        ev = {"type": "audio", "t": ev["t"], "pcm16": base64.b64encode(ev["pcm"]).decode()}
                    elif ev["type"] == "error" and syn:
                        syn.done("error", error=str(ev.get("detail")), error_type="EngineError")
                    yield (json.dumps(ev, ensure_ascii=False) + "\n").encode()
                    if await request.is_disconnected():
                        if syn:
                            syn.done("cancelled", error="client_disconnected")
                        break
            except (GeneratorExit, asyncio.CancelledError):
                if syn:
                    syn.done("cancelled", error="client_disconnected")
                raise
            except Exception as e:
                if syn:
                    syn.done("error", error=str(e), error_type=type(e).__name__)
                raise
            finally:
                if syn:
                    syn.done()
        return StreamingResponse(body(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store",
                                 "X-Sample-Rate": str(OUT_SR)})
    pcm, seg_ev, words, vis = [], [], [], []
    async for ev in _timed_events(rid, q, segs):
        if ev["type"] == "audio":
            pcm.append(ev["pcm"])
        elif ev["type"] == "segment":
            seg_ev.append({k: v for k, v in ev.items() if k != "type"})
        elif ev["type"] == "words":
            words += ev["items"]
        elif ev["type"] == "visemes":
            vis += ev["items"]
        elif ev["type"] == "error":
            if syn:
                syn.done("error", error=str(ev["detail"]), error_type="EngineError")
            raise HTTPException(status_code=500, detail=ev["detail"])
    data = b"".join(pcm)
    if syn:
        syn.audio(len(data))
        syn.done()
    log.info("timed %s audio=%.2fs words=%d in %.2fs", label, len(data) / 2 / OUT_SR, len(words), time.perf_counter() - t0)
    return Response(content=json.dumps({
        "sample_rate": OUT_SR, "duration": round(len(data) / 2 / OUT_SR, 3),
        "audio_wav_base64": base64.b64encode(_wav_bytes(data, OUT_SR)).decode(),
        "segments": seg_ev, "words": sorted(words, key=lambda w: w["start"]),
        "visemes": sorted(vis, key=lambda v: v["start"])}, ensure_ascii=False), media_type="application/json")


class EngineClient:
    """API-side handle to the engine-worker process(es) (see engine_worker.py).

    Owns the spawned process(es) + their IPC queues. One reader thread per worker drains that
    worker's output queue and hands each PCM chunk to the right request's asyncio.Queue via
    call_soon_threadsafe, so delivery happens on the event loop the instant a chunk arrives.

    Multi-GPU: TTS_GPUS="1,2" spawns one worker per listed physical GPU (each worker restricts
    itself with CUDA_VISIBLE_DEVICES before touching CUDA); each request goes to the worker with the
    fewest unfinished requests.
    """
    def __init__(self, cfg: dict, args):
        ctx = mp.get_context("spawn")  # spawn: child inits CUDA fresh; parent never touches it
        base = {
            "model": args.model, "speakers_dir": args.speakers_dir, "device": args.device,
            "attn": args.attn, "out_sr": int(args.rate), "frames_per_chunk": cfg["frames_per_chunk"],
            "gen_params": config.gen_params(), "batch_size": cfg["batch_size"],
        }
        self.workers = []
        # TTS_POST_PROCS processes per worker do its audio post-processing (resampling, tempo, effects, joins: see
        # engine_worker.post_handle), off the engine loop's one core; 0 = in the worker itself
        n_post = max(0, int(os.environ.get("TTS_POST_PROCS", "4")))
        for wid, g in enumerate(cfg["gpus"] or [None]):
            w = SimpleNamespace(in_q=ctx.Queue(), out_q=ctx.Queue(), load=0, wid=wid, gpu=g)
            w.post_qs = [ctx.Queue() for _ in range(n_post)]
            w.posts = [ctx.Process(target=post_main, args=(q, w.out_q, i), name=f"tts-post-{wid}-{i}", daemon=True)
                       for i, q in enumerate(w.post_qs)]
            for p in w.posts:
                p.start()
            wcfg = dict(base, gpu=g, worker_id=wid, post_qs=w.post_qs)
            w.proc = ctx.Process(target=fast_main, args=(w.in_q, w.out_q, wcfg),
                                 name=f"tts-engine-worker-{wid}", daemon=True)
            w.proc.start()
            self.workers.append(w)
        self.speakers: list = []
        for w in self.workers:             # workers load in parallel; wait for all
            self.speakers = self._wait_ready(w)
        self.out_sr = int(args.rate)
        # req_id -> [asyncio.Queue, worker, running, level.Leveler or None] (loop thread only)
        self._reqs: dict = {}
        self._composite: dict = {}       # rid -> (task, part rids) of a plan put out in parts (submit_plan)
        self._counter = 0
        self._loop = None

    def _wait_ready(self, w, timeout: float = 1800.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = w.out_q.get(timeout=5)
            except _queue.Empty:
                if not w.proc.is_alive():
                    raise RuntimeError(f"engine worker {w.wid} died during startup")
                continue
            if isinstance(msg, tuple) and msg and msg[0] == READY:
                log.info("engine worker %d (gpu=%s) ready", w.wid, w.gpu)
                return list(msg[1])
        raise RuntimeError(f"engine worker {w.wid} did not become ready in time")

    def list_speakers(self):
        """The voices, sorted: one list for the whole run (the voices are fixed at startup), so per-request
        lookups index it once (bot_tts._by_prefix) instead of re-sorting and scanning 6k+ names each time."""
        if getattr(self, "_sorted", None) is None:
            self._sorted = sorted(self.speakers)
            self._names = frozenset(self._sorted)
        return self._sorted

    def has_speaker(self, name) -> bool:
        self.list_speakers()
        return name in self._names

    def attach_loop(self, loop):
        self._loop = loop
        for w in self.workers:
            threading.Thread(target=self._read_loop, args=(w,), name=f"engine-reader-{w.wid}",
                             daemon=True).start()

    def _read_loop(self, w):
        # Blocking get releases the GIL; runs in its own thread, never on the event loop.
        while True:
            rid, payload = w.out_q.get()
            loop = self._loop
            if loop is not None:
                loop.call_soon_threadsafe(self._deliver, rid, payload)

    def _deliver(self, rid, payload):
        ent = self._reqs.get(rid)
        if ent is not None:
            lev = ent[3]
            last = isinstance(payload, tuple) and len(payload) == 2 and payload[0] == level.LAST
            if last:                                                   # the request's last chunk (engine_worker)
                payload = payload[1]
            if lev is not None and isinstance(payload, (bytes, bytearray)) and payload:
                payload = lev.process(payload, last)                   # the output level (level.py)
            ent[0].put_nowait(payload)
            if payload is None and ent[2]:      # stream finished on the worker side
                ent[2] = False
                ent[1].load -= 1
                if isinstance(ent[0], _NullQ):  # client already gone
                    self._reqs.pop(rid, None)

    def submit(self, text, lang, speaker):
        rid = self._counter
        self._counter += 1
        q = asyncio.Queue()
        w = min(self.workers, key=lambda x: (x.load, x.wid))
        w.load += 1
        self._reqs[rid] = [q, w, True, self._leveler([speaker])]
        w.in_q.put((rid, text, lang, speaker))
        return rid, q

    def _leveler(self, speakers, instruct=None):
        return level.Leveler(self.out_sr, level.is_whisper(speakers, instruct)) if level.ENABLED else None

    def submit_chain(self, segments, ctx="prev", instruct=None, marks=False):
        """segments: [{"text", "lang", "speaker"}, ...] voiced as one code-switched line; instruct: the
        instruction text for all of it (instructs.resolve), or None; marks: ("__seg__", k) put on the queue after
        each segment's audio (word timings, /v2/multilang/timed)."""
        rid = self._counter
        self._counter += 1
        q = asyncio.Queue()
        w = min(self.workers, key=lambda x: (x.load, x.wid))
        w.load += 1
        self._reqs[rid] = [q, w, True, self._leveler([s.get("speaker") for s in segments], instruct)]
        msg = {"segments": segments, "ctx": ctx}
        if instruct:
            msg["instruct"] = instruct
        if marks:
            msg["marks"] = True
        w.in_q.put((rid, msg, None, CHAIN))
        return rid, q

    def submit_plan(self, segs, ctx="prev", instruct=None, marks=False):
        """A plan from bot_tts.build_plan: a plain chain goes to submit_chain; one with chorus segments
        ({"chorus": [segments of each voice]}) is put out in parts -- the chains between the choruses and every
        chorus voice submitted at once (they generate side by side), their audio sent in order on one queue, a chorus
        when all its voices are done, mixed into unison (chorus.py). The same (rid, queue) contract as submit_chain;
        release(rid) cancels every part. marks: a ("__seg__", ...) after each segment of the plan, a chorus too."""
        if not any(s.get("chorus") for s in segs):
            return self.submit_chain(segs, ctx, instruct, marks)
        parts, run = [], []
        for s in segs:
            if s.get("chorus"):
                if run:
                    parts.append(("chain", run))
                    run = []
                parts.append(("chorus", s))
            else:
                run.append(s)
        if run:
            parts.append(("chain", run))
        subs = []
        for kind, p in parts:
            if kind == "chain":
                subs.append(("chain", [self.submit_chain(p, ctx, instruct, marks)], None))
            else:
                subs.append(("chorus", [self.submit_chain(m, "prev", instruct) for m in p["chorus"]],
                             (p["text"], p.get("post") or {})))
        rid = self._counter
        self._counter += 1
        q = asyncio.Queue()
        task = asyncio.get_running_loop().create_task(self._run_parts(q, subs, marks))
        self._composite[rid] = (task, [r for _, rs, _ in subs for r, _ in rs])
        return rid, q

    async def _run_parts(self, q, subs, marks=False):
        try:
            for kind, rs, text in subs:
                if kind == "chain":
                    (rid, sq), = rs
                    while True:
                        p = await sq.get()
                        if p is None:
                            break
                        q.put_nowait(p)
                        if _is_err(p):
                            q.put_nowait(None)
                            return
                    self.release(rid)
                    continue
                pcms = []
                for rid, sq in rs:
                    chunks = []
                    while True:
                        p = await sq.get()
                        if p is None:
                            break
                        if _is_err(p):
                            q.put_nowait(p)
                            q.put_nowait(None)
                            return
                        chunks.append(p)
                    self.release(rid)
                    pcms.append(b"".join(chunks))
                text, post = text
                mixed = await asyncio.to_thread(chorus.mix, pcms, self.out_sr)
                end = (text or "").rstrip()[-1:]           # the pause after it: <pause>'s, else by its punctuation
                gap = post.get("pause_after", 0.3 if end in ".!?…" else 0.18 if end in ",;:" else 0.08)
                lead = b"\0\0" * int(post.get("pause_before", 0) * self.out_sr)
                q.put_nowait(lead + mixed + b"\0\0" * int(gap * self.out_sr))
                if marks:
                    q.put_nowait(("__seg__", "chorus"))
            q.put_nowait(None)
        except asyncio.CancelledError:
            raise
        except Exception as e:                              # a mix that fails ends the line with an error
            log.error("chorus line failed: %s", e)
            q.put_nowait(("__error__", f"chorus failed: {e}"))
            q.put_nowait(None)

    def release(self, rid):
        comp = self._composite.pop(rid, None)
        if comp is not None:                                # a plan in parts: stop it and every part
            comp[0].cancel()
            for r in comp[1]:
                self.release(r)
            return
        ent = self._reqs.pop(rid, None)
        if ent is not None and ent[2]:
            # Client left before the stream ended: cancel it on the worker (frees the slot) and keep
            # counting its load until the worker's terminal None arrives, via a sink entry.
            self._reqs[rid] = [_NullQ(), ent[1], True, None]
            ent[1].in_q.put(("__cancel__", rid, None, None))


class _NullQ:
    def put_nowait(self, payload):
        pass


@app.on_event("startup")
async def _on_startup():
    if ENGINE is not None:
        ENGINE.attach_loop(asyncio.get_running_loop())
        log.info("engine reader attached to event loop")
    try:                                   # /v2/multilang (+ the dual endpoints, bot/tts): load the language models now
        import lang_split
        import multilang
        t = time.perf_counter()
        lang_split.splitter()
        multilang.splitter()
        log.info("lang_split + multilang loaded in %.1fs", time.perf_counter() - t)
    except Exception as e:                 # the other endpoints work without them
        log.warning("language detection not loaded (/v2/multilang, dual, bot/tts unavailable): %s", e)


def main():
    global ENGINE, OUT_SR
    ap = argparse.ArgumentParser(description="Defaults come from the repo-root .env (.env.example); CLI flags override.")
    ap.add_argument("--model", default=_CFG["model"])
    ap.add_argument("--speakers-dir", default=_CFG["speakers_dir"])
    ap.add_argument("--device", default=_CFG["device"], help="single-worker device when TTS_GPUS is empty")
    ap.add_argument("--attn", default=_CFG["attn"], choices=["sdpa", "eager"])
    ap.add_argument("--host", default=_CFG["host"])
    ap.add_argument("--port", type=int, default=_CFG["port"])
    ap.add_argument("--rate", type=int, default=_CFG["out_sr"], help="output sample rate (Hz)")
    args = ap.parse_args()
    _SLOW_NEW["path"] = str(args.speakers_dir).rstrip("/") + ".slow_p4.json"   # /v2/speakers slow_new (_slow_new)

    OUT_SR = args.rate
    # API keys, limits, the request log, the admin console (auth.py). Here, not at import: the engine workers are
    # spawned processes that import this module again, and must not open the database or start its writer.
    store = auth.Store()
    app.add_middleware(auth.AuthMiddleware, store=store)
    obs.install(app)          # after auth: the outermost, so http_request counts its rejections too
    bot_tts.CACHE_ID = store.cache_id
    auth.register(app, store, lambda: _engine_status(args))
    if os.path.isdir(ADMIN_DIR):
        app.mount("/admin", _UiFiles(directory=ADMIN_DIR, html=True), name="admin")
    log.info("auth: %d API keys, %d admin accounts, mode %s", len(store.keys), store.admin_count(),
             "enforce" if store.enforce else "report")
    store.event("start", f"{os.path.basename(str(args.model).rstrip('/'))}, {args.speakers_dir}, port {args.port}")
    t_start = time.time()
    ENGINE = EngineClient(_CFG, args)
    log.info("serving on http://%s:%d  %d voices", args.host, args.port, len(ENGINE.list_speakers()))
    store.event("ready", f"{len(ENGINE.list_speakers())} voices, {_CFG['batch_size']} slots, "
                         f"engine up in {time.time() - t_start:.0f} s")
    # proxy_headers off: uvicorn would take the client address from X-Forwarded-For sent through HAProxy, which a client
    # can write itself; auth.py reads HAProxy's X-Real-IP instead
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", server_header=False, proxy_headers=False)


def _engine_status(args) -> dict:
    """The engine's side of the admin console's status: slots, streams in flight per worker, voices, config."""
    if ENGINE is None:
        return {"ready": False}
    return {"ready": True, "slots": _CFG["batch_size"], "workers": [{"id": w.wid, "gpu": w.gpu, "streams": w.load,
                                                                      "alive": w.proc.is_alive()} for w in ENGINE.workers],
            "streams": sum(w.load for w in ENGINE.workers), "voices": len(ENGINE.list_speakers()),
            "model": os.path.basename(str(args.model).rstrip("/")), "speakers_dir": args.speakers_dir,
            "instruct_mode": os.environ.get("TTS_INSTRUCT_MODE", ""), "out_sr": OUT_SR, "max_text_chars": MAX_TEXT_CHARS}


if __name__ == "__main__":
    main()
