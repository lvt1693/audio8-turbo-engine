#!/usr/bin/env python3
"""
Separate-process TTS engine worker (one per GPU).

The autoregressive generation is kept out of the FastAPI/asyncio process: this process owns the
GPU model and runs fast_engine.FastEngine single-threaded with no event loop, and talks to the
API over multiprocessing queues:

    API process  --(req_id, text, lang, speaker)-->  in_q   --> fast_main
    fast_main    --(req_id, pcm_bytes | None)----->  out_q  --> API process (async relay)

PCM is resampled + int16-encoded HERE so the API process only relays bytes.
"""
from __future__ import annotations

import math
import os
import queue as _queue
import re
import time
import traceback
import zlib
from collections import OrderedDict, deque

import numpy as np

import effects
import elongate
import tempo

# Sentinels carried over out_q alongside (req_id, payload):
#   payload = bytes  -> a PCM16 chunk for that request
#   payload = None   -> that request finished (end of stream)
#   payload = ("__error__", msg) -> that request failed
# in_q also accepts ("__cancel__", req_id, None, None) when the client disconnected, and
# (req_id, {"segments": [{"text", "lang", "speaker"[, "ctx"][, "sfx"]}, ...], "ctx": "prev"|"full"|"none"
# [, "instruct": text][, "marks": True]}, None, CHAIN) for a code-switched line voiced segment by segment (see
# fast_main; marks: ("__seg__", k) after each segment's audio, for word timings); a
# segment's own "ctx" overrides the line's for the context it is admitted with; "sfx": a sound (sfx.py), never
# context for the segments after it; "stretch": it ends with a drawn-out word, held whole and its vowel held
# (elongate.py); "post": what is done to its decoded audio (effects.py: "tempo", "em" -- held whole --, "fx",
# "pause_after" / "pause_before" seconds of silence); "instruct" (instructs.py) holds for every segment of the line.
READY = "__ready__"   # first out_q message: (READY, [speaker names])
CHAIN = "__chain__"

# Frame counts below are in 12.5 Hz frames (80 ms, the Qwen3-TTS codec they were tuned on); fast_main scales them by
# the engine's own frame rate (Audio8: 44100 / 2048 = 21.5 Hz) through FRAME_SCALE.
FRAME_SCALE = 1.0

# code-switch chains
SIL_AMP = 10 ** (-45 / 20)   # a frame whose RMS is below this is silence
LEAD_MAX = 12                # a continuation silent for longer than this is resampled ...
TRIES_MAX = 3                # ... at most this many times
# from this retry on a silent continuation goes without the previous segments as context (queue_segment):
# the 1.7B CustomVoice talker often ends a continuation at once when it continues a span in another language
CTX_DROP_TRY = int(os.environ.get("TTS_CHAIN_CTX_DROP_TRY", "2"))
SHORT_CHARS = 24             # continuations this short (in text_units) are held whole until they end
# A sound segment (sfx.py) ends after this many silent frames once it has sounded and run 80 % of its reference
# clip's frames: the model makes the sound and then often never says EOS, it goes on in silence to the frame cap
# (yuna, a sound alone: 27/76 takes silent or with > 0.8 s of dead tail; 1/76 after). Not before 80 %: a clip can
# have a gap inside (a throat clear, then "ahem"; a sigh's breath in and out: 4-8 silent frames). Sounds are held
# whole and resampled when they run on or come out silent; silent on every try, the sound is left out.
SFX_QUIET = 5
SFX_MIN_SHARE = 0.8
SFX_TAIL = 3                 # silent frames kept after a sound (0.24 s; a switch after it cuts to _keep anyway)


def _sound_min(item):
    """Frames a sound segment runs before silence may end it (SFX_MIN_SHARE of its reference clip)."""
    rc = getattr(item, "ref_code", None)
    return int(SFX_MIN_SHARE * len(rc)) if rc is not None else 0
# reference + instruction (TTS_INSTRUCT_MODE=guide): an instructed line of a voice with a reference is one
# guided generation. Same model (no TTS_GUIDE_DIRECTOR): three slots of this engine (fast_engine.guide_groups).
# Cross-model (TTS_GUIDE_DIRECTOR = another checkpoint, e.g. the 1.7B clone + instruct merge steering the
# 0.6B): one slot here + its two rows in the director (fast_engine.guide_cross).
# TTS_GUIDE_W = the weight spec, "spec[,language:spec...]" (fast_engine.parse_guide_spec: "5", "4/10",
# "12@12:6"); the instruction's effect sets in at a threshold that depends on the model pair and language.
GUIDE = os.environ.get("TTS_INSTRUCT_MODE", "xvec").strip().lower() == "guide"


def _guide_weights(spec):
    w = {"": "4"}
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        m = re.match(r"^([A-Za-z]+)\s*[:=]\s*(.+)$", part)
        if m:
            w[m.group(1).lower()] = m.group(2).strip()
        else:
            w[""] = part
    return w


GUIDE_W = _guide_weights(os.environ.get("TTS_GUIDE_W", "4"))
# admission by compute (cross-model): a line costs 1 unit, a steered line TTS_GUIDE_UNITS (its 0.6B row + the
# director's 2 rows of the 1.7B talker: ~4 plain rows of tick time); while TTS_MAX_UNITS (0 = off) would be
# exceeded, or no director pair is free (TTS_GUIDE_DIRECTOR_POOL), the next request waits for one to end
GUIDE_UNITS = float(os.environ.get("TTS_GUIDE_UNITS", "4"))
MAX_UNITS = float(os.environ.get("TTS_MAX_UNITS", "0"))


def guide_weight(lang):
    return GUIDE_W.get(str(lang).lower(), GUIDE_W[""])
CHAIN_DUMP = os.environ.get("TTS_CHAIN_DUMP", "")   # debug: dir that gets every chain segment's raw
                                                    # per-tick audio + tokens + codes (before gating)


# startup warm-up: a voice named {speaker}_{lang}[_...] is only used for its own language, so only that
# language's prompt-prefix KV is precomputed for it. That KV stays on the GPU, so warming every voice in
# every language of TTS_WARM_LANGS costs memory for pairs that are never used (156 voices: 2 GB more).
# Names without a language part get all of TTS_WARM_LANGS; any other (voice, language) pair is computed
# on its first request.
_VOICE_LANG = re.compile(r"_(vi|en|zh|ja|ko|fr|de|es|pt|ru|it)(?=_|$)")
_LANG_WARM = {"vi": ("Auto", "Vietnamese"), "en": ("English",), "zh": ("Chinese",), "ja": ("Japanese",),
              "ko": ("Korean",), "fr": ("French",), "de": ("German",), "es": ("Spanish",),
              "pt": ("Portuguese",), "ru": ("Russian",), "it": ("Italian",)}


def _voice_warm_langs(name, default):
    m = _VOICE_LANG.search(name)
    return _LANG_WARM[m.group(1)] if m else default


# Voice pool (user, 2026-10-05): the pika voices and the bots' voices are always loaded; the other voice
# families share TTS_VOICE_SLOTS places, the least recently used one dropped when a new one comes in.
PINNED_FAMILIES = [f.strip() for f in os.environ.get(
    "TTS_PINNED_FAMILIES", "pika,pika-1,pika-2,pika-3,boy,girl,mely,uncle,justin,grandma,baby,belle-b,"
                           "belle-b-2,grandpa,sieunhanxanh,sieunhantim").split(",") if f.strip()]
VOICE_SLOTS = int(os.environ.get("TTS_VOICE_SLOTS", "10"))


def voice_family(name):
    """{family}_{lang}[_attrs] -> family; a name with no language part is its own family (pika_bac)."""
    m = _VOICE_LANG.search(name)
    return name[:m.start()] if m else name


def pool_unit(name):
    """The voice pool's unit of a voice: its family, or "{family}/{style}" for a style voice
    ({family}_{lang}..._{style}, TTS_INSTRUCT_MODE=voices) -- each style comes and goes on its own, and only
    plain voices are pinned (a family's 34 styles x languages x speeds would pin ~1 GB)."""
    import instructs
    fam = voice_family(name)
    last = name.rsplit("_", 1)[-1]
    if fam != name and last in {k for k, *_ in instructs.PRESETS}:
        return f"{fam}/{last}"
    return fam


class VoicePool:
    """Which voices keep their prompt state (prefix KV per language + prompt embeds, ~7 MB a voice) on
    the GPU. Pinned families (PINNED_FAMILIES, and every family whose name starts with "pika", so the
    recorded pika_* voices too) always do: warmed at startup, never dropped. The other families share
    `slots` places, least recently used out: a family used again moves to the front; a new one comes in
    by computing its state on first use (a ~60-token prefill per voice and language, a few ms), and past
    `slots` the family used longest ago is dropped (`evict`, fast_engine.evict_voices). Dropping is
    always safe: an admitted row keeps its own copy of the KV in its slot, and a dropped voice is just
    rebuilt when it comes back. The vocoder snapshots of every voice stay in host memory, so any voice
    of the speakers dir can be used at any time while VRAM stays bounded by pinned + slots."""

    def __init__(self, names, pinned=PINNED_FAMILIES, slots=VOICE_SLOTS, evict=None, log=print):
        self.fam = {n: pool_unit(n) for n in names}
        self.members = {}
        for n, f in self.fam.items():
            self.members.setdefault(f, []).append(n)
        pin = set(pinned)
        self.pinned = {f for f in self.members if "/" not in f and (f in pin or f.startswith("pika"))}
        self.missing = sorted(pin - set(self.members))       # pinned names with no voice on disk
        self.slots, self.evict, self.log = max(1, int(slots)), evict, log
        self.lru: "OrderedDict[str, bool]" = OrderedDict()
        self.loads = self.drops = 0

    def family(self, name):
        return self.fam.get(name) or pool_unit(name)

    def is_pinned(self, name):
        return self.family(name) in self.pinned

    def use(self, names):
        """The voices of an admission batch are about to be used: refresh their families, then drop the
        least recently used families past the slot count -- never one this batch needs (a batch needing
        more than `slots` families overflows until a later batch). Returns the families dropped."""
        need = set()
        for n in names:
            f = self.family(n)
            if f in self.pinned:
                continue
            need.add(f)
            if f in self.lru:
                self.lru.move_to_end(f)
            else:
                self.lru[f] = True
                self.loads += 1
        dropped = []
        for f in list(self.lru):
            if len(self.lru) <= self.slots:
                break
            if f not in need:
                del self.lru[f]
                dropped.append(f)
                self.drops += 1
                if self.evict:
                    self.evict(self.members.get(f, [f]))
        if dropped:
            self.log(f"[pool] dropped {', '.join(dropped)}; in the {self.slots} slots: {', '.join(self.lru)}")
        return dropped


def text_units(text):
    """Length of `text` in Latin-letter equivalents of speech time: a Han character or a Hangul block is
    a whole syllable (~0.2-0.25 s, about 3 Latin letters), a kana a mora (~2), anything else 1."""
    n = 0
    for c in text:
        o = ord(c)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or 0xAC00 <= o <= 0xD7AF:
            n += 3
        elif 0x3040 <= o <= 0x30FF:
            n += 2
        else:
            n += 1
    return n


def _seg_cap(text):
    """Frame cap of one segment: ~3x a normal speech rate. A very short target is where ICL goes
    wrong -- the model doesn't stop and reads its prompt (ref transcript, context) on and on."""
    return int((12 + 2 * text_units(text)) * FRAME_SCALE)


def _sound_cap(text, item):
    """Frame cap of a sound segment (sfx.py): its voice's reference clip is that sound, so it lasts about as
    long as the clip, not as its letters read ("ewww!" 2.4 s): 2x the clip's frames (scaled up for a text longer
    than the clip's transcript) + 4."""
    rc = getattr(item, "ref_code", None)
    if rc is None:
        return int(max(30, 12 + 4 * text_units(text)) * FRAME_SCALE)
    scale = max(1.0, text_units(text) / max(1, text_units(getattr(item, "ref_text", "") or "")))
    return int(2 * len(rc) * scale) + 4


_SENT_END = set(".!?…。！？")
_CLAUSE_END = set(",;:，、；：")
_CLOSERS = "\"'”’»)）]」』"


def _keep(text):
    """(tail_s, lead_s): silence kept at a switch after the segment `text` -- after its last speech
    frame / before the next segment's first one -- by the punctuation that ends `text` (ASCII or CJK
    full-width, closing quotes / brackets skipped). Sized to the pauses of one-pass speech: a sentence
    end ~0.3-0.45 s, a comma ~0.2-0.25 s, no punctuation a word gap (~0.1-0.15 s, the speech frames on
    either side carry some silence of their own)."""
    t = text.rstrip().rstrip(_CLOSERS)[-1:]
    return (0.24, 0.16) if t in _SENT_END else (0.16, 0.08) if t in _CLAUSE_END else (0.06, 0.04)


PAUSE_TAIL_S, PAUSE_LEAD_S = 0.04, 0.02      # around a <pause>: only this much of the segments' own silence


def _seg_keep(seg):
    """_keep for a chain segment tuple: a <pause> after it (post "pause_after") keeps only a sliver of the silence
    around it, the pause itself being the silence asked for."""
    if (seg[6] or {}).get("pause_after"):
        return PAUSE_TAIL_S, PAUSE_LEAD_S
    return _keep(seg[0])


# A segment that ends on a comma / a sentence end gets its _keep tail even when the voice left less silence than that
# after it: padded, not only trimmed (user 2026-10-21: "sau dấu , ko có nghỉ tí": a comma cut, bot_tts.cut_pauses,
# measured 0.09-0.33 s without it). A word gap and a <pause> are left as they are. TTS_PUNCT_PAD=0: trim only.
PUNCT_PAD = os.environ.get("TTS_PUNCT_PAD", "1").strip().lower() not in ("0", "false", "no", "off")


def _seg_pad(seg):
    """Seconds of silence the segment's tail is padded up to (0: none, see PUNCT_PAD)."""
    t = seg[0].rstrip().rstrip(_CLOSERS)[-1:]
    if not PUNCT_PAD or (seg[6] or {}).get("pause_after") or t not in _SENT_END | _CLAUSE_END:
        return 0.0
    return _keep(seg[0])[0]


# Switches joined (user 2026-10-21: "tớ (nghỉ 1 tí) dislike những người ... dựa vào khoảng cách giữa các từ đã nói
# mà cắt/thêm nghỉ, và blend 2 audio với nhau"). The gating above works in whole 80 ms frames: the last sounding frame
# of segment k and the first of k+1 carry silence of their own, so a switch with no punctuation had 0.1-0.2 s of it
# and one after a sentence end 0.23-0.72 s (a slow voice's tempo stretching the padded tail). Here, on the output
# audio (after each segment's own tempo / effect), the end of k's last word and the start of k+1's first one are
# found to 5 ms and the silence between them is cut or filled to the switch's gap (_switch_gap):
#   no punctuation   SWITCH_GAP, or the gap the voice leaves between its own words in this line where that is longer
#                    (the median of its measured gaps, at most JOIN_GAP_MAX: a slow voice). The voice's own word gap
#                    alone (~0.04 s) was too tight (user: "khoảng đệm vẫn phải có hợp lý chứ, quá sát làm 'là cat
#                    đó' nó bị dồn dập"); 0.1-0.2 s before the join read as a pause. TTS_SWITCH_GAP
#   , ; :            JOIN_PUNCT clause (the comma pause the user approved: median 0.28 s)
#   . ! ?  / ...     JOIN_PUNCT sentence / ellipsis
# k's ring-out fades under k+1's first breath. Left as before: around a sound, a <pause> or an effect, and where k+1
# starts without k as context (another speaker / style: a turn in a dialogue). The tail after k's last word waits
# for k+1's first speech (a few frames); JOIN_HOLD frames of k are held back so its end is still here when it ends
# (never the line's first chunk: no first-audio latency). TTS_SWITCH_JOIN=0: as before.
SWITCH_JOIN = os.environ.get("TTS_SWITCH_JOIN", "1").strip().lower() not in ("0", "false", "no", "off")
SWITCH_GAP = float(os.environ.get("TTS_SWITCH_GAP", "0.11"))
JOIN_GAP_MAX = 0.20
JOIN_PUNCT = {"clause": 0.28, "sentence": 0.40, "ellipsis": 0.45}
JOIN_HOLD = 2
_ENV_S = 0.005                   # envelope window
_LOUD_DB = -30.0                 # speech: louder than this under the line's loud level (and than SIL_AMP)


def _envelope(y, sr):
    hop = max(1, int(_ENV_S * sr))
    n = len(y) // hop
    if n == 0:
        return np.zeros(0, np.float32), hop
    return np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12), hop


def _loud(y, sr, level):
    e, hop = _envelope(y, sr)
    return e > max(SIL_AMP, level * 10 ** (_LOUD_DB / 20)), hop


def _speech_end(y, sr, level):
    """Sample index right after the last speech in y (0: none in it)."""
    loud, hop = _loud(y, sr, level)
    idx = np.flatnonzero(loud)
    return int((idx[-1] + 1) * hop) if idx.size else 0


def _speech_start(y, sr, level):
    """Sample index of the first speech in y (None: none in it)."""
    loud, hop = _loud(y, sr, level)
    idx = np.flatnonzero(loud)
    return int(idx[0] * hop) if idx.size else None


def _word_gaps(y, sr, level, lo=0.015, hi=0.25):
    """The silences between sounding stretches of y, lo..hi seconds long: the gaps the voice leaves between words."""
    loud, hop = _loud(y, sr, level)
    idx = np.flatnonzero(loud)
    out, run = [], 0
    for v in (loud[idx[0]:idx[-1] + 1] if idx.size else ()):
        if v:
            if run and lo <= run * hop / sr <= hi:
                out.append(run * hop / sr)
            run = 0
        else:
            run += 1
    return out


def _join_gap(gaps):
    """The gap at a joined switch with no punctuation: SWITCH_GAP, or the median of the voice's word gaps so far
    where that is longer (at most JOIN_GAP_MAX)."""
    return float(max(SWITCH_GAP, min(JOIN_GAP_MAX, np.median(gaps)))) if len(gaps) >= 3 else SWITCH_GAP


def _join(tail, head, onset, sr, gap):
    """Segment k's audio after its last word (`tail`: ring-out, silence) + segment k+1's first audio (`head`, speech
    from sample `onset`) -> one stretch with `gap` seconds from the end of the word to the start of the next: the
    tail cut / filled to it and faded out, the head coming in 15 ms before its onset, over the end of the tail."""
    g = int(round(gap * sr))
    lead = min(onset, int(0.015 * sr), g)
    h = np.array(head[onset - lead:], np.float32)
    t = np.array(tail[:g], np.float32)
    if t.size:
        t *= np.cos(np.linspace(0.0, np.pi / 2, t.size, dtype=np.float32)) ** 2
    if lead:
        h[:lead] *= np.sin(np.linspace(0.0, np.pi / 2, lead, dtype=np.float32)) ** 2
    out = np.zeros(max(t.size, g - lead + h.size), np.float32)
    out[:t.size] += t
    out[g - lead:g - lead + h.size] += h
    return out


def _switch_gap(segs, k):
    """The silence at the switch after chain segment k, from the end of its last word to the start of the next one's
    first: seconds, "word" (the voice's own word gap, _join_gap), or None (the switch is not joined: SWITCH_JOIN)."""
    if not SWITCH_JOIN or k + 1 >= len(segs):
        return None
    a, b = segs[k], segs[k + 1]
    pa, pb = a[6] or {}, b[6] or {}
    if (a[4] or b[4] or b[3] == "none" or pa.get("pause_after") or pb.get("pause_before")
            or pa.get("fx") or pb.get("fx")):
        return None
    t = a[0].rstrip().rstrip(_CLOSERS)
    if t.endswith("...") or t.endswith("…"):
        return JOIN_PUNCT["ellipsis"]
    if t[-1:] in _SENT_END:
        return JOIN_PUNCT["sentence"]
    if t[-1:] in _CLAUSE_END:
        return JOIN_PUNCT["clause"]
    return "word"


def _cut(frames, seconds, sr, keep_end):
    """Keep `seconds` of the concatenated silent `frames` -- the head (keep_end=False) or the
    tail (keep_end=True) -- with a 5 ms fade at the cut so it cannot click."""
    y = np.concatenate(frames) if frames else np.zeros(0, np.float32)
    n = min(y.size, int(round(seconds * sr)))
    y = (y[y.size - n:] if keep_end else y[:n]).copy()
    f = min(n, int(0.005 * sr))
    if f:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        if keep_end:
            y[:f] *= ramp
        else:
            y[n - f:] *= ramp[::-1]
    return y


# ---------------------------------------------------------------------------------------------------------------
# Post-processing of a row's audio: native rate -> out_sr (soxr), tempo (tempo.py), effect (effects.py), <pause>s,
# joined switches, PCM16 -> out_q. One message per emitted chunk:
#   ("emit", rid, uid, y, final, seg_end, gap, k, marks, chained, voice, post)   y = float32 at the native rate
#   ("raw", rid, payload)                                                         passed on to out_q as (rid, payload)
#   ("cfg", {native_sr, out_sr, rs_quality, up, join_hold})                       first, once
# The state lives per line (rid: resampler, join / level / word gaps) and per row (uid = one segment's try: tempo,
# effect, pause sent). It runs in the engine worker itself (TTS_POST_PROCS=0) or in TTS_POST_PROCS processes the API
# starts next to the worker (post_main), every message of a line to the same one (rid % n): the engine loop at 21.5
# frames a second x 100+ rows has no CPU to spare for it (one core: GPU 69 % busy at 128 streams, 2026-10-31).
# ---------------------------------------------------------------------------------------------------------------
class PostStates:
    def __init__(self, cfg=None):
        self.cfg = dict(cfg or {})
        self.lines: dict = {}            # rid -> {rs, held, sent, join, join_gap, lvl, gaps, uids}
        self.rows: dict = {}             # uid -> {tempo, fx, pre}


def _pcm16(y):
    return (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def post_handle(S, msg, put, stat=None, tag="[post]"):
    """One post-processing message (see above); put(item) sends (rid, payload) on to the API."""
    if msg[0] == "cfg":
        S.cfg.update(msg[1])
        return
    if msg[0] == "raw":
        rid, payload = msg[1], msg[2]
        put((rid, payload))
        if payload is None:
            L = S.lines.pop(rid, None)
            for u in (L or {}).get("uids", ()):
                S.rows.pop(u, None)
        return
    _, rid, uid, y, final, seg_end, gap, k, marks, chained, voice, post = msg
    c = S.cfg
    native_sr, out_sr, up, hold = c["native_sr"], c["out_sr"], c["up"], c["join_hold"]
    t0 = time.perf_counter()
    L = S.lines.get(rid)
    if L is None:
        import soxr
        L = S.lines[rid] = {"rs": soxr.ResampleStream(native_sr, out_sr, 1, dtype="float32", quality=c["rs_quality"])
                            if native_sr != out_sr else None, "held": None, "sent": False, "join": None,
                            "join_gap": None, "lvl": 0.0, "gaps": [], "uids": set()}
    R = S.rows.get(uid)
    if R is None:
        R = S.rows[uid] = {"tempo": None, "fx": None, "pre": False}
        L["uids"].add(uid)
    if L["held"] is not None:
        y = np.concatenate([L["held"], y])
        L["held"] = None
    joins = chained and gap is not None              # this segment ends on a joined switch
    if joins and not final and not seg_end and L["sent"] and y.size > hold * up:
        L["held"], y = y[-hold * up:], y[:-hold * up]   # its last word's end must still be here
    if L["rs"] is not None:
        y = L["rs"].resample_chunk(y, last=final)
    if stat:
        t0 = stat("e.resample", t0)
    post = post or {}
    if R["tempo"] is None:   # slowed to TTS_SLOW_TEMPO / TTS_(VI_)NORMAL_TEMPO (tempo.py), <slow> words more (effects.py)
        rate = tempo.rate(voice) * post.get("tempo", 1.0)
        R["tempo"] = False
        if rate != 1.0:
            try:
                R["tempo"] = tempo.Tempo(out_sr, rate)
            except Exception as e:
                print(f"{tag} {rid}: no tempo ({e}); read as it is", flush=True)
    if R["tempo"]:
        y = R["tempo"].push(y) if y.size else np.zeros(0, np.float32)
        if final or seg_end:                          # this voice's segment ends: the rest of it
            y = np.concatenate([y, R["tempo"].flush()])
    if stat:
        t0 = stat("e.tempo", t0)
    if post.get("fx"):                                # the segment's effect (effects.py), streamed
        if R["fx"] is None:
            try:
                R["fx"] = effects.make(post["fx"], out_sr)
            except Exception as e:
                print(f"{tag} {rid}: no effect {post['fx']!r} ({e})", flush=True)
                R["fx"] = False
        if R["fx"]:
            y = R["fx"].process(y) if y.size else y
            if final or seg_end:                      # an echo / a room rings on
                y = np.concatenate([y, R["fx"].flush()])
    if not R["pre"] and post.get("pause_before"):     # <pause> opening the line
        y = np.concatenate([np.zeros(int(post["pause_before"] * out_sr), np.float32), y])
    R["pre"] = True
    if (final or seg_end) and post.get("pause_after"):   # <pause> after it: exactly that much silence
        y = np.concatenate([y, np.zeros(int(post["pause_after"] * out_sr), np.float32)])
    if chained:                                       # joined switches (SWITCH_JOIN)
        if y.size:                                    # the line's loud level and the voice's word gaps so far
            L["lvl"] = max(L["lvl"], float(np.percentile(_envelope(y, out_sr)[0], 95)) if
                           y.size >= 2 * int(_ENV_S * out_sr) else 0.0)
            L["gaps"].extend(_word_gaps(y, out_sr, L["lvl"]))
        if L["join"] is not None and y.size:          # after a joined switch: on into this segment's first word
            on = _speech_start(y, out_sr, L["lvl"])
            if on is not None:
                g = L["join_gap"]
                y = _join(L["join"], y, on, out_sr, _join_gap(L["gaps"][-40:]) if g == "word" else g)
                L["join"] = None
            elif final or seg_end:
                y, L["join"] = np.concatenate([L["join"], y]), None
            else:                                     # its lead only so far: it waits with the tail
                L["join"], y = np.concatenate([L["join"], y]), np.zeros(0, np.float32)
        if joins and seg_end and not final:           # what follows its last word waits for the next segment
            se = _speech_end(y, out_sr, L["lvl"])
            tail, y = y[se:], y[:se]
            L["join"] = tail if L["join"] is None else np.concatenate([L["join"], tail])
            L["join_gap"] = gap
    if stat:
        t0 = stat("e.fx+join", t0)
    if y.size:   # the last chunk tagged: the API's output level fades it out (level.py)
        put((rid, ("__last__", _pcm16(y)) if final else _pcm16(y)))
        L["sent"] = True
        if stat:
            stat("e.put", t0)
    if (final or seg_end) and marks:
        put((rid, ("__seg__", k)))                    # the audio so far ends segment k (timing.py, the API)
    if seg_end or final:
        S.rows.pop(uid, None)
        L["uids"].discard(uid)
    if final:
        put((rid, None))
        S.lines.pop(rid, None)


def post_main(in_q, out_q, idx=0):
    """A post-processing process (TTS_POST_PROCS): post_handle over its in_q, until the API process goes."""
    import threading

    def _watch_parent(ppid):
        while True:
            time.sleep(2)
            if os.getppid() != ppid:
                os._exit(0)
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()
    S = PostStates()
    tag = f"[post {idx}]"
    while True:
        msg = in_q.get()
        try:
            post_handle(S, msg, out_q.put, tag=tag)
        except Exception:
            print(f"{tag} failed:\n" + traceback.format_exc(), flush=True)
            if msg[0] == "emit":                       # the line goes on without that chunk; its end still ends it
                if msg[4]:
                    out_q.put((msg[1], None))


_UID = iter(range(1, 1 << 62))


class _Row:
    __slots__ = ("rid", "slot", "frames", "cap", "pend", "npend", "target", "rs", "started", "uid",
                 "chain", "codes", "vflags", "hold", "voiced", "lead_drop", "whole", "buf", "raw", "shadow", "quiet",
                 "sound_min", "voice", "tempo", "post", "fx", "pre")

    def __init__(self, rid, slot, cap, target, rs, chain=None, shadow=()):
        self.rid, self.slot, self.cap, self.target, self.rs = rid, slot, cap, target, rs
        self.uid = next(_UID)     # this row (one segment's try): its post-processing state (post_handle)
        self.shadow = shadow      # guided group: the slots of its x-vector rows (their audio is unused)
        self.frames = 0
        self.started = False
        self.pend = []
        self.npend = 0
        self.chain = chain        # code-switch state shared by the segments of one line, or None
        self.codes = []           # chain rows: codec frame [16] of every generated frame
        self.vflags = []          # chain rows: whether frame i carries speech
        self.hold = []            # chain rows: silent frames held back (continuation lead / tail)
        self.voiced = False       # chain rows: this segment has produced speech
        self.lead_drop = 0        # chain rows: leading silent frames dropped from this segment
        self.whole = False        # chain rows: hold the whole segment until it ends (short ones)
        self.buf = []             # chain rows: the held audio frames of such a segment
        self.quiet = 0            # chain rows held whole: silent frames since the segment last sounded
        self.sound_min = None     # a sound segment: frames before silence may end it (_sound_min); else None
        self.voice = None         # the voice it reads with: a slow one is slowed to TTS_SLOW_TEMPO (tempo.py)
        self.tempo = None         # ... by this tempo.Tempo, made at its first audio (False: none needed)
        self.post = {}            # the segment's "post" (effects.py): tempo, em, fx, pause_after, pause_before
        self.fx = None            # its effects.Fx, made at its first audio
        self.pre = False          # pause_before sent
        self.raw = []             # TTS_CHAIN_DUMP: (started, audio, token) of every tick


def fast_main(in_q, out_q, cfg: dict):
    """Drive fast_engine.FastEngine: continuous batching over fixed slots, one CUDA-graph replay
    per tick, results read back one tick late so the CPU never stalls the GPU.

    Per iteration: pull arrivals -> admit into free slots (one batched prefill) -> replay the tick
    graph for the smallest bucket covering the busy slots and queue async copies of (codebook-0
    token that tick consumed, 1920 samples of audio) to pinned memory -> process the PREVIOUS
    tick's copies: a row whose consumed token was EOS is finished (its frame is garbage), else its
    frame of audio is appended and emitted once its chunk target is reached. Audio is resampled
    with one soxr stream per request (no chunk-boundary artifacts) and sent as PCM16.

    Code-switch chains (CHAIN messages): each segment is its own generation with its own speaker
    and language; when segment k ends, segment k+1 is admitted into the SAME slot (kept reserved)
    with segment k's codes + text appended to its speaker's ICL reference ("ctx": prev = the
    previous segment only, full = all previous segments, none = no context), so it continues right
    after what was said, and it keeps the slot's vocoder state so the waveform has no seam.
    Silence is gated per segment: a non-final segment's trailing silence is held back, and it and
    the next segment's leading silence are cut to a few frames each by the punctuation at the
    switch (_keep), so an unpunctuated switch sounds like a word gap, not a comma; a continuation
    that stays silent longer than LEAD_MAX frames (or ends silent) is resampled. Each segment is
    capped by its text length (_seg_cap: a runaway re-reads the prompt); a short continuation
    (<= SHORT_CHARS) is held whole until it ends so a runaway can be resampled unheard, a longer
    one (or the first segment, streamed for TTFA) is cut at the cap."""
    gpu = cfg.get("gpu")
    if gpu is not None and str(gpu) != "":
        # before torch initialises CUDA; PCI order so the id means the same GPU as in nvidia-smi
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import threading

    def _watch_parent(ppid):
        while True:
            time.sleep(2)
            if os.getppid() != ppid:
                os._exit(0)
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()

    import heapq
    import soxr
    import torch
    from tts_engine import TTSEngine

    wid = cfg.get("worker_id", 0)
    tag = f"[fast_worker {wid} gpu={gpu}]"
    eng = TTSEngine(
        cfg["model"], device="cuda:0" if gpu is not None else cfg["device"],
        speakers_dir=cfg["speakers_dir"], attn_implementation=cfg["attn"],
        gen_params=cfg["gen_params"], batch_size=cfg["batch_size"])
    fe = eng.fast
    global FRAME_SCALE, LEAD_MAX, SFX_QUIET, SFX_TAIL, JOIN_HOLD
    FRAME_SCALE = float(getattr(fe, "fps", 12.5)) / 12.5
    LEAD_MAX, SFX_QUIET, SFX_TAIL, JOIN_HOLD = (max(1, round(n * FRAME_SCALE)) for n in (LEAD_MAX, SFX_QUIET, SFX_TAIL,
                                                                                        JOIN_HOLD))
    # the vocoder snapshots on disk (made once), in memory the pinned voices + TTS_SNAP_RAM recently used
    # (fast_engine.SnapStore); TTS_SNAP_DIR=none keeps every one in memory
    snap_dir = os.environ.get("TTS_SNAP_DIR", "").strip() or cfg["speakers_dir"].rstrip("/") + ".snaps"
    if snap_dir.lower() != "none":
        from fast_engine import SnapStore
        fe.voc_snaps = SnapStore(snap_dir, int(os.environ.get("TTS_SNAP_RAM", "2048")))
    director = os.environ.get("TTS_GUIDE_DIRECTOR", "").strip()
    if GUIDE and director:
        fe.attach_director(os.path.expanduser(director), os.environ.get("TTS_GUIDE_DIRECTOR_SPEAKERS", "speakers_17b_v3"),
                           log=lambda s: print(f"{tag} {s}", flush=True))
    CROSS = fe.dir is not None
    fe.capture(log=lambda s: print(f"{tag} {s}", flush=True))
    t_w = time.perf_counter()
    warm_langs = [l.strip() for l in os.environ.get("TTS_WARM_LANGS", "English,Auto,Chinese,Vietnamese").split(",") if l.strip()]
    items = {}
    for name in eng.list_speakers():
        try:
            items[name] = eng._item_for(name)
        except Exception as e:
            print(f"{tag} speaker load failed for {name}: {e}", flush=True)
    try:
        fe.prime_speakers(list(items.items()))     # all rows are free: B references at a time
    except Exception as e:                         # one at a time below (and on first use)
        print(f"{tag} batched speaker priming failed: {e}", flush=True)
    pool = VoicePool(list(items), evict=fe.evict_voices, log=lambda s: print(f"{tag} {s}", flush=True))
    fe.voc_snaps.resident = {n for n in items if pool.is_pinned(n)}    # the pinned voices stay in memory
    for n in fe.voc_snaps.resident:
        fe.voc_snaps.get(n)
    warmed = 0
    for name, item in items.items():
        try:
            fe.prime_speaker(name, item)           # no-op when primed above
            # JIT the prefill kernels and pre-compute the prompt-prefix KV of the pinned voices per
            # language (slot 0 is scratch here: it's free, the next real admission overwrites it); the
            # other voices get theirs on first use (VoicePool)
            if pool.is_pinned(name) or not warmed:
                for lang in _voice_warm_langs(name, warm_langs):
                    fe.admit([0], [{"text": "Warm up.", "lang": eng._resolve_lang(lang), "item": item, "key": name}])
                warmed += 1
        except Exception as e:
            print(f"{tag} speaker prime/warmup failed for {name}: {e}", flush=True)
    torch.cuda.synchronize()
    snap_mb = sum(t.numel() * t.element_size() for s, _ in fe.voc_snaps.values() for t in s) / 2 ** 20
    on_disk = f", {fe.voc_snaps.on_disk()} on disk in {fe.voc_snaps.path}" if fe.voc_snaps.path else ""
    print(f"{tag} {len(items)} voices primed ({snap_mb:.0f} MB of vocoder snapshots in host memory{on_disk}), "
          f"{warmed} warmed in {time.perf_counter()-t_w:.1f}s; pinned families "
          f"{len(pool.pinned)} ({sum(len(pool.members[f]) for f in pool.pinned)} voices, "
          f"{fe.voice_state_bytes() / 2 ** 20:.0f} MB GPU prompt state), {VOICE_SLOTS} slots for the other "
          f"{len(pool.members) - len(pool.pinned)} families"
          + (f"; pinned but no voice on disk: {', '.join(pool.missing)}" if pool.missing else ""), flush=True)

    B, eos, up = fe.B, fe.eos, fe.up
    native_sr, out_sr = eng._native_sr(), int(cfg["out_sr"])
    fpc = max(1, int(cfg["frames_per_chunk"]))
    first_fpc = max(1, int(os.environ.get("TTS_FIRST_CHUNK_FRAMES", "1")))
    gather_s = float(os.environ.get("TTS_ADMIT_GATHER_MS", "4")) / 1000.0
    rs_quality = os.environ.get("TTS_RESAMPLE_QUALITY", "HQ")
    NBUF = 4
    tok_h = torch.empty(NBUF, B, dtype=torch.long, pin_memory=True)
    aud_h = torch.empty(NBUF, B, up, dtype=torch.float32, pin_memory=True)
    fr_h = torch.empty(NBUF, B, fe.ncg, dtype=torch.long, pin_memory=True)
    events = [torch.cuda.Event() for _ in range(NBUF)]

    slot_rid = [None] * B
    free = list(range(B)); heapq.heapify(free)
    rows: dict = {}
    waiting: deque = deque()
    inflight: deque = deque()
    cont: list = []            # code-switch continuations to admit: (rid, reserved slot, req, chain)
    tick = 0

    # TTS_WORKER_STATS=1: where the loop's time goes, printed every 10 s (ms of each part per second of wall time)
    STATS = os.environ.get("TTS_WORKER_STATS", "0").strip().lower() not in ("0", "", "false", "no", "off")
    st = {}
    st_t0 = [time.perf_counter()]

    def stat(k, t0):
        t = time.perf_counter()
        st[k] = st.get(k, 0.0) + t - t0
        return t

    def stat_flush():
        now = time.perf_counter()
        dt = now - st_t0[0]
        if dt < 10:
            return
        ticks = st.pop("#ticks", 0.0)
        rows_n = st.pop("#rows", 0.0)
        parts = "  ".join(f"{k} {1000 * v / dt:.0f}" for k, v in sorted(st.items()))
        print(f"{tag} [stats] {ticks / dt:.1f} ticks/s, {rows_n / max(1, ticks):.0f} rows/tick | ms/s: {parts}", flush=True)
        st.clear()
        st_t0[0] = now

    # post-processing (post_handle): in TTS_POST_PROCS processes the API started (cfg["post_qs"], every message of a
    # line to the same one), else here
    post_qs = list(cfg.get("post_qs") or [])
    pst = PostStates(dict(native_sr=native_sr, out_sr=out_sr, rs_quality=rs_quality, up=up, join_hold=JOIN_HOLD))
    for q in post_qs:
        q.put(("cfg", pst.cfg))

    def post_put(rid, msg):
        if post_qs:
            post_qs[(rid if isinstance(rid, int) else zlib.crc32(str(rid).encode())) % len(post_qs)].put(msg)
        else:
            post_handle(pst, msg, out_q.put, stat=stat, tag=tag)

    def send(rid, payload):
        """(rid, payload) to the API, in order with the line's audio."""
        post_put(rid, ("raw", rid, payload))

    def emit(r, final, seg_end=False):
        t_e = time.perf_counter()
        ch = r.chain
        gap = _switch_gap(ch["segs"], ch["k"]) if ch is not None else None    # this segment ends on a joined switch
        y = np.concatenate(r.pend) if r.pend else np.zeros(0, np.float32)
        r.pend = []
        r.npend = 0
        post_put(r.rid, ("emit", r.rid, r.uid, y, final, seg_end, gap, ch["k"] if ch is not None else 0,
                         bool(ch is not None and ch.get("marks")), ch is not None, r.voice, r.post or {}))
        stat("emit", t_e)

    guided = {}                # cross-model: slot -> frames its director rows hold (the tick needs their rows)
    cost = {}                  # slot -> compute units it holds (MAX_UNITS)

    def steer(result):
        """Slots guide_cross now steers: their director budget, and they cost GUIDE_UNITS."""
        guided.update(result)
        for s in result:
            cost[s] = GUIDE_UNITS

    def unsteer(s):
        guided.pop(s, None)
        if s in cost:
            cost[s] = 1.0

    def release(slot, shadow=()):
        for s in (slot, *shadow):
            slot_rid[s] = None
            heapq.heappush(free, s)
            cost.pop(s, None)
        if GUIDE:
            fe.clear_groups([slot, *shadow])
            guided.pop(slot, None)

    def finish(r):
        emit(r, True)
        release(r.slot, r.shadow)
        rows.pop(r.rid, None)

    def fail(rid, slot, msg, shadow=()):
        send(rid, ("__error__", msg)); send(rid, None)
        if slot is not None:
            release(slot, shadow)

    def groupable(req, ch):
        it = req["item"]
        return bool(ch is not None and ch.get("guide") and it.icl_mode and it.ref_code is not None and it.ref_text)

    def seg_rows(req, ch):
        """The engine rows of one segment: itself, or -- an instructed line in guide mode, a voice with a
        reference -- the voice's plain ICL prompt (the anchor, whose audio is sent), plus, same-model, its
        x-vector prompt with the instruction and without (fast_engine.guide_groups; cross-model those two
        are the director's rows, guide_cross)."""
        if not groupable(req, ch):
            return [req]
        anchor = dict(req, instruct="", xvec=False)
        return [anchor] if CROSS else [anchor, dict(req, xvec=True), dict(req, instruct="", xvec=True)]

    # admissions: admit() (new lines) and admit_cont() (a line's next segment) each queue their rows in `batch`;
    # run_batch() prefills them all in ONE engine call (a prefill's fixed cost -- ~500 kernel launches -- was most of
    # the loop's time at 128 streams, 2026-10-31) and hands each its prompt lengths
    batch = []
    ADMIT_EVERY = max(1, int(os.environ.get("TTS_ADMIT_EVERY", "2")))
    ADMIT_BATCH = max(1, int(os.environ.get("TTS_ADMIT_BATCH", "8")))
    last_admit = [-ADMIT_EVERY]

    def run_batch():
        if not batch:
            return
        jobs = list(batch)
        batch.clear()
        slots_all = [s for sl, _, _, _ in jobs for s in sl]
        reqs_all = [q for _, rq, _, _ in jobs for q in rq]
        try:
            lens = fe.admit(slots_all, reqs_all)
        except Exception:
            print(f"{tag} admit failed:\n" + traceback.format_exc(), flush=True)
            for _, _, _, on_fail in jobs:
                on_fail()
            return
        o = 0
        for _, rq, on_lens, _ in jobs:
            on_lens(lens[o:o + len(rq)])
            o += len(rq)

    def admit():
        take, nfree, npairs = [], len(free), fe.pairs_free()
        uroom = MAX_UNITS - sum(cost.values()) if MAX_UNITS > 0 else float("inf")
        while waiting:
            rid, text, lang, speaker = waiting[0]
            styled = GUIDE and speaker == CHAIN and bool(text.get("instruct"))
            need = 3 if styled and not CROSS else 1  # a same-model guided line holds three slots to its end
            steered = styled and CROSS
            units = GUIDE_UNITS if steered else need
            if need > nfree or (steered and npairs < 1) or (units > uroom and (rows or take)):
                break                                # first come, first served: the rest waits too
            waiting.popleft()
            chain = None
            try:
                if speaker == CHAIN:
                    segs = [(s["text"], s["lang"], s["speaker"], s.get("ctx"), bool(s.get("sfx")), s.get("stretch"),
                             s.get("post") or {}) for s in text["segments"]]
                    chain = {"segs": segs, "k": 0, "mode": text.get("ctx", "prev"), "hist": [], "tries": 0,
                             "instruct": text.get("instruct") or "", "guide": styled, "shadow": (),
                             "marks": bool(text.get("marks"))}
                    text, lang, speaker = segs[0][:3]
                item = eng._item_for(speaker)
                if speaker not in fe.voc_snaps:
                    fe.prime_speaker(speaker, item)
                take.append((rid, {"text": text, "lang": eng._resolve_lang(lang), "item": item, "key": speaker,
                                   "instruct": chain["instruct"] if chain else ""}, chain))
                nfree -= need
                npairs -= steered
                uroom -= units
            except Exception as e:
                send(rid, ("__error__", str(e))); send(rid, None)
        if not take:
            return
        pool.use([r["key"] for _, r, _ in take])       # a family out of the slots is dropped first
        held, a_slots, a_reqs, spans = [], [], [], []
        for _, req, chain in take:
            ss = [heapq.heappop(free) for _ in range(3 if chain is not None and chain["guide"] and not CROSS else 1)]
            rr = seg_rows(req, chain)
            held.append(ss)
            spans.append((len(a_reqs), len(rr)))
            a_slots += ss[:len(rr)]
            a_reqs += rr

        def on_fail():
            for (rid, _, _), ss in zip(take, held):
                send(rid, ("__error__", "engine error")); send(rid, None)
                for s in ss:
                    heapq.heappush(free, s)
        batch.append((a_slots, a_reqs, lambda lens: admitted(take, held, spans, lens), on_fail))

    def admitted(take, held, spans, lens):
        groups, ungroup, cross = [], [], []
        for (rid, req, chain), ss, (o, n) in zip(take, held, spans):
            ls = lens[o:o + n]
            if min(ls) < 0:
                send(rid, ("__error__", "text too long for TTS_MAX_SEQ")); send(rid, None)
                for s in ss:
                    heapq.heappush(free, s)
                continue
            cap = max(1, min(fe.max_new, fe.S - 2 - max(ls)))
            rs = None                      # (the line's resampler lives with its post-processing: post_handle)
            if chain is not None:
                chain["rs"] = rs
                chain["shadow"] = tuple(ss[1:])
                seg0 = chain["segs"][0]
                cap = min(cap, _sound_cap(seg0[0], req["item"]) if seg0[4] else _seg_cap(seg0[0]))
            if n == 3:
                groups.append((ss[0], ss[1], ss[2], guide_weight(req["lang"])))
            elif len(ss) > 1:
                ungroup += ss
            elif CROSS and groupable(req, chain):
                cross.append((ss[0], req, guide_weight(req["lang"])))
            rows[rid] = _Row(rid, ss[0], cap, first_fpc, rs, chain, tuple(ss[1:]))
            rows[rid].voice = req["key"]
            if chain is not None:
                rows[rid].post = chain["segs"][0][6]
            if chain is not None and chain["segs"][0][4]:      # a sound opening the line: held whole too
                rows[rid].whole, rows[rid].sound_min = True, _sound_min(req["item"])
            elif chain is not None and (chain["segs"][0][5] or chain["segs"][0][6].get("em")):
                rows[rid].whole = True                         # ... a drawn-out word's / a stressed one's (at its end)
            for s in ss:
                slot_rid[s] = rid
                cost[s] = 1.0
        fe.clear_groups(ungroup)
        fe.guide_groups(groups)
        steer(fe.guide_cross(cross))

    # ---------------- code-switch chains ----------------
    def queue_segment(r):
        """(Re)queue segment chain["k"] of r's line into r's slot, which stays reserved. A continuation that
        came out silent CTX_DROP_TRY times is retried without the previous segments as context (from the
        voice's own reference, like ctx=none): one audible restart at that switch instead of a lost span."""
        ch = r.chain
        text, lang, spk, seg_ctx = ch["segs"][ch["k"]][:4]
        mode = seg_ctx or ch["mode"]      # a segment may override the line's context mode
        hist = ch["hist"] if mode == "full" else ch["hist"][-1:] if mode == "prev" else []
        if hist and ch["tries"] >= CTX_DROP_TRY:
            hist = []
            if ch["tries"] == CTX_DROP_TRY:
                print(f"{tag} chain {r.rid} segment {ch['k']}: silent {CTX_DROP_TRY}x with context, retried without",
                      flush=True)
        req = {"text": text, "lang": eng._resolve_lang(lang), "item": eng._item_for(spk), "key": spk,
               "keep_voc": True, "instruct": ch["instruct"]}
        if hist:
            req["ctx_codes"] = torch.from_numpy(np.concatenate([c for c, _ in hist]))
            req["ctx_text"] = " ".join(t for _, t in hist)
        rows.pop(r.rid, None)             # the segment gets a fresh row under the same rid
        cont.append((r.rid, r.slot, req, ch))

    def admit_cont():
        items = list(cont)
        cont.clear()
        pool.use([q["key"] for _, _, q, _ in items])
        a_slots, a_reqs, spans = [], [], []
        for _, s, q, ch in items:
            rr = seg_rows(q, ch)
            spans.append((len(a_reqs), len(rr)))
            a_slots += [s, *ch.get("shadow", ())][:len(rr)]
            a_reqs += rr

        def on_fail():
            for rid, s, _, ch in items:
                fail(rid, s, "engine error", ch.get("shadow", ()))
        batch.append((a_slots, a_reqs, lambda lens: cont_admitted(items, spans, a_slots, lens), on_fail))

    def cont_admitted(items, spans, a_slots, lens):
        groups, ungroup, cross = [], [], []
        for (rid, s, req, ch), (o, n) in zip(items, spans):
            ls = lens[o:o + n]
            if min(ls) < 0 and req.get("ctx_codes") is not None:     # context too long: go on without it
                req = {k: v for k, v in req.items() if k not in ("ctx_codes", "ctx_text")}
                ls = fe.admit(a_slots[o:o + n], seg_rows(req, ch))
            shadow = ch.get("shadow", ())
            if min(ls) < 0:
                fail(rid, s, "text too long for TTS_MAX_SEQ", shadow)
                continue
            sound = ch["segs"][ch["k"]][4]
            cap = max(1, min(fe.max_new, fe.S - 2 - max(ls),
                             _sound_cap(req["text"], req["item"]) if sound else _seg_cap(req["text"])))
            if n == 3:
                groups.append((s, shadow[0], shadow[1], guide_weight(req["lang"])))
            elif shadow:
                ungroup += [s, *shadow]
            elif CROSS and ch.get("guide"):          # the previous segment's steering ends here
                ungroup.append(s)
                unsteer(s)
                if groupable(req, ch):
                    cross.append((s, req, guide_weight(req["lang"])))
            r = _Row(rid, s, cap, first_fpc, ch["rs"], ch, shadow)
            r.voice = req["key"]
            r.post = ch["segs"][ch["k"]][6]
            r.whole = text_units(req["text"]) <= SHORT_CHARS or sound or bool(ch["segs"][ch["k"]][5]) \
                or bool(r.post.get("em"))
            r.sound_min = _sound_min(req["item"]) if sound else None
            rows[rid] = r
        fe.clear_groups(ungroup)                     # (frees their director pairs before guide_cross takes them)
        fe.guide_groups(groups)
        steer(fe.guide_cross(cross))

    def chain_audio(r, a, v=None):
        """Gate one frame of a chain row's audio (v: whether it sounds, if already known); returns True if the segment
        must be resampled."""
        ch = r.chain
        if v is None:
            v = float(np.sqrt(np.mean(a * a))) > SIL_AMP
        if r.whole:                         # held until it ends, gated then (end_segment)
            r.buf.append(a)
            r.voiced = r.voiced or v
            r.quiet = 0 if v or not r.voiced else r.quiet + 1
            return not r.voiced and len(r.buf) > LEAD_MAX and ch["tries"] < TRIES_MAX
        r.vflags.append(v)
        if not r.voiced:
            if v:
                r.voiced = True
                if ch["k"] > 0 and r.hold:
                    lead_s = _seg_keep(ch["segs"][ch["k"] - 1])[1]
                    r.lead_drop = max(0, len(r.hold) - math.ceil(lead_s * native_sr / up))
                    r.pend.append(_cut(r.hold, lead_s, native_sr, keep_end=True)); r.npend += 1
                else:
                    r.pend.extend(r.hold); r.npend += len(r.hold)
                r.hold = []
                r.pend.append(a); r.npend += 1
            elif ch["k"] == 0:              # the line's first segment is never gated (TTFA)
                r.pend.append(a); r.npend += 1
            else:
                r.hold.append(a)
                if len(r.hold) > LEAD_MAX and ch["tries"] < TRIES_MAX:
                    return True
        elif v:
            if r.hold:                      # a pause inside the segment: release it
                r.pend.extend(r.hold); r.npend += len(r.hold); r.hold = []
            r.pend.append(a); r.npend += 1
        elif ch["k"] == len(ch["segs"]) - 1:
            r.pend.append(a); r.npend += 1
        else:
            r.hold.append(a)                # maybe the segment's trailing silence
        return False

    def retry_segment(r):
        r.chain["tries"] += 1
        queue_segment(r)

    def end_segment(r, runaway=False):
        """Segment done (EOS, or its frame cap = runaway). A short segment held whole is resampled
        on a runaway (nothing of it was sent yet), else gated now as if it had been streamed."""
        ch = r.chain
        k = ch["k"]
        if CHAIN_DUMP and r.raw:
            seg = ch["segs"][k]
            np.savez(os.path.join(CHAIN_DUMP, f"{r.rid}_{k}_{ch['tries']}.npz"),
                     audio=np.stack([a for _, a, _ in r.raw]), started=np.array([s for s, _, _ in r.raw]),
                     toks=np.array([t for _, _, t in r.raw]),
                     codes=np.stack(r.codes) if r.codes else np.zeros((0, fe.ncg), np.int64),
                     text=seg[0], lang=seg[1], speaker=seg[2], runaway=runaway, sr=native_sr)
        if r.whole:
            # a run-on is resampled, and so is a sound that never sounded (the first segment too: nothing of it sent)
            if (runaway or (not r.voiced and ch["segs"][k][4])) and ch["tries"] < TRIES_MAX:
                retry_segment(r)
                return
            if ch["segs"][k][4]:
                # a sound without its dead lead (a line opening with one is not gated: up to 1 s of it measured) and
                # tail (SFX_TAIL frames kept); silent on every try: left out
                loud = [i for i, a in enumerate(r.buf) if float(np.sqrt(np.mean(a * a))) > SIL_AMP]
                r.buf = r.buf[max(0, loud[0] - 1):loud[-1] + 1 + SFX_TAIL] if loud else []
            buf, r.buf, r.whole, r.voiced = r.buf, [], False, False
            flags = None
            em = bool((r.post or {}).get("em"))
            if (ch["segs"][k][5] or em) and buf:
                # a drawn-out word ends the segment: its vowel held (elongate.py); a stressed word (<em>): longer,
                # higher, louder (effects.emphasize). The context codes keep the voiced flags of the frames as
                # generated.
                flags = [float(np.sqrt(np.mean(a * a))) > SIL_AMP for a in buf]
                y = y0 = np.concatenate(buf)
                if ch["segs"][k][5]:
                    try:
                        tr = {}
                        y = elongate.stretch(y0, native_sr, ch["segs"][k][0], ch["segs"][k][5], tr)
                        at = "no voice to hold" if tr.get("at") is None else f"at {tr['at']:.2f}s"
                        print(f"{tag} {r.rid}: held {ch['segs'][k][0][-24:]!r} {at} of {len(y0) / native_sr:.2f}s, "
                              f"+{(len(y) - len(y0)) / native_sr:.2f}s", flush=True)
                    except Exception:
                        print(f"{tag} {r.rid}: stretch failed:\n" + traceback.format_exc(), flush=True)
                if em:
                    try:
                        y = effects.emphasize(y, native_sr)
                    except Exception:
                        print(f"{tag} {r.rid}: emphasis failed:\n" + traceback.format_exc(), flush=True)
                buf = [y[i:i + up] for i in range(0, len(y), up)]
            for a in buf:
                chain_audio(r, a)
            if flags is not None:
                r.vflags = flags
        if not r.voiced and k > 0 and ch["tries"] < TRIES_MAX:     # came out silent: resample
            retry_segment(r)
            return
        if k == len(ch["segs"]) - 1:
            r.pend.extend(r.hold); r.npend += len(r.hold); r.hold = []
            finish(r)
            return
        text = ch["segs"][k][0]
        tail_s = _seg_keep(ch["segs"][k])[0]
        kt = math.ceil(tail_s * native_sr / up)        # frames of it kept in the context codes
        y = _cut(r.hold, tail_s, native_sr, keep_end=False)
        pad = int(round(_seg_pad(ch["segs"][k]) * native_sr)) - y.size
        if pad > 0:                                    # a comma / sentence end the voice ran on through
            y = np.concatenate([y, np.zeros(pad, np.float32)])
        if y.size:
            r.pend.append(y); r.npend += 1
        r.hold = []
        emit(r, False, seg_end=True)
        n = min(len(r.codes), len(r.vflags))
        vi = [i for i in range(n) if r.vflags[i]]
        end = min(n, vi[-1] + 1 + kt) if vi else n
        # a sound (sfx) is no context for what follows: the speech before it is (a clip that short as the
        # context makes some voices run on, like a few words of another language)
        if end > r.lead_drop and not ch["segs"][k][4]:
            ch["hist"].append((np.stack(r.codes[r.lead_drop:end]), text))
        ch["k"] += 1
        ch["tries"] = 0
        queue_segment(r)

    out_q.put((READY, eng.list_speakers()))
    print(f"{tag} ready (slots={B}, buckets={fe.buckets}, max_seq={fe.S}, first_chunk={first_fpc}f, "
          f"chunk={fpc}f, sr={out_sr}"
          + (f", director pairs={fe.pairs}, units: steered {GUIDE_UNITS:g} / max {MAX_UNITS:g}" if CROSS else "") + ")",
          flush=True)

    def intake(msg):
        if msg[0] == "__cancel__":                          # client disconnected
            rid = msg[1]
            r = rows.get(rid)
            if r is not None:
                finish(r)
                return
            for i, w in enumerate(waiting):
                if w[0] == rid:
                    del waiting[i]
                    send(rid, None)
                    return
            for i, c in enumerate(cont):                    # between two segments of a chain
                if c[0] == rid:
                    del cont[i]
                    send(rid, None)
                    release(c[1], c[3].get("shadow", ()))
                    return
        else:
            waiting.append(msg)

    while True:
        try:
            while True:
                intake(in_q.get_nowait())
        except _queue.Empty:
            pass
        if not rows and not inflight and not waiting and not cont:
            intake(in_q.get())                              # idle: block for work
            deadline = time.perf_counter() + gather_s       # then gather a burst briefly
            while True:
                t = deadline - time.perf_counter()
                if t <= 0:
                    break
                try:
                    intake(in_q.get(timeout=t))
                except _queue.Empty:
                    break
            if not waiting:
                continue
        t_l = time.perf_counter()
        # while rows are running, arrivals -- new lines and a line's next segment -- are admitted every ADMIT_EVERY
        # ticks (or as soon as ADMIT_BATCH wait), several to one prefill: at most a tick (~30-40 ms) more to their first
        # audio. A continuation's slot waits safely: an ended row keeps consuming EOS (FastEngine._tick), so only
        # silence runs through the vocoder state it keeps
        if (cont or (waiting and free)) and (not rows or tick - last_admit[0] >= ADMIT_EVERY
                                            or len(waiting) + len(cont) >= ADMIT_BATCH):
            if cont:                                        # a listener is already mid-line: first
                admit_cont()
            if waiting and free:
                admit()
            run_batch()
            last_admit[0] = tick
        t_l = stat("admit", t_l)
        busy = [s for s in range(B) if slot_rid[s] is not None]
        if busy:
            b = fe.bucket_for(busy[-1] + 1)
            st["#ticks"] = st.get("#ticks", 0.0) + 1
            st["#rows"] = st.get("#rows", 0.0) + len(busy)
            try:
                fe.replay(b, fe.dir_bucket() if CROSS and guided else 0)   # the director's pairs in use, if any
            except Exception:
                print(f"{tag} tick failed:\n" + traceback.format_exc(), flush=True)
                for r in list(rows.values()):
                    send(r.rid, ("__error__", "engine error")); finish(r)
                inflight.clear()
                continue
            j = tick % NBUF
            tok_h[j, :b].copy_(fe.tok_in[:b], non_blocking=True)
            aud_h[j, :b].copy_(fe.audio[:b], non_blocking=True)
            fr_h[j, :b].copy_(fe.frame_prev[:b], non_blocking=True)
            events[j].record()
            inflight.append((j, [(s, slot_rid[s], rows.get(slot_rid[s])) for s in busy]))
            tick += 1
        t_l = stat("replay", t_l)
        while inflight and (len(inflight) > 1 or not busy):
            j, snap = inflight.popleft()
            t_w = time.perf_counter()
            events[j].synchronize()
            t_l = stat("wait", t_w)
            # one copy of the tick's rows (the pinned buffers are reused NBUF ticks later), each row a view of it; the
            # tokens as Python ints, every row's loudness at once
            nb = (snap[-1][0] + 1) if snap else 0
            toks = tok_h[j, :nb].tolist()
            aud = aud_h[j, :nb].numpy().copy()
            frs = fr_h[j, :nb].numpy().copy()
            loud = (np.sqrt(np.mean(aud * aud, axis=1)) > SIL_AMP).tolist()
            # tick t's audio is frame t-1's (vocoder runs one frame behind, see FastEngine._tick);
            # toks[s] is codebook-0 of frame t (frs[s] is frame t): EOS there means frame t-1 was
            # the last real one.
            for s, rid, row in snap:
                r = rows.get(rid)
                if r is None or r is not row or s != r.slot:   # finished, or a later segment owns the rid
                    continue                                   # now; or a guided group's x-vector row
                if CHAIN_DUMP and r.chain is not None:
                    r.raw.append((r.started, aud[s], int(toks[s])))
                if r.started:
                    r.frames += 1
                    if r.chain is None:
                        r.pend.append(aud[s])
                        r.npend += 1
                    elif chain_audio(r, aud[s], loud[s]):
                        retry_segment(r)
                        continue
                else:
                    r.started = True                   # first tick: audio is the ref's last frame
                if s in guided and r.frames >= guided[s]:
                    # its director rows' KV is full (TTS_GUIDE_DIRECTOR_MAX_SEQ): the line goes on unsteered,
                    # in the style it has
                    fe.clear_groups([s])             # (its director pair goes back to the pool)
                    unsteer(s)
                    print(f"{tag} {r.rid}: steering ended at frame {r.frames} (director KV full)", flush=True)
                if r.chain is not None and toks[s] != eos:
                    r.codes.append(frs[s])
                # a sound that has sounded, run most of its clip's length and gone quiet is done: no EOS needed
                quiet_end = r.sound_min is not None and r.quiet >= SFX_QUIET and r.frames >= r.sound_min
                if toks[s] == eos or r.frames >= r.cap or quiet_end:
                    if r.chain is None:
                        finish(r)
                    else:
                        end_segment(r, runaway=toks[s] != eos and not quiet_end)
                elif r.npend >= r.target:
                    emit(r, False)
                    r.target = min(fpc, r.target * 2)
            stat("rows(incl emit)", t_l)
        if STATS:
            stat_flush()
