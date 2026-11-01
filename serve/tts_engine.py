#!/usr/bin/env python3
"""
Model + voice loading for the serving stack.

TTSEngine loads the Audio8 checkpoint (models/Audio8-TTS-Realistic-0.6b: ArkttsModel + its processor, Hugging Face
custom code), the precomputed voices from a speakers folder (<name>.pt = the reference's codec codes + transcript, made
by create_speakers.py, so no reference audio is encoded at request time), and builds the FastEngine (fast_engine.py)
that engine_worker.fast_main() drives in its own process.
"""
from __future__ import annotations

import glob
import os
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch

from fast_engine import FastEngine

# ISO codes / short names accepted for the language parameter. The checkpoint has no language id: the text alone says
# what language it is, so the name only travels along (normalizers, logs) and an unknown one is "Auto".
LANG_ALIASES = {"vi": "Vietnamese", "vie": "Vietnamese", "vn": "Vietnamese", "tiếng việt": "Vietnamese",
                "en": "English", "eng": "English", "zh": "Chinese", "cn": "Chinese", "ja": "Japanese",
                "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
                "ru": "Russian", "it": "Italian", "ar": "Arabic", "fi": "Finnish", "hi": "Hindi", "id": "Indonesian",
                "ms": "Malay", "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "sv": "Swedish", "th": "Thai",
                "tr": "Turkish", "yue": "Cantonese"}
LANGS = {"auto", "vietnamese", "english", "chinese", "japanese", "korean", "french", "german", "spanish", "portuguese",
         "russian", "italian", "arabic", "finnish", "hindi", "indonesian", "malay", "dutch", "norwegian", "polish",
         "swedish", "thai", "turkish", "cantonese"}


def load_model(model_path: str, device: str = "cuda:0"):
    """(ArkttsModel in bf16 on `device`, its processor)."""
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16).eval().to(device)
    return model, proc


def voice_item(ref_code, ref_text):
    """A voice as the engine reads it: the reference's codes [T, 10] and its transcript."""
    return SimpleNamespace(ref_code=ref_code, ref_text=ref_text, icl_mode=True, x_vector_only_mode=False,
                           ref_spk_embedding=None)


class TTSEngine:
    def __init__(self, model_path: str, device: str = "cuda:0", speakers_dir: str = "speakers",
                 attn_implementation: str = "sdpa", gen_params: Optional[dict] = None,
                 batch_size: int = 16):
        self.gen_params = dict(gen_params or {})   # decoding params from .env (config.gen_params())
        self.speakers_dir = speakers_dir
        print(f"[engine] loading model: {model_path} (device={device}) ...", flush=True)
        self.model, self.processor = load_model(model_path, device)
        self.fast = FastEngine(self.model, self.processor.tokenizer, self.gen_params, max_slots=batch_size)
        self.supported_langs = set(LANGS)
        self.speakers: Dict[str, dict] = {}
        self.load_speakers(speakers_dir)
        print(f"[engine] ready. {len(self.speakers)} speakers | langs={sorted(self.supported_langs)}", flush=True)
        if self.gen_params:
            print(f"[engine] gen_params={self.gen_params}", flush=True)

    # ---- speakers ----
    def load_speakers(self, speakers_dir: str) -> None:
        self.speakers = {}
        names = set()
        for p in glob.glob(os.path.join(speakers_dir, "*.pt")):
            names.add(os.path.splitext(os.path.basename(p))[0])
        for p in glob.glob(os.path.join(speakers_dir, "*.wav")):
            names.add(os.path.splitext(os.path.basename(p))[0])
        for name in sorted(names):
            try:
                self.speakers[name] = self._load_speaker(name, speakers_dir)
            except Exception as e:
                print(f"[engine] WARN: failed to load speaker '{name}': {e}", flush=True)

    @staticmethod
    def _close_ref_text(t: str) -> str:
        """End the reference transcript as a sentence. The model continues the reference (ICL): with no final stop
        the target is read as the same sentence and a vowel-initial first word merges with the reference's last sound
        (Qwen3-TTS: Ích -> Xích, Ô tô -> Cô tô, Ừ dropped)."""
        t = (t or "").strip()
        return t + "." if t and t[-1] not in ".!?…。！？\"'”’)" else t

    def _load_speaker(self, name: str, d: str) -> dict:
        pt = os.path.join(d, name + ".pt")
        if os.path.exists(pt):
            p = torch.load(pt, map_location="cpu", weights_only=True)
            item = voice_item(p["ref_code"].long(), self._close_ref_text(p.get("ref_text", "")))
            return {"item": item, "ref_audio": None, "ref_text": item.ref_text, "precomputed": True}
        wav = os.path.join(d, name + ".wav")
        if os.path.exists(wav):
            txt = os.path.join(d, name + ".txt")
            ref_text = ""
            if os.path.exists(txt):
                with open(txt, encoding="utf-8") as f:
                    ref_text = self._close_ref_text(f.read())
            return {"item": None, "ref_audio": wav, "ref_text": ref_text, "precomputed": False}
        raise FileNotFoundError(f"no {name}.pt or {name}.wav in {d}")

    def list_speakers(self) -> List[str]:
        return sorted(self.speakers)

    def _resolve_lang(self, lang: Optional[str]) -> str:
        if lang:
            lang = LANG_ALIASES.get(str(lang).lower(), lang)
            if str(lang).lower() in self.supported_langs:
                return lang
        return "Auto"

    def _item_for(self, speaker: str):
        """The voice's prompt item (its reference encoded once if it has no .pt)."""
        if speaker not in self.speakers:
            raise KeyError(f"unknown speaker '{speaker}'. available: {len(self.speakers)} voices")
        sp = self.speakers[speaker]
        if sp["item"] is None:
            from create_speakers import encode_refs
            codes = encode_refs(self.model, [sp["ref_audio"]])[0]
            sp["item"] = voice_item(codes, sp["ref_text"])
        return sp["item"]

    def _native_sr(self) -> int:
        return int(self.model.config.codec_sample_rate)
