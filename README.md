# Audio8-fast-streaming-ccu

[Tiếng Việt](README.vi.md)

A streaming inference engine for [Edge0/Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b),
built for many concurrent streams (CCU) and a short time to first audio (TTFA) on one GPU: Triton kernels, CUDA
graphs, continuous batching and a stateful streaming codec decoder. On one RTX 3090 it serves 112 concurrent streams
in real time (P95 TTFA 190 ms, P95 RTF 0.65) and starts a single stream in 44 ms. The checkpoint as shipped runs one
stream at RTF ~2.1.

**Status: experiment, stopped (2026-10-31).** The engine reproduces the checkpoint, but the checkpoint's Vietnamese
was not good enough for the product it was built for (see [Vietnamese quality](#vietnamese-quality)). The code is
published as a reference for serving this model family fast.

## Results

One RTX 3090 (24 GB, 350 W), 2026-10-31. Measured end to end through the streaming HTTP API (16 kHz PCM out); the load
generator is not in this repo. RTF = wall time / audio duration, TTFA included. Target: P95 TTFA < 800 ms, P95 RTF < 0.7.

| Load | TTFA p50 / p95 | RTF p95 |
|---|---|---|
| 1 request | 44 / 53 ms | 0.09 |
| burst of 64 | 243 / 402 ms | 0.40 |
| burst of 128 | 370 / 627 ms | 0.68 |
| 112 streams, sustained 60 s | 169 / 190 ms | **0.65** |
| 128 streams, sustained 60 s | 187 / 210 ms | 0.74 |

Under sustained load the card sits at its power cap. A tick at 192 rows is ~72 % codec decoder, ~20 % fast AR and
~11 % slow AR.

## What is in this repo

```
serve/fast_engine.py      the engine: FastEngine (slots, prefill, the tick, CUDA graphs), the streaming codec decoder
serve/fast_kernels.py     Triton kernels: RMSNorm, bias + RoPE + KV store, decode / prefill attention, GEMMs, codec convs
serve/tts_engine.py       model loading, voices (.pt), the engine behind the API
serve/create_speakers.py  encodes reference wavs into voices (codec codes + transcript)
serve/config.py           settings from .env
serve/engine_worker.py    the GPU worker process: scheduling, streaming chunks, audio post-processing processes
serve/production_api.py   the HTTP API entry point
serve/run_profile.sh      bare-metal start script
.env.example              every setting, documented
work/                     probes, parity tests, benchmarks and the Vietnamese evaluation used while building it
```

The engine core (`fast_engine.py`, `fast_kernels.py`, `tts_engine.py`, `create_speakers.py`, `config.py`) is
self-contained. `engine_worker.py` and `production_api.py` are the files changed to plug the engine into an existing
production TTS API (originally built for Qwen3-TTS). The rest of that API (text normalization, auth, UI, audio effects
and other modules they import) is not in this repo, so those two files are here for reading, not for running as is.

The `work/` scripts carry the paths of the machine they ran on (`/home/ubuntu/truc_ai/Audio8`); change `ROOT` before
running them.

## Requirements

* NVIDIA GPU with Triton support (tuned and measured on an RTX 3090).
* Tested with torch 2.6.0 (cu124), torchaudio 2.6.0, transformers 4.57.3, the Triton that ships with that torch,
  numpy, soundfile.
* The checkpoint: `hf download Edge0/Audio8-TTS-Realistic-0.6b --local-dir models/Audio8-TTS-Realistic-0.6b`
  (loaded with `trust_remote_code`).

## Using the engine

```python
import sys
sys.path.insert(0, "serve")
from create_speakers import encode_refs
from fast_engine import FastEngine
from tts_engine import load_model, voice_item

model, proc = load_model("models/Audio8-TTS-Realistic-0.6b")        # bf16, trust_remote_code
voice = voice_item(encode_refs(model, ["ref.wav"])[0], "The exact transcript of ref.wav.")
eng = FastEngine(model, proc.tokenizer, dict(do_sample=True, temperature=0.8, top_k=50, top_p=0.95,
                                              max_new_tokens=1024), max_slots=64)
eng.capture()                  # tunes GEMMs and codec convs, captures the CUDA graphs

reqs = [{"text": "Hello there! Today we learn new words.", "item": voice, "key": "my-voice"}]
eng.admit([0], reqs)           # prefill into slot 0; the voice's prompt prefix is KV-cached under its key
b = eng.bucket_for(len(reqs))
chunks, first = [], True
while True:
    eng.replay(b)              # one frame for every slot in the bucket: 2048 samples at 44.1 kHz
    if not first:              # the first tick's audio belongs to the reference, not the line
        chunks.append(eng.audio[0].float().cpu())
    first = False
    if int(eng.tok_in[0]) == eng.eos:
        break
```

`work/engine_probe.py` is a complete runnable version of this, with the parity checks against the HF checkpoint.
Tuning results are cached in the files named by `TTS_GEMM_TUNE_CACHE` and `TTS_VOC_TUNE_CACHE`, so only the first
start on a new GPU model pays for them (~5-10 min).

## How it works

The model is a DualAR (Fish Audio S2 style): a 24-layer slow AR (width 896, GQA 14/2) predicts one semantic token per
frame, a 4-layer fast AR predicts the frame's other 9 codebooks, and a causal codec decodes 2048 samples per frame
(44.1 kHz, ~21.5 frames per second) through a window-128 transformer and a DAC-style conv decoder. There are no
language ids: the text alone tells the model the language. A voice is the reference transcript in the system turn plus
the reference's codec codes.

* **Kernels.** Slow AR, fast AR and the codec transformer run on fused weights and Triton kernels: residual + RMSNorm,
  qkv bias + RoPE + KV store in one kernel, split-K decode attention, packed prefill attention, fused SiLU * mul. The
  checkpoint's interleaved RoPE runs as rotate_half on q / k rows permuted at load time. GEMMs go to cuBLAS or a Triton
  kernel per shape, whichever was faster when tuned.
* **Smaller head.** The semantic head reads only the 4096 semantic rows + EOS of the tied 155776-row embedding (37x
  fewer head FLOPs).
* **Fast AR.** Positions 0 and 1 in one 2-token pass: 9 passes per frame instead of 10.
* **Streaming codec decoder.** Each row keeps its conv tails and a 128-slot KV ring, so a frame costs one frame of work
  and the result equals the whole-utterance decode (48 dB SNR vs fp32; the HF bf16 decode reaches 42 dB). The decoder
  convs are channels-last fp16 implicit-GEMM Triton kernels with the tile shape tuned per layer, and the snake
  activation fused into the 1x1 convs. Every row's state lives in one byte arena, so resetting or loading rows is one
  kernel.
* **One CUDA graph per tick.** Per batch bucket, one graph holds the whole tick: the codec decodes the previous frame on
  a side stream while the fast AR and the slow step run, then sampling (the checkpoint's top-k / top-p order and its
  repetition-aware resampling). Prefill has CUDA graphs per token bucket (64 to 2048).
* **Continuous batching.** Requests join and leave slots every tick; a voice's prompt prefix is prefilled once and kept
  as KV; admissions are batched into one prefill. A code-switched line continues the previous segment in the assistant
  turn, so each segment prefills only its own tokens.
* **The worker.** Resampling, tempo, joins and PCM encoding run in separate post-processing processes, so the engine
  loop only feeds the GPU. That moved the bottleneck from the CPU (GPU 69 % busy) to the GPU.

## Correctness

* Prompt: token for token the same as the checkpoint's processor (`work/engine_probe.py`).
* Teacher-forced (`work/tf_probe.py`): the engine's argmax agrees with HF as often as HF's own forward agrees with HF's
  generate (~97 % on every codebook; the rest are bf16 near-ties).
* Codec decoder: rows are independent, state snapshots restore exactly, 48 dB SNR vs the fp32 decode
  (`work/voc_check.py`).

## Vietnamese quality

Vietnamese is why the project stopped, and the cause is the checkpoint, not the engine. 16 short Vietnamese lines,
scored with whisper-large-v3-turbo (CER per character, WER per syllable; `work/vi_eval.py`, `work/vi_sweep.py`,
`work/vi_peek.py`):

| | CER | WER |
|---|---|---|
| The reference recordings themselves (the ASR's floor) | 0.7 % | 2.5 % |
| This engine | 10.8 % | 31 % |
| The HF checkpoint's own `generate`, same voice and text | 11.0 % | 29 % |
| The model card's Vietnamese row (CV3) | | 28.1 % |

The errors are wrong tones and vowels (vòi → vối, nhất → nhận, tắm mát → tằm mạ), the same words in the engine and in
HF. Lower temperature (0.6, 0.4), top-k 10 and seven other voices did not fix it (best voice ~9 % CER / 24 % WER). The
model card's English row is 2.9 % WER. Fixing Vietnamese would take a Vietnamese fine-tune; the engine would serve the
fine-tuned weights unchanged.

One note for anyone scoring TTS with Whisper: the transformers ASR `pipeline` hallucinated "hãy subscribe cho kênh
ghiền mì gõ" on some of these wavs; calling the model's `generate` on 16 kHz features directly did not
(`work/vi_peek.py`).
