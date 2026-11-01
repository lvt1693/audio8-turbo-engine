# Audio8-fast-streaming-ccu

[English](README.md) (bản chính)

Engine suy luận streaming cho [Edge0/Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b),
tối ưu cho nhiều luồng đồng thời (CCU) và thời gian ra audio đầu tiên (TTFA) ngắn trên một GPU: kernel Triton, CUDA
graph, continuous batching và bộ giải mã codec streaming có giữ trạng thái. Trên một RTX 3090, engine phục vụ 112 luồng
đồng thời theo thời gian thực (P95 TTFA 190 ms, P95 RTF 0.65) và bắt đầu một luồng đơn trong 44 ms. Checkpoint gốc chỉ
chạy được một luồng với RTF ~2.1.

**Trạng thái: thử nghiệm, đã dừng (31/10/2026).** Engine tái tạo đúng checkpoint, nhưng tiếng Việt của checkpoint chưa
đủ tốt cho sản phẩm cần dùng (xem [Chất lượng tiếng Việt](#chất-lượng-tiếng-việt)). Code được công khai để tham khảo
cách phục vụ nhanh họ model này.

## Kết quả

Một RTX 3090 (24 GB, 350 W), 31/10/2026. Đo đầu cuối qua HTTP API streaming (đầu ra PCM 16 kHz); công cụ tạo tải không
có trong repo này. RTF = thời gian thực tế / độ dài audio, đã tính cả TTFA. Mục tiêu: P95 TTFA < 800 ms, P95 RTF < 0.7.

| Tải | TTFA p50 / p95 | RTF p95 |
|---|---|---|
| 1 request | 44 / 53 ms | 0.09 |
| 64 request cùng lúc | 243 / 402 ms | 0.40 |
| 128 request cùng lúc | 370 / 627 ms | 0.68 |
| 112 luồng liên tục 60 s | 169 / 190 ms | **0.65** |
| 128 luồng liên tục 60 s | 187 / 210 ms | 0.74 |

Khi tải liên tục, card chạm trần công suất. Một tick 192 hàng gồm ~72 % bộ giải mã codec, ~20 % fast AR và ~11 % slow
AR.

## Repo gồm những gì

```
serve/fast_engine.py      engine: FastEngine (slot, prefill, tick, CUDA graph), bộ giải mã codec streaming
serve/fast_kernels.py     kernel Triton: RMSNorm, bias + RoPE + ghi KV, attention decode / prefill, GEMM, conv của codec
serve/tts_engine.py       nạp model, giọng (.pt), lớp engine phía sau API
serve/create_speakers.py  encode wav tham chiếu thành giọng (mã codec + transcript)
serve/config.py           cấu hình từ .env
serve/engine_worker.py    process worker GPU: lập lịch, chia chunk streaming, các process hậu xử lý audio
serve/production_api.py   điểm vào HTTP API
serve/run_profile.sh      script khởi động trên máy thật
.env.example              toàn bộ cấu hình, có chú thích
work/                     script dò, kiểm tra khớp HF, benchmark và bài đo tiếng Việt dùng trong quá trình làm
```

Lõi engine (`fast_engine.py`, `fast_kernels.py`, `tts_engine.py`, `create_speakers.py`, `config.py`) chạy độc lập.
`engine_worker.py` và `production_api.py` là các file đã sửa để gắn engine vào một TTS API production có sẵn (vốn viết
cho Qwen3-TTS). Phần còn lại của API đó (chuẩn hoá văn bản, xác thực, giao diện, hiệu ứng audio và các module khác mà
hai file này import) không có trong repo, nên hai file này để đọc tham khảo, không chạy nguyên trạng được.

Các script trong `work/` mang đường dẫn của máy đã chạy chúng (`/home/ubuntu/truc_ai/Audio8`); sửa `ROOT` trước khi
chạy.

## Yêu cầu

* GPU NVIDIA hỗ trợ Triton (đã tune và đo trên RTX 3090).
* Đã kiểm thử với torch 2.6.0 (cu124), torchaudio 2.6.0, transformers 4.57.3, Triton đi kèm bản torch đó, numpy,
  soundfile.
* Checkpoint: `hf download Edge0/Audio8-TTS-Realistic-0.6b --local-dir models/Audio8-TTS-Realistic-0.6b`
  (nạp với `trust_remote_code`).

## Dùng engine

```python
import sys
sys.path.insert(0, "serve")
from create_speakers import encode_refs
from fast_engine import FastEngine
from tts_engine import load_model, voice_item

model, proc = load_model("models/Audio8-TTS-Realistic-0.6b")        # bf16, trust_remote_code
voice = voice_item(encode_refs(model, ["ref.wav"])[0], "Transcript khớp chính xác với ref.wav.")
eng = FastEngine(model, proc.tokenizer, dict(do_sample=True, temperature=0.8, top_k=50, top_p=0.95,
                                              max_new_tokens=1024), max_slots=64)
eng.capture()                  # tune GEMM và conv của codec, capture CUDA graph

reqs = [{"text": "Xin chào các bạn nhỏ!", "item": voice, "key": "my-voice"}]
eng.admit([0], reqs)           # prefill vào slot 0; phần prompt chung của giọng được giữ KV theo key
b = eng.bucket_for(len(reqs))
chunks, first = [], True
while True:
    eng.replay(b)              # mỗi slot trong bucket thêm một frame: 2048 mẫu ở 44.1 kHz
    if not first:              # audio của tick đầu là của tham chiếu, không phải của câu
        chunks.append(eng.audio[0].float().cpu())
    first = False
    if int(eng.tok_in[0]) == eng.eos:
        break
```

`work/engine_probe.py` là bản chạy được đầy đủ của đoạn trên, kèm các phép kiểm tra khớp với checkpoint HF. Kết quả
tune được lưu vào các file mà `TTS_GEMM_TUNE_CACHE` và `TTS_VOC_TUNE_CACHE` trỏ tới, nên chỉ lần khởi động đầu trên một
dòng GPU mới phải tốn thời gian tune (~5-10 phút).

## Cách hoạt động

Model là DualAR (kiểu Fish Audio S2): slow AR 24 lớp (rộng 896, GQA 14/2) dự đoán một token ngữ nghĩa mỗi frame, fast
AR 4 lớp dự đoán 9 codebook còn lại của frame, và codec nhân quả giải mã 2048 mẫu mỗi frame (44.1 kHz, ~21.5 frame mỗi
giây) qua một transformer cửa sổ 128 và bộ giải mã conv kiểu DAC. Model không có language id: chỉ văn bản cho model
biết đang đọc ngôn ngữ nào. Một giọng là transcript của tham chiếu trong system turn cộng với mã codec của tham chiếu.

* **Kernel.** Slow AR, fast AR và transformer của codec chạy trên trọng số đã gộp và kernel Triton: residual + RMSNorm,
  bias qkv + RoPE + ghi KV trong một kernel, attention decode split-K, attention prefill dạng packed, SiLU * mul gộp.
  RoPE xen kẽ của checkpoint chạy dưới dạng rotate_half trên các hàng q / k đã hoán vị lúc nạp. Mỗi kích thước GEMM đi
  cuBLAS hoặc kernel Triton, tuỳ cái nào nhanh hơn lúc tune.
* **Head nhỏ hơn.** Head ngữ nghĩa chỉ đọc 4096 hàng ngữ nghĩa + EOS của embedding dùng chung 155776 hàng (ít hơn 37
  lần FLOP).
* **Fast AR.** Vị trí 0 và 1 chạy chung một lượt 2 token: 9 lượt mỗi frame thay vì 10.
* **Bộ giải mã codec streaming.** Mỗi hàng giữ phần đuôi của các conv và một vòng KV 128 ô, nên mỗi frame chỉ tốn công
  của một frame, và kết quả bằng đúng việc giải mã cả câu (SNR 48 dB so với fp32; bản giải mã bf16 của HF đạt 42 dB).
  Các conv của bộ giải mã là kernel Triton implicit-GEMM fp16 channels-last, kích thước tile tune riêng cho từng lớp,
  activation snake gộp vào các conv 1x1. Trạng thái của mọi hàng nằm trong một vùng nhớ byte chung, nên reset hay nạp
  nhiều hàng chỉ cần một kernel.
* **Một CUDA graph cho mỗi tick.** Với mỗi bucket batch, một graph chứa trọn một tick: codec giải mã frame trước trên
  một stream phụ trong lúc fast AR và bước slow chạy, rồi đến sampling (đúng thứ tự top-k / top-p của checkpoint và
  cơ chế lấy mẫu lại khi lặp). Prefill có CUDA graph theo bucket số token (64 đến 2048).
* **Continuous batching.** Request vào và ra slot ở mỗi tick; phần prompt chung của một giọng chỉ prefill một lần rồi
  giữ dưới dạng KV; các request mới được gom vào một lần prefill. Câu trộn ngôn ngữ thì đoạn sau nối tiếp đoạn trước
  trong assistant turn, nên mỗi đoạn chỉ prefill token của chính nó.
* **Worker.** Resample, đổi tempo, ghép đoạn và mã hoá PCM chạy ở các process hậu xử lý riêng, để vòng lặp engine chỉ
  việc nuôi GPU. Nhờ vậy nút cổ chai chuyển từ CPU (GPU chỉ bận 69 %) sang GPU.

## Độ chính xác

* Prompt: khớp từng token với processor của checkpoint (`work/engine_probe.py`).
* Teacher-forced (`work/tf_probe.py`): argmax của engine khớp HF với tỉ lệ bằng tỉ lệ forward của chính HF khớp với
  `generate` của HF (~97 % trên mọi codebook; phần còn lại là các cặp gần bằng nhau do bf16).
* Bộ giải mã codec: các hàng độc lập, snapshot trạng thái khôi phục chính xác, SNR 48 dB so với giải mã fp32
  (`work/voc_check.py`).

## Chất lượng tiếng Việt

Tiếng Việt là lý do dự án dừng, và nguyên nhân nằm ở checkpoint, không phải engine. 16 câu tiếng Việt ngắn, chấm bằng
whisper-large-v3-turbo (CER theo ký tự, WER theo âm tiết; `work/vi_eval.py`, `work/vi_sweep.py`, `work/vi_peek.py`):

| | CER | WER |
|---|---|---|
| Chính các file tham chiếu (mức sàn của ASR) | 0.7 % | 2.5 % |
| Engine này | 10.8 % | 31 % |
| `generate` gốc của checkpoint HF, cùng giọng và câu | 11.0 % | 29 % |
| Dòng tiếng Việt trên model card (CV3) | | 28.1 % |

Lỗi là sai thanh điệu và nguyên âm (vòi → vối, nhất → nhận, tắm mát → tằm mạ), sai đúng những từ đó ở cả engine lẫn
HF. Giảm nhiệt độ (0.6, 0.4), top-k 10 và bảy giọng khác đều không sửa được (giọng tốt nhất ~9 % CER / 24 % WER). Dòng
tiếng Anh trên model card là 2.9 % WER. Muốn tiếng Việt tốt phải fine-tune bằng dữ liệu tiếng Việt; engine sẽ chạy
trọng số đã fine-tune mà không cần sửa gì.

Lưu ý khi chấm TTS bằng Whisper: `pipeline` ASR của transformers bịa ra câu "hãy subscribe cho kênh ghiền mì gõ" trên
một số file; gọi thẳng `generate` của model trên đặc trưng 16 kHz thì không bị (`work/vi_peek.py`).
