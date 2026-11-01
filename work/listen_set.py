"""A listening set from the Audio8 API (:29100): dual endpoint (north / south, styles, slow English, code-switching)
and multilang. Writes work/listen/NN_name.wav + index.txt. Run on .30: venv/bin/python work/listen_set.py [tag]"""
import os
import sys
import time

import requests

BASE = "http://127.0.0.1:29100"
TAG = sys.argv[1] if len(sys.argv) > 1 else "a8"
OUT = f"/home/ubuntu/truc_ai/Audio8/work/listen_{TAG}"
os.makedirs(OUT, exist_ok=True)

DUAL = [
    ("vi_north_greeting", {"text": "Xin chào các bạn nhỏ! Hôm nay chúng mình sẽ cùng nhau học về các loài động vật sống trong rừng nhé."}),
    ("vi_north_elephant", {"text": "Con voi là loài vật trên cạn to lớn nhất. Nó có cái vòi rất dài để uống nước và tắm mát."}),
    ("vi_north_numbers", {"text": "Bây giờ là 7 giờ 30 phút sáng, ngày 31 tháng 10 năm 2026. Nhiệt độ ngoài trời khoảng 25 độ C."}),
    ("vi_north_praise", {"text": "Cậu giỏi quá! Thêm một lần nữa nhé, mình cùng đếm từ một đến mười nào."}),
    ("en_animals", {"text": "Hello there! Today we are going to learn some new words about animals. Are you ready?"}),
    ("en_apple", {"text": "Look at this! It's a big red apple. Can you say apple with me?"}),
    ("mix_adventure", {"text": "Hôm nay chúng ta học từ mới, the word is adventure, nghĩa là cuộc phiêu lưu, let's say it together!"}),
    ("mix_tags_park", {"text": "[en]Good morning! [vi]Hôm nay trời đẹp quá, [en]let's go to the park [vi]và chơi thả diều nhé."}),
    ("vi_south_market", {"text": "Bữa nay mình đi chợ mua trái cây nha, có xoài, có mận, ngon lắm đó.", "lora": "south"}),
    ("vi_style_sad", {"text": "Tớ buồn quá, hôm nay con mèo của tớ bị lạc mất rồi.", "instruct": "sad"}),
    ("en_style_excited", {"text": "Wow, we did it! That was amazing, high five!", "instruct": "excited"}),
    ("en_slow", {"text": "Please repeat after me: elephant, giraffe, and crocodile.", "speed": "slow"}),
    ("vi_plain_none", {"text": "Mình là Pika, rất vui được làm quen với cậu. Cậu tên là gì vậy?", "instruct": "none"}),
]
MULTI = [
    ("ml_fr_ja_alice", "Bonjour! Je m'appelle Alice. 今日はいい天気ですね。", "alice"),
    ("ml_es_de_adam", "¡Hola amigos! Heute lernen wir zusammen neue Wörter.", "adam"),
    ("ml_vi_en_ja_alice", "Xin chào, mình là Alice. Nice to meet you! よろしくお願いします。", "alice"),
]

idx = []
for i, (name, params) in enumerate(DUAL):
    t0 = time.perf_counter()
    r = requests.post(BASE + "/v2/speech/dual_vietnamese_english", json=dict(params, stream=False), timeout=300)
    dt = time.perf_counter() - t0
    f = f"{OUT}/{i:02d}_{name}.wav"
    open(f, "wb").write(r.content)
    idx.append(f"{i:02d}_{name}.wav\t{r.status_code}\t{len(r.content)} B\t{dt:.2f}s\t{r.headers.get('X-Speaker', '')}\t{params}")
    print(idx[-1], flush=True)
for j, (name, text, spk) in enumerate(MULTI):
    i = len(DUAL) + j
    t0 = time.perf_counter()
    r = requests.get(BASE + "/v2/multilang", params={"text": text, "speaker": spk, "stream": "false"}, timeout=300)
    dt = time.perf_counter() - t0
    f = f"{OUT}/{i:02d}_{name}.wav"
    open(f, "wb").write(r.content)
    idx.append(f"{i:02d}_{name}.wav\t{r.status_code}\t{len(r.content)} B\t{dt:.2f}s\t{r.headers.get('X-Speaker', '')}\t{text}")
    print(idx[-1], flush=True)
open(f"{OUT}/index.txt", "w", encoding="utf-8").write("\n".join(idx) + "\n")
print("LISTEN_DONE", OUT)
