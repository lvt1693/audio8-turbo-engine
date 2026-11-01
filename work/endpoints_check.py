"""Quick check of the other Audio8 API endpoints (:29100): UI, admin, speakers, timed (word timings), sing, chant,
bot/tts, normalize. Run on .30: venv/bin/python work/endpoints_check.py"""
import time

import requests

B = "http://127.0.0.1:29100"


def show(name, r, t0):
    ct = r.headers.get("content-type", "")
    extra = ""
    if "json" in ct:
        try:
            j = r.json()
            extra = str({k: (v if not isinstance(v, (list, str)) or len(str(v)) < 80 else f"<{len(v)}>")
                         for k, v in (j.items() if isinstance(j, dict) else [("list", j)])})[:220]
        except Exception:
            extra = r.text[:200]
    print(f"{name:28s} {r.status_code} {len(r.content):8d} B {time.perf_counter() - t0:5.2f}s {ct[:24]:24s} {extra}", flush=True)


for name, fn in [
    ("GET /ui/", lambda: requests.get(B + "/ui/")),
    ("GET /admin/", lambda: requests.get(B + "/admin/")),
    ("GET /health", lambda: requests.get(B + "/health")),
    ("GET /v2/speakers", lambda: requests.get(B + "/v2/speakers", params={"group": "true"})),
    ("GET /v2/normalize", lambda: requests.get(B + "/v2/normalize", params={"text": "Hôm nay 31/10/2026, giá 25.000đ"})),
    ("POST /v2/multilang/timed", lambda: requests.post(B + "/v2/multilang/timed", json={"text": "Xin chào, hello my friend!"})),
    ("GET /v2/sing/tunes", lambda: requests.get(B + "/v2/sing/tunes")),
    ("POST /v2/sing", lambda: requests.post(B + "/v2/sing", json={"text": "Một con vịt xòe ra hai cái cánh", "tune": "twinkle"})),
    ("POST /v2/chant", lambda: requests.post(B + "/v2/chant", json={"text": "One two three four, let's go!"})),
    ("POST /bot/tts", lambda: requests.post(B + "/bot/tts", json={"text": "Chào cậu, hôm nay mình học tiếng Anh nhé. Let's start!"})),
    ("GET /v2/preview", lambda: requests.get(B + "/v2/preview", params={"speaker": "pika-2", "instruct": "happy"})),
]:
    t0 = time.perf_counter()
    try:
        show(name, fn(), t0)
    except Exception as e:
        print(f"{name:28s} EXC {e}")
print("EP_DONE")
