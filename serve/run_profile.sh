#!/bin/bash
# Start the TTS API on bare metal (Docker: docker-compose.yml at the repo root). Both read the one repo-root .env.
#   serve/run_profile.sh [profile | -] [port] [KEY=VALUE ...]
# Settings, the first that has a key wins:
#   1. KEY=VALUE arguments   this start only
#   2. the profile           profiles/<profile>.env, a preset (model, mode, slots) over the .env, this start only
#   3. the repo-root .env    every setting (copy .env.example; never rewritten by this script)
#   4. the built-in defaults (config.py)
# No profile (or -): the .env alone, what the server runs. 1 and 2 go into the server's environment, which config.py
# never overrides; if the server does not come up with them, it is started again from the .env alone. The port:
# the argument, else TTS_PORT, else 29100 (never 29000 / 29001: Queen TTS on the same machine). Logs to
# api_<port>.log at the repo root. The test line sends the server's own key, internal.key in the auth dir
# (TTS_AUTH_DIR, auth.py).
set -u
S=$(cd "$(dirname "$0")" && pwd)
R=$(dirname "$S")
P=${1:--}; [ "$P" = - ] && P=
[ $# -gt 0 ] && shift
OV=()                              # this start's KEY=VALUE: the profile's, then the arguments'
if [ -n "$P" ]; then
  F=$S/profiles/$P.env
  [ -f "$F" ] || { echo "no profile $F; profiles: $(ls $S/profiles | sed 's/\.env$//' | tr '\n' ' ')"; exit 1; }
  while IFS= read -r kv; do OV+=("$kv"); done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$F" | sed 's/[[:space:]]*#.*$//')
fi
PORT=
if [ $# -gt 0 ] && [[ $1 =~ ^[0-9]+$ ]]; then PORT=$1; shift; fi
OV+=("$@")
cd "$S"

val() {   # a setting as the server will see it: this start's (the last one), else the .env's (its last line)
  local kv v="" found=""
  for kv in "${OV[@]}"; do [ "${kv%%=*}" = "$1" ] && { v=${kv#*=}; found=1; }; done
  [ -n "$found" ] || v=$(grep -sE "^$1=" "$R/.env" | tail -1 | cut -d= -f2-)
  echo "$v" | cut -d'#' -f1 | tr -d ' '
}
PORT=${PORT:-$(val TTS_PORT)}; PORT=${PORT:-29100}
# 29000 / 29001 are Queen TTS (Qwen3-TTS) on the same machine: its process matches the pkill pattern below. Never.
case "$PORT" in 29000|29001) echo "port $PORT belongs to Queen TTS: refusing (Audio8 runs on 29100)"; exit 1;; esac
NAME=${P:-.env}; [ ${#OV[@]} -gt 0 ] && [ -z "$P" ] && NAME=".env + $*"
LOG=$R/api_$PORT.log
TRITON=$R/data/cache/triton; mkdir -p "$TRITON"
start() {
  local spk gpus auth n
  spk=$(val TTS_SPEAKERS_DIR); gpus=$(val TTS_GPUS); auth=$(val TTS_AUTH_DIR)
  pkill -f "[p]roduction_api.py --speakers-dir [^ ]* --port $PORT"
  for i in $(seq 1 30); do
    pgrep -f "[p]roduction_api.py --speakers-dir [^ ]* --port $PORT" > /dev/null || break
    sleep 1
  done
  sleep 3
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${gpus:-0} TRITON_CACHE_DIR=$TRITON \
    setsid env "${OV[@]}" bash -c "exec ../venv/bin/python -u production_api.py --speakers-dir ${spk:-../data/speakers/speakers_a8} --port $PORT > $LOG 2>&1" \
    < /dev/null > /dev/null 2>&1 &
  for i in $(seq 1 120); do
    sleep 5
    if curl -s -m 3 http://127.0.0.1:$PORT/health | grep -q '"ok"'; then
      # up is not enough: one real line must come back as audio (an engine that captured its graphs but has no
      # memory left for a prefill answers /health and fails every request)
      # every endpoint needs an API key (auth.py): the server's own key, internal.key in the auth dir
      n=$(curl -s -m 120 -G http://127.0.0.1:$PORT/v2/speech/english --data-urlencode "text=Hello, this is a quick check." \
          -H "X-API-Key: $(cat "${auth:-../data/auth}/internal.key" 2>/dev/null)" \
          --data-urlencode "instruct=none" -o /dev/null -w "%{http_code} %{size_download}")
      [ "${n%% *}" = 200 ] && [ "${n##* }" -gt 10000 ] && return 0
      echo "[$NAME] /health ok but a test line failed ($n)"
      return 1
    fi
    pgrep -f "[p]roduction_api.py --speakers-dir [^ ]* --port $PORT" > /dev/null || return 1
  done
  return 1
}

if start; then
  echo "[$NAME] up on :$PORT -- $(grep -a 'ready (slots' $LOG | tail -1 | sed 's/.*ready //')"
else
  echo "[$NAME] did not come up on :$PORT:"; grep -aE 'Traceback|Error|out of memory' $LOG | head -5
  cp $LOG "$LOG.failed_${P:-env}"           # kept: the restart below overwrites the log
  if [ ${#OV[@]} -gt 0 ]; then
    OV=()
    echo "starting again from the .env alone"
    start && echo "[.env] up on :$PORT -- $(grep -a 'ready (slots' $LOG | tail -1 | sed 's/.*ready //')"
  fi
  exit 1
fi
