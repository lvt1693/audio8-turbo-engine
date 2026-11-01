#!/usr/bin/env bash
# Pack the local Audio8 tree (code only) and unpack it into /home/ubuntu/truc_ai/Audio8 on .30 (LF line endings).
# Usage (Git Bash, from D:/TEST_PROJECTS/Audio8): bash work/push.sh
set -euo pipefail
cd "$(dirname "$0")/.."
tar --exclude='./work/hf_model' --exclude='./work/*.tar' --exclude='./work/out_*' --exclude='__pycache__' \
    -cf work/push.tar serve docker Dockerfile docker-compose.yml .dockerignore .gitattributes .gitignore .env.example \
    models/README.md data/README.md LICENSE README.md work/*.py work/*.sh
cd work
MSYS_NO_PATHCONV=1 python rx.py --put D:/TEST_PROJECTS/Audio8/work/push.tar /home/ubuntu/truc_ai/Audio8/push.tar
MSYS_NO_PATHCONV=1 python rx.py "cd /home/ubuntu/truc_ai/Audio8 && tar -xf push.tar && rm push.tar && find serve work docker -name '*.py' -o -name '*.sh' -o -name '*.env' | xargs sed -i 's/\r$//' && chmod +x serve/*.sh 2>/dev/null; echo PUSHED"
rm -f push.tar
