#!/usr/bin/env bash
# Audio8 venv on .30: the same package set as Queen TTS (torch 2.6.0 cu124, transformers 4.57.3) so the
# Triton kernels behave as they do there.
set -euo pipefail
cd /home/ubuntu/truc_ai/Audio8
V=venv/bin
$V/pip install -q --upgrade pip wheel setuptools
$V/pip install -q torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
$V/pip install -q -r /home/ubuntu/truc_ai/Qwen3-TTS/docker/requirements.txt
$V/pip install -q --no-deps praat-parselmouth==0.4.7
$V/python -c "import torch, transformers, triton; print('OK', torch.__version__, transformers.__version__, triton.__version__)"
echo VENV_DONE
