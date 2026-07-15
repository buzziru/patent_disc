# patent_disc RunPod 훈련 이미지 (P1: A.X-Encoder / KoBERT, 188-way multi-label)
#
# 설계 원칙:
#   - 환경 SSOT = uv.lock. 이미지 안에서 `uv sync --frozen`으로 그대로 재현
#     → 로컬(Windows CPU)·Colab·RunPod·Lightning 4곳 버전 일치.
#   - torch 2.11.0+cu128 / flash-attn cu12torch2.11 (프리빌트 휠, gpu 그룹).
#     프리빌트라 nvcc 불필요 → devel CUDA 툴킷을 이미지에 담지 않는다.
#   - 데이터는 이미지에 넣지 않는다(HF Hub streaming).
#
# 학습용으로 runpod/pytorch 베이스를 쓴다 — SSH/Jupyter/PUBLIC_KEY 처리가
# 엔트리포인트에 이미 들어 있어 "SSH 설정"이라는 변수를 제거한다. 베이스가 제공하는
# 시스템 torch는 쓰지 않고, 훈련은 아래 /opt/venv(uv.lock 재현본)로 돌린다.
#
# 배치/Lightning 공용 슬림 이미지가 필요하면 베이스만 아래로 교체(엔트리포인트 없음 → sshd 직접 구성 또는 배치 실행):
#   FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# uv 바이너리(정적) 반입
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 훈련 환경은 /opt/venv에 고정. HF/wandb 캐시는 볼륨(/workspace)으로 유도해
# Stop 후에도 모델 재다운로드가 없게 한다(컨테이너 디스크는 Stop 시 소실).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/workspace/.hf_cache \
    WANDB_PROJECT=patent_disc \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 코드는 이미지 밖
#    --group gpu: flash-attn 프리빌트 휠(linux 마커). --no-dev: nbdime 등 dev 제외.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group gpu \
&& /opt/venv/bin/python -m ipykernel install --name patent_disc --display-name "patent_disc (/opt/venv)"


# 엔트리포인트/CMD를 덮지 않는다 — runpod 베이스의 기본 CMD가 SSH/Jupyter를 띄운다.
# 접속 후 훈련은 venv 인터프리터로 실행:
#   /opt/venv/bin/python -c "import torch, flash_attn; print(torch.__version__)"
#   또는  cd /app && uv run --no-sync jupyter ...  (uv.lock 재해석 없이 /opt/venv 사용)
