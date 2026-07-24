"""환경·시드 — 실행 경로 간 재현성.

RunPod 이미지는 `WANDB_PROJECT`·`HF_HOME`을 bake하고 `HF_TOKEN`·`WANDB_API_KEY`를 pod에 주입한다.
`setup_env`는 미설정 시에만 기본값을 채워 Colab 등 이미지 밖 실행에도 이식되도록 한다(멱등).

실행 환경마다 값이 달라 정적 기본값을 둘 수 없는 항목(`HF_HOME`·`WANDB_NOTEBOOK_NAME`)은 `overrides`로 받는다 —
`TrainingRunner`가 `TrainConfig.hf_cache`·`notebook_name`을 여기로 넘긴다. 캐시 경로의 SSOT는
`TrainConfig.hf_cache` 하나이며, 여기에 절대경로를 중복해 두지 않는다.
"""

import os
import random

import numpy as np
import torch

_ENV_DEFAULTS = {
    "WANDB_PROJECT": "patent_disc",
    "HF_HUB_DISABLE_XET": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
}


def setup_env(overrides: dict = None) -> None:
    """환경변수를 미설정 시에만 채운다(이미 bake된 값은 보존)."""
    for k, v in {**_ENV_DEFAULTS, **(overrides or {})}.items():
        os.environ.setdefault(k, v)


def set_seed(seed: int) -> None:
    """random·numpy·torch(CPU/CUDA) 시드 고정(단일 시드 방법론, ADR-0011)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
