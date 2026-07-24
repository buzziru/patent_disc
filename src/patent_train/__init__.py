"""patent_train — A.X-Encoder-base 멀티라벨(188 Mno) 훈련 하니스.

노트북은 이 패키지 최상위에서 파사드·설정·손실 레지스트리·OOM 탐침을 임포트한다.

    from patent_train import TrainingRunner, TrainConfig, build_loss, LOSSES, probe_batches

관심사별 모듈: config(레시피 설정)·backbones(인코더+데이터셋 레지스트리)·data(로드·절단·collator)·
model(팩토리)·losses(레지스트리)·metrics(F1)·trainer(손실 주입)·runner(파사드)·probe(OOM)·env(시드·환경).
손실 축은 ADR-0009로 닫혔고 남은 레버는 모델·레시피 — 백본 교체(레지스트리 키)와 config 주입만으로 변형한다.
"""

from .config import TrainConfig
from .backbones import Backbone, BACKBONES, get_backbone
from .losses import LOSSES, build_loss, FocalLoss, BCELoss, ZlprLoss, AsymmetricLoss
from .metrics import make_compute_metrics
from .model import build_model
from .data import PatentData, MultiLabelCollator
from .trainer import LossTrainer
from .runner import TrainingRunner
from .probe import probe_batches
from .env import set_seed, setup_env

__all__ = [
    "TrainConfig",
    "TrainingRunner",
    "Backbone",
    "BACKBONES",
    "get_backbone",
    "LOSSES",
    "build_loss",
    "FocalLoss",
    "BCELoss",
    "ZlprLoss",
    "AsymmetricLoss",
    "make_compute_metrics",
    "build_model",
    "PatentData",
    "MultiLabelCollator",
    "LossTrainer",
    "probe_batches",
    "set_seed",
    "setup_env",
]
