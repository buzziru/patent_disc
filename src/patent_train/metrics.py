"""평가 지표 — sigmoid@τ 멀티라벨 F1.

`micro_f1`이 헤드라인이자 `metric_for_best_model` 기준(PROJECT.md 평가 프로토콜).
벤더 연속성용 앵커(top-1 weighted)도 병기한다. τ는 config에서 주입한다.

τ 규약(`DEFAULT_TAU`)·`sigmoid`·F1 3종(`f1_triple`)·`empty_rate`의 SSOT다 —
훈련 중 모델 선택에 쓰는 지표와 하류 오류 분석(`error_analysis`)의 지표가 같은 정의를 보도록
양쪽이 이 모듈을 공유한다.
"""

import numpy as np
from sklearn.metrics import f1_score

DEFAULT_TAU = 0.5      # 멀티라벨 임계값. sigmoid τ=0.5 ⟺ logit≥0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def f1_triple(Y: np.ndarray, pred: np.ndarray) -> dict:
    """멀티라벨 F1 3종 — 정답·예측은 같은 shape의 (N, C) 이진 행렬."""
    return {
        "micro_f1":  float(f1_score(Y, pred, average="micro",   zero_division=0)),
        "macro_f1":  float(f1_score(Y, pred, average="macro",   zero_division=0)),
        "sample_f1": float(f1_score(Y, pred, average="samples", zero_division=0)),
    }


def empty_rate(pred: np.ndarray) -> float:
    """예측 라벨이 하나도 없는 문서의 비율."""
    return float((pred.sum(1) == 0).mean())


def make_compute_metrics(tau: float = DEFAULT_TAU):
    """HF Trainer용 `compute_metrics(eval_pred)` 클로저를 τ에 묶어 만든다."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        Y = np.asarray(labels).astype(int)
        pred = (sigmoid(logits) >= tau).astype(int)
        return {
            **f1_triple(Y, pred),          # micro_f1이 헤드라인(모델 선택 기준)
            "empty_rate": empty_rate(pred),
            "anchor_weighted_f1": f1_score(Y.argmax(1), logits.argmax(1), average="weighted", zero_division=0),
        }

    return compute_metrics
