"""멀티라벨 손실 — 레지스트리로 config에서 선택한다.

네 손실은 시그니처가 균일하다: `forward(logits, targets) -> scalar`(targets는 (N, C) 다중핫 float).
`LOSSES`에 등록하면 `build_loss(name, **params)`가 config의 `loss`·`loss_params`로 해석한다.
새 손실 추가 = 클래스 + 레지스트리 한 줄(trainer·runner 불변).

**리덕션 규약**: 요소별 손실(focal·bce·asl)은 (N, C) 전체 평균으로 환산한다 — `loss` 키만 바꿔도
같은 lr에서 기울기 크기가 비교 가능해야 손실 축이 통제된 대조가 된다. ZLPR은 문서 단위 pairwise
손실이라 요소 평균이 정의되지 않고 문서 평균이 native 환산이다.

구현은 검증된 노트북 기준이다(08_01 focal / 09_01 ZLPR / 09_02 ASL / 09_03 BCE).
손실 축은 ADR-0009로 닫혔고 focal(γ=2)이 운영값 — 네 손실은 재현·대조용으로 함께 보존한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss(Lin et al.). alpha는 전체 손실에 곱하는 스칼라 — 클래스 균형이 아니라 스케일."""

    def __init__(self, alpha: float = 0.25, gamma: int = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class BCELoss(nn.Module):
    """이진 교차 엔트로피(focal γ=0 등가) — 손실 축 진단 기준선."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


class ZlprLoss(nn.Module):
    """Zero-bounded Log-Sum-Exp Pairwise(Su et al.) — 정답/오답 로짓 간 pairwise 순위 손실."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets) -> torch.Tensor:
        targets = targets.float()
        logits = (1 - 2 * targets) * logits                # pos -> -s_i, neg -> s_j
        logits_pos = logits - (1 - targets) * 1e12         # {-s_i} 유지; neg 위치 -> -inf
        logits_neg = logits - targets * 1e12               # {s_j} 유지;  pos 위치 -> -inf
        zeros = torch.zeros_like(logits[..., :1])
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)
        pos = torch.logsumexp(logits_pos, dim=-1)          # log(1 + Σ_pos e^{-s_i})
        neg = torch.logsumexp(logits_neg, dim=-1)          # log(1 + Σ_neg e^{s_j})
        return (pos + neg).mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss(Ridnik et al., ICCV 2021) — 음성에 강한 focusing + 확률 이동 마진.

    원 논문은 라벨 축 합(문서당 합)으로 정의하나, 여기서는 모듈 리덕션 규약대로 요소 평균을 쓴다
    (합 대비 1/C 배 — C=188). 손실 값 자체가 아니라 스케일만 바뀌므로 최적점은 같고, 같은 lr에서
    focal·bce와 기울기 크기가 맞는다.
    """

    def __init__(self, gamma_pos: int = 0, gamma_neg: int = 4, margin: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin

    def forward(self, logits, targets) -> torch.Tensor:
        p = torch.sigmoid(logits)
        pm = (p - self.margin).clamp(min=0.0)              # p_m = max(p - m, 0)
        eps = 1e-8
        loss_pos = (1 - p) ** self.gamma_pos * torch.log(p.clamp(min=eps))
        loss_neg = pm ** self.gamma_neg * torch.log((1 - pm).clamp(min=eps))
        loss = targets * loss_pos + (1 - targets) * loss_neg
        return -loss.mean()


# 레지스트리 — 손실 추가 시 여기에 한 줄. build_loss·runner가 자동 반영한다.
LOSSES = {
    "focal": FocalLoss,
    "bce": BCELoss,
    "zlpr": ZlprLoss,
    "asl": AsymmetricLoss,
}


def build_loss(name: str, **params) -> nn.Module:
    """레지스트리 키와 하이퍼파라미터로 손실 모듈을 만든다."""
    if name not in LOSSES:
        raise KeyError(f"미등록 손실: {name!r} — 등록된 키: {sorted(LOSSES)}")
    return LOSSES[name](**params)
