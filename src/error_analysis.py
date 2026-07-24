"""로짓·라벨 기반 오류 분석 하니스.

특허 다중레이블 분류(188 Mno / 17 Lno)의 test 로짓·정답을 받아 오류를 분해한다.
06_01·09_04 등 여러 노트북이 공유하며, 계산만 담당한다(표 출력·포맷은 노트북 셀에 둔다).

구성
- `LabelSpace`      : 188 Mno 열 ↔ 17 Lno 축 사영.
- `EvalAxes`        : 모델 독립 공유 축(정답 `Y`·길이 bin·카디널리티 `k_gold`·`tau`).
- `ModelResult`     : 모델별 파생(확률·top1·오류 마스크·혼동행렬)을 1회 계산해 보유.
- 기법(technique)   : 균일 시그니처 `fn(axes, m) -> dict`. `TECHNIQUES`에 등록하면
                      `analyze_model`이 순회해 결과를 합친다. 기법 추가 = 함수 + 레지스트리 한 줄.
- 교차 모델         : `compare`·`hard_core_mask`·`hierarchy_verdict`.
- `ErrorAnalysis`   : 위 요소를 묶은 파사드. 노트북은 이 클래스 하나만 import해
                      `from_labels → set_data → add` 로 로드·분석하고 메서드로 교차 모델을 낸다.

τ 규약·`sigmoid`·F1 3종·`empty_rate`는 훈련 하니스의 `patent_train.metrics`에서 가져온다 —
훈련 중 모델 선택 지표와 여기의 분석 지표가 같은 정의를 보게 하려는 것이다.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from patent_train.metrics import DEFAULT_TAU, empty_rate, f1_triple, sigmoid

DEFAULT_BINS = ["<=512", "512-1024", "1024-2048", ">2048"]


# ── 로짓·라벨 로딩 ─────────────────────────────────────────────────────────

def load_logits(cache_dir: Path, tag: str, split: str) -> np.ndarray:
    """`logits_{tag}_{split}.npy` 로짓 캐시를 읽는다(없으면 FileNotFoundError)."""
    fp = Path(cache_dir) / f"logits_{tag}_{split}.npy"
    if not fp.exists():
        raise FileNotFoundError(f"로짓 캐시 없음: {fp} — 09_00(손실)/06_00(focal)으로 덤프 후 다운로드할 것")
    print(f"[load] {fp.name}")
    return np.load(fp)


def build_gold(label_ids, n: int, num_labels: int) -> np.ndarray:
    """문서별 정답 id 리스트 → (N, C) 다중핫 bool."""
    Y = np.zeros((n, num_labels), dtype=bool)
    for i, ids in enumerate(label_ids):
        Y[i, ids] = True
    return Y


# ── 계층 사영 ─────────────────────────────────────────────────────────────

class LabelSpace:
    """
    188 Mno 열 ↔ 17 Lno 축
    188-dim 이산 선택(top1 or pred) → [M2L 사영] → 17-dim
    """

    def __init__(self, id2mno: dict, mno2lno: dict, num_labels: int = 188):
        self.C = num_labels
        self.mno_of_col = [id2mno[str(c)] for c in range(self.C)]     # JSON 키는 문자열. (C,)
        self.lno_of_col = [mno2lno[m] for m in self.mno_of_col]       # (C, )
        self.lnos = sorted(set(mno2lno.values()))                     # (17,)
        self.lno_index = {l: i for i, l in enumerate(self.lnos)}      # len = C
        self.L = len(self.lnos)                                       # (17,)
        self.lno_idx = np.array([self.lno_index[l] for l in self.lno_of_col])     # (C,) 열→Lno
        self.M2L = np.zeros((self.C, self.L), dtype=int)              # (C, L) (188, 17)
        self.M2L[np.arange(self.C), self.lno_idx] = 1
        assert self.M2L.sum() == self.C                               # M2L은 행별로 1이 한 개. sum=188

    def to_lno(self, X: np.ndarray) -> np.ndarray:
        """(N, C) Mno 다중핫 → (N, L) Lno 다중핫."""
        return (X.astype(int) @ self.M2L) > 0


# ── 공유 축 · 모델별 파생 ──────────────────────────────────────────────────

@dataclass
class EvalAxes:
    """모델 독립 공유 축. 모든 기법이 정답·길이·카디널리티·tau를 여기서 읽는다."""
    ls: LabelSpace
    Y: np.ndarray
    length_bin: np.ndarray
    bins: list = None
    tau: float = DEFAULT_TAU
    k_gold: np.ndarray = field(init=False)          # 문서별 정답 라벨 개수(행방향 합)

    def __post_init__(self):
        if self.bins is None:
            self.bins = DEFAULT_BINS
        self.k_gold = self.Y.sum(1)

    @classmethod
    def from_dataset(cls, ds, ls: LabelSpace, num_labels: int, tau: float = DEFAULT_TAU, bins=None):
        Y = build_gold(ds["label_ids"], len(ds), num_labels)
        return cls(ls=ls, Y=Y, length_bin=np.array(ds["length_bin"]), bins=bins, tau=tau)


@dataclass
class ModelResult:
    """모델별 파생을 1회 계산해 보유 — 기법들이 재계산 없이 공유한다.

    top1/err/sibling/cross는 앵커(top-1) 오류 분해 마스크,
    confusion은 앵커 오류 문서의 (정답 Lno × 예측 Lno) 17×17 혼동 행렬이다.
    """
    axes: EvalAxes
    tag: str
    logits: np.ndarray
    P: np.ndarray = field(init=False)
    top1: np.ndarray = field(init=False)
    err: np.ndarray = field(init=False)
    sibling: np.ndarray = field(init=False)
    cross: np.ndarray = field(init=False)
    confusion: np.ndarray = field(init=False)

    @classmethod
    def build(cls, axes: EvalAxes, tag: str, logits: np.ndarray) -> "ModelResult":
        m = cls.__new__(cls)
        m.axes, m.tag, m.logits = axes, tag, logits
        m.P = sigmoid(logits)

        ls, Y = axes.ls, axes.Y
        n = len(Y)
        top1 = logits.argmax(1)                       # (N,) 가장 확신하는 Mno
        hit = Y[np.arange(n), top1]                   # top1의 멀티핫
        err = ~hit
        gold_col = ls.to_lno(Y)[:, ls.lno_idx]        # (N, C) 열 c가 자신의 Lno 정답 집합에 속하는가
        sibling = err & gold_col[np.arange(n), top1]  # 대분류는 맞았는데 형제 중분류를 헷갈린 오류
        cross = err & ~gold_col[np.arange(n), top1]   # top1의 부모 Lno조차 정답 Lno 밖 — 대분류를 벗어난 오류
        assert int(sibling.sum() + cross.sum()) == int(err.sum())
        assert abs(float(err.mean()) - (1.0 - float(hit.mean()))) < 1e-12

        m.top1, m.err, m.sibling, m.cross = top1, err, sibling, cross
        m.confusion = _lno_confusion(err, top1, Y, ls)
        return m


def _lno_confusion(err: np.ndarray, top1: np.ndarray, Y: np.ndarray, ls: LabelSpace) -> np.ndarray:
    """
    앵커 오류 문서의 (정답 Lno × 예측 Lno) 혼동 행렬.
    정답 Lno가 복수인 문서는 각 정답 Lno에 1씩 계상하므로 행 합이 오류 수보다 클 수 있다.
    """
    M = np.zeros((ls.L, ls.L), dtype=int)   # 17 * 17 혼동 행렬
    gold_L = ls.to_lno(Y)
    for i in np.where(err)[0]:              # 틀린 문서(err)만 순회
        pl = ls.lno_idx[top1[i]]           # 문서 i의 top1 예측 Mno가 속한 Lno 인덱스
        for gl in np.where(gold_L[i])[0]:  # 정답 Mno의 모든 Lno를 순회
            M[gl, pl] += 1
    return M


# ── 기법(technique): fn(axes, m) -> dict ──────────────────────────────────

def anchor_stats(axes: EvalAxes, m: ModelResult) -> dict:
    """앵커(top-1) 오류 분해 — sibling(대분류 적중·중분류 실패) vs cross-Lno(대분류 이탈)."""
    ls, Y = axes.ls, axes.Y
    n = len(Y)
    err, sibling = m.err, m.sibling
    hit = Y[np.arange(n), m.top1]
    gold_col = ls.to_lno(Y)[:, ls.lno_idx]        # (N, C)

    # 귀무 기준 — 오답을 비정답 클래스에서 균등 추출할 때 sibling이 될 확률.
    # 대분류당 Mno가 2~20개로 고르지 않아 문서마다 다르므로 오류 문서 평균으로 잡는다.
    chance = float(((gold_col & ~Y).sum(1)[err] / (~Y).sum(1)[err]).mean())
    ratio = float(sibling.sum() / err.sum())      # 전체 오류 중 sibling 비율

    return {
        "p@1": round(float(hit.mean()), 4),
        "n_error": int(err.sum()),
        "error_rate": round(float(err.mean()), 4),
        "sibling": int(sibling.sum()),
        "cross_lno": int(m.cross.sum()),
        "sibling_ratio": round(ratio, 4),
        "chance_sibling_ratio": round(chance, 4),
        "sibling_enrichment": round(ratio / chance, 2),
    }


def multilabel_stats(axes: EvalAxes, m: ModelResult) -> dict:
    """멀티라벨(τ) 오류 분해 — FP·FN과 그 sibling 비율(대분류는 일치).

    FP: 예측한 오답 라벨의 Lno가 정답 Lno 집합에 있는가
    FN: 놓친 정답 라벨의 Lno가 예측 Lno 집합에 있는가(= 대분류는 맞췄으나 중분류를 놓침)
    """
    ls, Y, tau = axes.ls, axes.Y, axes.tau
    pred = m.P >= tau                            # (N, C) 문서 i에서 라벨 c를 예측했는가
    FP, FN = pred & ~Y, Y & ~pred
    gold_col = ls.to_lno(Y)[:, ls.lno_idx]       # (N, C)
    pred_col = ls.to_lno(pred)[:, ls.lno_idx]    # (N, C)
    fp_sib, fn_sib = FP & gold_col, FN & pred_col
    return {
        "tau": tau,
        "empty_rate": round(empty_rate(pred), 6),
        "fp": int(FP.sum()),
        "fp_sibling": int(fp_sib.sum()),
        "fp_sibling_ratio": round(float(fp_sib.sum() / FP.sum()), 4),
        "fn": int(FN.sum()),
        "fn_sibling": int(fn_sib.sum()),
        "fn_sibling_ratio": round(float(fn_sib.sum() / FN.sum()), 4),
    }


def lno_stats(axes: EvalAxes, m: ModelResult) -> dict:
    """Mno 예측을 M2L로 사영해 유도한 대분류 성능(별도 Lno 헤드 없음) + 계층 확장 이득 추정."""
    ls, Y, tau = axes.ls, axes.Y, axes.tau
    logits, P = m.logits, m.P
    n = len(Y)
    pred = P >= tau
    YL, predL = ls.to_lno(Y), ls.to_lno(pred)
    top1 = logits.argmax(1)
    lno_p1 = float(YL[np.arange(n), ls.lno_idx[top1]].mean())    # Lno 단계 정확도
    flat_p1 = float(Y[np.arange(n), top1].mean())

    # 오라클-Lno: 정답 Lno에 속한 열로만 제한한 뒤 argmax → 완벽한 Lno 단계를 가정한 상한
    restricted = np.where(YL[:, ls.lno_idx], logits, -np.inf)
    oracle_p1 = float(Y[np.arange(n), restricted.argmax(1)].mean())

    # 2단계 추정 = Lno 단계 정확도 × 조건부 정확도. Lno가 틀리면 회복 불가라는 전제
    two_stage = lno_p1 * oracle_p1

    return {
        **{k: round(v, 4) for k, v in f1_triple(YL, predL).items()},   # Lno 축 F1 3종
        "p@1": round(lno_p1, 4),
        "oracle_lno_p@1": round(oracle_p1, 4),
        "two_stage_p@1_est": round(two_stage, 4),
        "delta_vs_flat": round(two_stage - flat_p1, 4),
    }


def lno_confusion_record(axes: EvalAxes, m: ModelResult) -> dict:
    """저장용 혼동 행렬 레코드(라벨 축 + 17×17 행렬)."""
    return {"lnos": axes.ls.lnos, "matrix": m.confusion.tolist()}


def r_precision(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """문서별 상위 |정답| 예측의 정밀도."""
    order = np.argsort(-P, axis=1)
    k = Y.sum(1)
    hits = np.array([Y[i, order[i, :k[i]]].sum() for i in range(len(Y))], dtype=float)
    return hits / np.maximum(k, 1)


def count_bin_stats(axes: EvalAxes, m: ModelResult) -> dict:
    """라벨 개수 bin — 단일(k=1) vs 다라벨(k≥2)."""
    P, Y, tau = m.P, axes.Y, axes.tau
    pred = P >= tau
    k = Y.sum(1)
    rp = r_precision(P, Y)
    inter = (pred & Y).sum(1)
    sample_f1 = 2 * inter / np.maximum(pred.sum(1) + k, 1)
    out = {}
    for name, sel in [("k=1", k == 1), ("k>=2", k >= 2)]:
        fp, fn = int((pred & ~Y)[sel].sum()), int((Y & ~pred)[sel].sum())
        out[name] = {
            "n": int(sel.sum()),
            "micro_f1": round(float(f1_score(Y[sel], pred[sel], average="micro", zero_division=0)), 4),
            "sample_f1": round(float(sample_f1[sel].mean()), 4),
            "r_precision": round(float(rp[sel].mean()), 4),
            "fp": fp,
            "fn": fn,
            "fp_fn_ratio": round(fp / max(fn, 1), 4),
        }
    return out


def cardinality_stats(axes: EvalAxes, m: ModelResult) -> dict:
    """k≥2 결손을 주 지표(micro)로 환산 — 오라클 카디널리티 상한·과소예측 진단."""
    logits, P, Y, tau = m.logits, m.P, axes.Y, axes.tau
    k = Y.sum(1)
    pred = P >= tau
    m2 = k >= 2
    cf = pred.copy()                              # k≥2 문서만 상위 k개로 교체(랭킹 불변)
    for i in np.where(m2)[0]:
        row = np.zeros(pred.shape[1], dtype=bool)
        row[np.argpartition(-logits[i], k[i])[:k[i]]] = True
        cf[i] = row
    micro = float(f1_score(Y, pred, average="micro", zero_division=0))
    micro_oracle = float(f1_score(Y, cf, average="micro", zero_division=0))
    k_pred = pred.sum(1)
    return {
        "pos_instances_k1": int(Y[k == 1].sum()),
        "pos_instances_k>=2": int(Y[m2].sum()),
        "pos_share_k>=2": round(float(Y[m2].sum() / Y.sum()), 4),
        "micro": round(micro, 4),
        "micro_oracle_k_on_multi": round(micro_oracle, 4),
        "oracle_k_gain_pt": round(100 * (micro_oracle - micro), 3),
        "under_predict_rate_k>=2": round(float((k_pred[m2] < k[m2]).mean()), 4),
        "mean_k_pred_k>=2": round(float(k_pred[m2].mean()), 3),
        "mean_k_gold_k>=2": round(float(k[m2].mean()), 3),
        "mean_k_pred_k1": round(float(k_pred[k == 1].mean()), 3),
    }


def pair_symmetry(M: np.ndarray, ls: LabelSpace):
    """17×17 혼동 행렬 → 무향 쌍 질량·대칭도, off-diagonal 합계."""
    pairs = []
    for a in range(ls.L):
        for b in range(a + 1, ls.L):
            ab, ba = int(M[a, b]), int(M[b, a])
            tot = ab + ba
            if tot:
                pairs.append({
                    "pair": f"{ls.lnos[a]}<->{ls.lnos[b]}",
                    "ab": ab, "ba": ba, "total": tot,
                    "symmetry": round(min(ab, ba) / max(ab, ba), 3),   # 1.0 = 완전 대칭
                })
    pairs.sort(key=lambda d: -d["total"])
    off_total = int(M.sum() - np.trace(M))
    return pairs, off_total


def pair_oracle_gain(logits, top1, err, pairs, topn, Y, ls: LabelSpace) -> dict:
    """상위 topn 무향 쌍이 걸린 앵커 오류를, 정답 Lno 열로 제한해 재선택한 P@1 이득(상한, pt)."""
    sel = set()
    for p in pairs[:topn]:
        a, b = p["pair"].split("<->")
        ia, ib = ls.lno_index[a], ls.lno_index[b]
        sel |= {(ia, ib), (ib, ia)}
    YL = ls.to_lno(Y)
    fixed = 0
    for i in np.where(err)[0]:
        pl = ls.lno_idx[top1[i]]                 # 예측 top1의 부모 Lno
        gls = np.where(YL[i])[0]                 # 정답 Lno 집합
        if any((gl, pl) in sel for gl in gls):
            zi = np.where(YL[i][ls.lno_idx], logits[i], -np.inf)   # 정답 Lno 열로 제한
            if Y[i, zi.argmax()]:
                fixed += 1
    return {"n_fixed": int(fixed), "p@1_gain_pt": round(100 * fixed / len(Y), 3)}


def pair_stats(axes: EvalAxes, m: ModelResult) -> dict:
    """cross-Lno 무향 쌍 질량·대칭도 + 상위 쌍 국소 처리 상한(P@1 이득)."""
    ls, Y = axes.ls, axes.Y
    pairs, off_total = pair_symmetry(m.confusion, ls)
    return {
        "off_diagonal_total": off_total,
        "top5_pairs": pairs[:5],
        "top5_share": round(sum(p["total"] for p in pairs[:5]) / max(off_total, 1), 4),
        "top10_share": round(sum(p["total"] for p in pairs[:10]) / max(off_total, 1), 4),
        "oracle_gain": {f"top{n}": pair_oracle_gain(m.logits, m.top1, m.err, pairs, n, Y, ls)
                        for n in (1, 5, 10)},
    }


def per_class_f1(axes: EvalAxes, m: ModelResult) -> dict:
    """중분류(Mno)별 F1·support — 두 모델을 클래스 단위로 paired 대조할 때 쓴다."""
    ls, Y, tau = axes.ls, axes.Y, axes.tau
    pred = m.P >= tau
    f1 = f1_score(Y, pred, average=None, zero_division=0)
    support = Y.sum(0)
    return {ls.mno_of_col[c]: {"f1": round(float(f1[c]), 4), "support": int(support[c])}
            for c in range(ls.C)}


def length_bin_stats(axes: EvalAxes, m: ModelResult) -> dict:
    """길이 bin × 오류 유형 — cross-Lno(표현력)인지 FN(임계값)인지 가른다."""
    P, Y, tau = m.P, axes.Y, axes.tau
    length_bin, bins = axes.length_bin, axes.bins
    err, sibling, cross = m.err, m.sibling, m.cross
    pred = P >= tau
    FP, FN = pred & ~Y, Y & ~pred
    out = {}
    for b in bins:
        sel = length_bin == b
        n_err = int(err[sel].sum())
        out[b] = {
            "n": int(sel.sum()),
            "anchor_error_rate": round(float(err[sel].mean()), 4),
            "sibling": int(sibling[sel].sum()),
            "cross_lno": int(cross[sel].sum()),
            "sibling_ratio": round(float(sibling[sel].sum() / max(n_err, 1)), 4),
            "fp_per_doc": round(float(FP[sel].sum() / sel.sum()), 4),
            "fn_per_doc": round(float(FN[sel].sum() / sel.sum()), 4),
        }
    return out


# 레지스트리 — 기법 추가 시 여기에 한 줄. analyze_model·저장 루프가 자동 반영한다.
TECHNIQUES = {
    "anchor_error": anchor_stats,
    "multilabel_error": multilabel_stats,
    "lno_metrics": lno_stats,
    "lno_confusion": lno_confusion_record,
    "label_count_bins": count_bin_stats,
    "cardinality": cardinality_stats,
    "per_class_f1": per_class_f1,
    "pair_analysis": pair_stats,
    "length_bin_error": length_bin_stats,
}


def analyze_model(axes: EvalAxes, m: ModelResult, techniques: dict = TECHNIQUES) -> dict:
    """등록된 기법을 순회해 모델별 결과 레코드를 만든다({json_key: fn(axes, m)})."""
    return {key: fn(axes, m) for key, fn in techniques.items()}


# ── 교차 모델 ─────────────────────────────────────────────────────────────

def diff_profile(mask: np.ndarray, owner: ModelResult, axes: EvalAxes) -> dict:
    """차집합 문서의 오류 유형(owner 기준)·k≥2 비중·길이 bin 분포."""
    length_bin, bins, k_gold = axes.length_bin, axes.bins, axes.k_gold
    return {
        "n": int(mask.sum()),
        "k>=2": int((mask & (k_gold >= 2)).sum()),
        "sibling": int((mask & owner.sibling).sum()),
        "cross_lno": int((mask & owner.cross).sum()),
        "by_length_bin": {b: int((mask & (length_bin == b)).sum()) for b in bins},
    }


def compare(axes: EvalAxes, base: ModelResult, target: ModelResult, component: str) -> dict:
    """두 모델의 top-1 앵커 오류 차집합 — 어디서 고치고(fixed) 어디서 깨는지(broken)."""
    length_bin, bins = axes.length_bin, axes.bins
    base_err, target_err = base.err, target.err
    fixed = base_err & ~target_err      # base 오답 → target 정답
    broken = ~base_err & target_err     # base 정답 → target 오답
    return {
        "component": component,
        "base": base.tag,
        "target": target.tag,
        "fixed": diff_profile(fixed, base, axes),
        "broken": diff_profile(broken, target, axes),
        "net_gain": int(fixed.sum() - broken.sum()),
        # bin별 순이득 — 성분이 특정 길이 bin에 쏠리는지
        "net_by_bin": {
            b: int((fixed & (length_bin == b)).sum() - (broken & (length_bin == b)).sum())
            for b in bins
        },
        # bin별 교정률 = fixed / base 오류
        "fix_rate_by_bin": {
            b: round(float((fixed & (length_bin == b)).sum()
                           / max(int((base_err & (length_bin == b)).sum()), 1)), 4)
            for b in bins
        },
    }


def hard_core_mask(results) -> np.ndarray:
    """모든 모델이 공통으로 틀린 앵커 오류 문서 마스크."""
    return np.logical_and.reduce([m.err for m in results])


def hierarchy_verdict(tag: str, anchor_rec: dict, lno_rec: dict) -> dict:
    """계층 확장 판정 — 2단계(Lno→Mno) 추정 P@1이 flat P@1을 상회하는가."""
    delta = lno_rec["delta_vs_flat"]
    return {
        "anchor_tag": tag,
        "criterion": "2단계(Lno→Mno) 추정 P@1이 flat P@1을 상회하는가 여부",
        "flat_p@1": anchor_rec["p@1"],
        "lno_stage_p@1": lno_rec["p@1"],
        "oracle_lno_p@1": lno_rec["oracle_lno_p@1"],
        "two_stage_p@1_est": lno_rec["two_stage_p@1_est"],
        "delta_vs_flat": delta,
        "decision": "계층 확장 검토" if delta > 0 else "flat 유지",
        "descriptive": {
            "sibling_ratio": anchor_rec["sibling_ratio"],
            "chance_sibling_ratio": anchor_rec["chance_sibling_ratio"],
            "sibling_enrichment": anchor_rec["sibling_enrichment"],
        },
    }


# ── 파사드 ────────────────────────────────────────────────────────────────

class ErrorAnalysis:
    """오류 분석 세션 — 라벨 공간·공유 축·모델 결과·교차 모델 분석을 한 객체로 묶는다.

    노트북은 이 클래스 하나만 import한다.

        EA = ErrorAnalysis(label_mapping, num_labels=188, tau=0.5)   # 라벨 공간 구성
        EA.set_data(ds)                                              # 정답·길이·카디널리티 축
        EA.add(MODELS, cache_dir, split)                            # 로짓 로드→빌드→기법 분석
        EA.add_logits(tag, logits)                                  # 가공한 로짓 배열을 직접 등록
        EA.records[tag] / EA.models[tag]                            # 표·저장용 결과
        EA.compare(base, target) · EA.hard_core() · EA.hierarchy_verdict(tag) · EA.pair_symmetry(tag)
    """

    def __init__(self, label_mapping: dict, num_labels: int = 188, tau: float = DEFAULT_TAU, bins=None):
        self.ls = LabelSpace(label_mapping["id2mno"], label_mapping["mno2lno"], num_labels)
        self.num_labels = num_labels
        self._tau = tau
        self._bins = bins
        self.axes = None
        self.models = {}     # tag -> ModelResult
        self.records = {}    # tag -> analyze_model 결과

    def set_data(self, ds) -> "ErrorAnalysis":
        """데이터셋에서 공유 축(정답 `Y`·길이 bin·`k_gold`)을 구성한다."""
        self.axes = EvalAxes.from_dataset(ds, self.ls, self.num_labels, tau=self._tau, bins=self._bins)
        return self

    # 공유 축 접근자(set_data 이후) — 노트북 표·verify 셀에서 참조
    @property
    def Y(self): return self.axes.Y

    @property
    def length_bin(self): return self.axes.length_bin

    @property
    def k_gold(self): return self.axes.k_gold

    @property
    def bins(self): return self.axes.bins

    @property
    def tau(self): return self.axes.tau

    @property
    def n(self) -> int: return len(self.axes.Y)

    def add(self, models, cache_dir, split, techniques: dict = TECHNIQUES) -> "ErrorAnalysis":
        """모델 목록의 로짓을 읽어 `ModelResult` 빌드 + 기법 레지스트리 분석까지 수행."""
        for d in models:
            self.add_logits(d["tag"], load_logits(cache_dir, d["tag"], split), techniques)
        return self

    def add_logits(self, tag: str, logits: np.ndarray, techniques: dict = TECHNIQUES) -> "ErrorAnalysis":
        """로짓 배열을 직접 등록해 분석한다 — 캐시 파일 그대로가 아니라 가공한 로짓
        (예: 다른 test 판의 로짓을 공통 문서 행으로 정렬한 것)을 넣는 경로."""
        assert logits.shape == (self.n, self.num_labels), tag
        m = ModelResult.build(self.axes, tag, logits)
        self.models[tag] = m
        self.records[tag] = analyze_model(self.axes, m, techniques)
        return self

    def compare(self, base_tag: str, target_tag: str, component: str = "loss") -> dict:
        """두 모델의 top-1 앵커 오류 차집합(fixed/broken)."""
        return compare(self.axes, self.models[base_tag], self.models[target_tag], component)

    def hard_core(self, tags=None) -> np.ndarray:
        """지정 모델(기본 전체)이 공통으로 틀린 앵커 오류 마스크."""
        return hard_core_mask([self.models[t] for t in (tags or self.models)])

    def hierarchy_verdict(self, tag: str) -> dict:
        """앵커·Lno 결과로부터 계층 확장 판정."""
        r = self.records[tag]
        return hierarchy_verdict(tag, r["anchor_error"], r["lno_metrics"])

    def pair_symmetry(self, tag: str):
        """혼동 행렬의 무향 쌍 질량·대칭도, off-diagonal 합계."""
        return pair_symmetry(self.models[tag].confusion, self.ls)
