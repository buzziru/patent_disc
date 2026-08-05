"""훈련 곡선의 손실 상승 분해 — 평가 손실이 어느 원소에서 나오는가.

모든 훈련 런에서 val loss는 3~7 epoch에 최저를 찍고 이후 상승해 12 epoch에는 초기 수준으로
돌아오는 반면, F1 계열과 empty rate는 끝까지 개선된다. 이 스크립트는 그 손실이 **몇 개의 원소에
얼마나 몰려 있는지**를 세어, 상승의 정체가 성능 회귀가 아니라 소수 확신 오답으로의 질량 집중임을
보인다(서술은 `docs/experiments/training-curves.md`).

세 갈래를 잰다.

  1. **질량 집중도** — 원소별 focal(α=0.25, γ=2)을 내림차순 정렬해 상위 x%가 차지하는 손실 비율.
  2. **확률 포화도** — 양성의 p≥0.9 비율·음성의 p≤0.1 비율, 그리고 확신 오답(양성 p<0.1 ·
     음성 p>0.9)의 개수와 그것이 물고 있는 손실 질량.
  3. **운영점 위치** — global τ 스윕. 과신이 결정 경계를 밀었다면 최적 τ가 0.5에서 벗어나야 한다
     (벗어나지 않음 = F1 개선이 임계 아티팩트가 아님, [ADR-0004](../docs/adr/0004-threshold-policy.md)).

KD 타깃 진단도 같은 배터리로 낸다 — teacher 확률이 포화될수록 soft target `q`는 하드 라벨에
수렴해 증류할 정보가 줄어든다(`docs/experiments/knowledge-distillation.md`).

평가 축은 정리 test(11,244). 구 test(11,271) 축의 로짓은 `doc_ids_test.json`에서 위치를 찾아
사영한다(`scripts/hierarchy_loss_mass.py`와 같은 절차).

실행: `uv run python scripts/loss_mass_decomposition.py`
산출: `output/loss_mass_decomposition.json`

참조하는 HF 데이터셋은 공개 배포하지 않는다 — 재생성 절차는
`docs/data/data-pipeline.md`「가공 데이터셋은 배포하지 않는다 — 재현 경로」.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
ROOT = Path(os.environ["DATA_ROOT"])
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
sys.path.insert(0, str(ROOT / "src"))

from gold_labels import load_gold                    # noqa: E402  (저장소 동봉 정답 축 — 데이터셋 불필요)
from sklearn.metrics import f1_score                   # noqa: E402
from error_analysis import build_gold                  # noqa: E402

OUT = ROOT / "output"
RAW_DS = "ingyoun/patent-clean-text"
SPLIT = "test"
NUM_LABELS = 188
ALPHA, GAMMA = 0.25, 2                  # 프로젝트 확정 focal 하이퍼파라미터

MODELS = {
    "KoBERT(512)": "kobert-patent-baseline_len512",
    "exp2(A.X 512)": "modernbert-patent-len512",
    "exp1(A.X 8192)": "modernbert-patent-len8192",
    "ASL(A.X 512)": "modernbert-patent-len512-asl",
    "11_01(A.X 512 b128)": "modernbert-patent-len512-b128",
}
TEACHERS = {"exp1(A.X 8192)": 0.5, "ASL(A.X 512)": 0.2, "KoBERT(512)": 0.3}   # KD 게이트 선택 가중치
TOP_FRACS = [1e-4, 1e-3, 1e-2]          # 질량 집중도를 볼 상위 비율
TAUS = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]


def clean_axis():
    """정리 test의 정답 다중핫과, 구 test 행 → 정리 test 행 사영 인덱스."""
    ds = load_gold(SPLIT, OUT)
    Y = build_gold(ds["label_ids"], len(ds), NUM_LABELS).astype(bool)
    clean_ids = list(ds["document_id"])

    stored = json.loads((OUT / f"doc_ids_clean_{SPLIT}.json").read_text(encoding="utf-8"))
    assert stored == clean_ids, "doc_ids_clean이 정리 데이터셋 순서와 다르다(정리 축 로짓의 행 축)"

    old_ids = json.loads((OUT / f"doc_ids_{SPLIT}.json").read_text(encoding="utf-8"))
    pos = {d: i for i, d in enumerate(old_ids)}
    keep = np.array([pos[d] for d in clean_ids])
    assert np.all(np.diff(keep) > 0), "사영 인덱스가 단조 증가가 아니다(행 순서 불일치)"
    return Y, keep, len(old_ids)


def load_logits(tag, keep, n_old, n_clean):
    lg = np.load(OUT / f"logits_{tag}_{SPLIT}.npy").astype(np.float64)
    if lg.shape[0] == n_old:
        lg = lg[keep]
    assert lg.shape == (n_clean, NUM_LABELS), (tag, lg.shape)
    return lg


def focal_elements(logits, Y):
    """원소별 focal 손실 — 훈련 손실과 같은 정의(리덕션 전)."""
    bce = np.logaddexp(0, logits) - logits * Y          # -log p_t (수치 안정)
    return ALPHA * (1 - np.exp(-bce)) ** GAMMA * bce


def decompose(logits, Y):
    """한 모델의 손실 질량 집중도·포화도·운영점."""
    p = 1.0 / (1.0 + np.exp(-logits))
    fl = focal_elements(logits, Y)
    total = fl.sum()

    flat = np.sort(fl.ravel())[::-1]
    csum = np.cumsum(flat) / total
    concentration = {}
    for frac in TOP_FRACS:
        k = max(1, int(frac * flat.size))
        concentration[f"top_{frac:g}"] = {"n": k, "mass_share": float(csum[k - 1])}

    pos, neg = Y, ~Y
    conf_fn, conf_fp = pos & (p < 0.1), neg & (p > 0.9)     # 확신 오답
    n_conf = int(conf_fn.sum() + conf_fp.sum())

    tau_sweep = {}
    Yi = Y.astype(int)
    for tau in TAUS:
        pred = (p >= tau).astype(int)
        tau_sweep[f"{tau:.2f}"] = {
            "micro_f1": float(f1_score(Yi, pred, average="micro", zero_division=0)),
            "empty_rate": float((pred.sum(1) == 0).mean()),
        }
    best_tau = max(tau_sweep, key=lambda t: tau_sweep[t]["micro_f1"])

    return {
        "mean_focal_loss": float(fl.mean()),
        "n_elements": int(fl.size),
        "concentration": concentration,
        "mass_share_positive": float(fl[pos].sum() / total),
        "saturation": {
            "pos_ge_0.9": float((p[pos] >= 0.9).mean()),
            "neg_le_0.1": float((p[neg] <= 0.1).mean()),
        },
        "confident_errors": {
            "n": n_conf,
            "element_share": n_conf / fl.size,
            "mass_share": float((fl[conf_fn].sum() + fl[conf_fp].sum()) / total),
        },
        "tau_sweep": tau_sweep,
        "best_tau": best_tau,
        "tau_headroom_micro": tau_sweep[best_tau]["micro_f1"] - tau_sweep["0.50"]["micro_f1"],
    }


def soft_target_info(q, Y):
    """KD soft target의 정보량 — 하드 라벨과 구분되는 중간대(0.05~0.95) 원소가 얼마나 남는가."""
    mid = (q >= 0.05) & (q <= 0.95)
    ent = -(q * np.log(np.clip(q, 1e-12, 1)) + (1 - q) * np.log(np.clip(1 - q, 1e-12, 1)))
    return {
        "mid_band_share": float(mid.mean()),                 # 전체 원소 중 중간대 비율
        "mid_band_per_doc": float(mid.sum(1).mean()),        # 문서당 중간대 라벨 수(188 중)
        "mean_binary_entropy": float(ent.mean()),            # nat, 라벨당
        "l1_to_hard_label": float(np.abs(q - Y).mean()),     # 하드 라벨과의 평균 거리
    }


def main():
    Y, keep, n_old = clean_axis()
    n_clean = Y.shape[0]
    print(f"[axis] 정리 {SPLIT} {n_clean:,}행 × {NUM_LABELS} = {Y.size:,} 원소 · 양성 {Y.sum():,}({Y.mean()*100:.2f}%)")

    logits = {name: load_logits(tag, keep, n_old, n_clean) for name, tag in MODELS.items()}
    result = {name: decompose(lg, Y) for name, lg in logits.items()}

    print(f"\n{'모델':<22} {'mean loss':>10} {'상위0.1%질량':>12} {'확신오답':>9} {'그 질량':>8} {'최적τ':>6} {'τ헤드룸':>9}")
    for name, r in result.items():
        print(f"{name:<22} {r['mean_focal_loss']:>10.7f} {r['concentration']['top_0.001']['mass_share']*100:>11.1f}%"
              f" {r['confident_errors']['n']:>9,} {r['confident_errors']['mass_share']*100:>7.1f}%"
              f" {r['best_tau']:>6} {r['tau_headroom_micro']*100:>8.2f}pt")

    # KD soft target — 단일 teacher 확률과 게이트 가중 앙상블의 정보량 비교
    probs = {n: 1.0 / (1.0 + np.exp(-logits[n])) for n in TEACHERS}
    q = sum(w * probs[n] for n, w in TEACHERS.items())
    Yf = Y.astype(np.float64)
    soft = {n: soft_target_info(probs[n], Yf) for n in TEACHERS}
    soft["앙상블 q(0.5/0.2/0.3)"] = soft_target_info(q, Yf)

    print(f"\n{'KD soft target':<22} {'중간대 비율':>11} {'문서당 중간대':>13} {'평균 엔트로피':>13} {'하드라벨 L1':>11}")
    for name, s in soft.items():
        print(f"{name:<22} {s['mid_band_share']*100:>10.3f}% {s['mid_band_per_doc']:>13.2f}"
              f" {s['mean_binary_entropy']:>13.5f} {s['l1_to_hard_label']:>11.5f}")

    payload = {
        "axis": {"split": SPLIT, "n_docs": n_clean, "n_labels": NUM_LABELS,
                 "n_elements": int(Y.size), "n_positive": int(Y.sum())},
        "focal": {"alpha": ALPHA, "gamma": GAMMA},
        "models": result,
        "kd_soft_target": {"weights": TEACHERS, "stats": soft},
    }
    path = OUT / "loss_mass_decomposition.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {path}")


if __name__ == "__main__":
    main()
