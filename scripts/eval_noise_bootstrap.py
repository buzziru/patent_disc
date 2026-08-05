"""평가 표본 잡음 — 고정 test에 대한 bootstrap 신뢰구간.

시드 잡음(훈련 재현성)과는 다른 성분이다. 여기서 재는 것은 **모델을 고정한 채 test 문서 표본을
다시 뽑았을 때 지표가 얼마나 흔들리는가**이며, GPU 없이 이미 덤프된 로짓만으로 계산된다.
재표본 단위는 **문서**다 — 한 문서의 188개 라벨 결정은 서로 독립이 아니므로 함께 뽑고 함께 뺀다.

두 모델 비교에는 **같은 재표본을 양쪽에 적용하는 paired bootstrap**을 쓴다. 두 모델이 같은 문서에서
함께 틀리는 상관이 상쇄돼 델타 구간이 단일 모델 구간보다 훨씬 좁다 — 델타 판정은 이 구간으로 한다.

평가 축은 정리 test(11,244)로 통일한다. 구 test(11,271) 축의 로짓은 `doc_ids_test.json`에서 위치를
찾아 사영한다(`output/headline_cleaned_test.json`과 같은 절차) — 서로 다른 test에서 잰 값을
나란히 놓지 않기 위해서다.

실행: `uv run python scripts/eval_noise_bootstrap.py [--B 2000] [--seed 42]`
산출: `output/eval_noise_bootstrap.json`

참조하는 HF 데이터셋은 공개 배포하지 않는다 — 재생성 절차는
`docs/data/data-pipeline.md`「가공 데이터셋은 배포하지 않는다 — 재현 경로」.
"""

import argparse
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

from datasets import load_dataset          # noqa: E402  (HF_HOME 설정 뒤 import)
from error_analysis import build_gold      # noqa: E402

OUT = ROOT / "output"
RAW_DS = "ingyoun/patent-clean-text"       # 정리본 — test 11,244
SPLIT = "test"
NUM_LABELS = 188

# 표기 이름 → 로짓 캐시 tag
MODELS = {
    "exp1(8192)": "modernbert-patent-len8192",
    "exp2(512)": "modernbert-patent-len512",
    "KoBERT": "kobert-patent-baseline_len512",
    "BCE": "modernbert-patent-len512-bce",
    "ZLPR": "modernbert-patent-len512-zlpr",
    "ASL": "modernbert-patent-len512-asl",
    "11_01(b128)": "modernbert-patent-len512-b128",
    "11_04(seed153)": "modernbert-patent-seed153",
}

# 델타 = A − B. 문서에 이미 쓰여 있는 주장들.
PAIRS = [
    ("exp1(8192)", "exp2(512)", "길이 성분 (+0.84pt 주장)"),
    ("exp2(512)", "KoBERT", "모델 성분 (+0.99pt 주장)"),
    ("11_01(b128)", "exp2(512)", "신 레시피 focal vs 구 레시피 focal ('잡음 내' 주장)"),
    ("11_01(b128)", "BCE", "손실: focal vs BCE (동일 레시피)"),
    ("11_01(b128)", "ZLPR", "손실: focal vs ZLPR (동일 레시피)"),
    ("11_01(b128)", "ASL", "손실: focal vs ASL (동일 레시피)"),
    # 시드 축 — 설정이 같고 시드만 다른 쌍이라 델타의 기댓값이 0이다. 구간이 0을 포함해야 정상이며,
    # 그 폭이 '표본 잡음만으로 시드 델타가 얼마나 흔들리는가'다.
    ("11_04(seed153)", "11_01(b128)", "시드 쌍 — 훈련 잡음 (seed 153 − 42, 기댓값 0)"),
    ("11_04(seed153)", "BCE", "손실: focal vs BCE — 두 번째 시드(재판정)"),
]


def clean_axis():
    """정리 test의 정답 다중핫 Y와, 구 test 행 → 정리 test 행 사영 인덱스."""
    ds = load_dataset(RAW_DS, split=SPLIT)
    Y = build_gold(ds["label_ids"], len(ds), NUM_LABELS)
    clean_ids = list(ds["document_id"])

    stored = json.loads((OUT / f"doc_ids_clean_{SPLIT}.json").read_text(encoding="utf-8"))
    assert stored == clean_ids, "doc_ids_clean이 정리 데이터셋 순서와 다르다(정리 축 로짓의 행 축)"

    old_ids = json.loads((OUT / f"doc_ids_{SPLIT}.json").read_text(encoding="utf-8"))
    pos = {d: i for i, d in enumerate(old_ids)}
    keep = np.array([pos[d] for d in clean_ids])
    assert np.all(np.diff(keep) > 0), "사영 인덱스가 단조 증가가 아니다(행 순서 불일치)"
    return Y, keep, len(old_ids)


def ssot_micro(tag):
    """test micro-F1의 SSOT(정리 축). 없으면 None."""
    headline = json.loads((OUT / "headline_cleaned_test.json").read_text(encoding="utf-8"))
    if tag in headline["models"]:
        return headline["models"][tag]["new"]["micro"]
    fp = OUT / f"{tag}_metrics.json"
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))["test"]["test_micro_f1"]
    return None


def load_logits(tag, keep, n_old, n_clean):
    """로짓을 정리 test 축으로 얹는다. 파일이 없거나 행 수가 낯설면 None."""
    fp = OUT / f"logits_{tag}_{SPLIT}.npy"
    if not fp.exists():
        return None, f"로짓 캐시 없음 ({fp.name})"
    lg = np.load(fp)
    if lg.shape[0] == n_old:
        lg = lg[keep]
    elif lg.shape[0] != n_clean:
        return None, f"행 수 {lg.shape[0]}가 구 test({n_old})도 정리 test({n_clean})도 아님"
    return lg, None


def micro(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)


def per_doc_terms(logits, Y):
    """문서×라벨 TP/FP/FN. τ=0.5 ⟺ logit ≥ 0(focal native 임계, metrics.py 규약)."""
    pred = logits >= 0.0
    return (pred & Y), (pred & ~Y), (~pred & Y)


def boot_metrics(tp, fp, fn, W, k_gold):
    """재표본 가중 W(B, N)에 대한 지표 분포. 각 항은 (B,)."""
    n = tp.shape[0]
    tpf, fpf, fnf = tp.astype(np.float32), fp.astype(np.float32), fn.astype(np.float32)
    tpd, fpd, fnd = tpf.sum(1), fpf.sum(1), fnf.sum(1)
    multi = (k_gold >= 2).astype(np.float32)

    cls = [W @ m for m in (tpf, fpf, fnf)]                      # (B, C) 클래스별 합
    out = {
        "micro": micro(W @ tpd, W @ fpd, W @ fnd),
        "macro": micro(*cls).mean(1),
        "sample": (W @ micro(tpd, fpd, fnd)) / n,
        "micro_k>=2": micro(W @ (tpd * multi), W @ (fpd * multi), W @ (fnd * multi)),
    }
    point = {
        "micro": float(micro(tpd.sum(), fpd.sum(), fnd.sum())),
        "macro": float(micro(tpf.sum(0), fpf.sum(0), fnf.sum(0)).mean()),
        "sample": float(micro(tpd, fpd, fnd).mean()),
        "micro_k>=2": float(micro((tpd * multi).sum(), (fpd * multi).sum(), (fnd * multi).sum())),
    }
    return point, out


def ci(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000, help="부트스트랩 반복 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Y, keep, n_old = clean_axis()
    n = len(Y)
    k_gold = Y.sum(1)
    print(f"[축] 정리 test N={n:,} (구 test {n_old:,}에서 사영) · k>=2 {int((k_gold >= 2).sum()):,}건")

    # 재표본 가중 — 모든 모델이 같은 W를 본다(paired).
    rng = np.random.default_rng(args.seed)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=args.B).astype(np.float32)
    print(f"[부트스트랩] B={args.B:,} · seed={args.seed}")

    point, dist, skipped = {}, {}, {}
    for name, tag in MODELS.items():
        lg, why = load_logits(tag, keep, n_old, n)
        if lg is None:
            skipped[name] = why
            continue
        p, d = boot_metrics(*per_doc_terms(lg, Y), W, k_gold)

        ref = ssot_micro(tag)
        if ref is not None and abs(p["micro"] - ref) > 1e-3:
            # 행 순열이면 로짓이 다른 문서의 라벨과 대면한다 — SSOT와 크게 어긋난다.
            skipped[name] = f"SSOT 불일치 micro {p['micro']:.4f} vs {ref:.4f} — 행 축 어긋남(재덤프 필요)"
            continue
        point[name], dist[name] = p, d

    for name, why in skipped.items():
        print(f"[skip] {name}: {why}")

    metrics = ["micro", "macro", "sample", "micro_k>=2"]
    print(f"\n== 단일 모델 (점추정 · 95% CI · sd, pt) ==")
    print(f"{'모델':<14}" + "".join(f"{m:>30}" for m in metrics))
    for name in point:
        row = f"{name:<14}"
        for m in metrics:
            lo, hi = ci(dist[name][m])
            row += f"{100*point[name][m]:>10.2f} [{100*lo:5.2f},{100*hi:5.2f}] ±{100*dist[name][m].std():4.2f}"
        print(row)

    print(f"\n== paired 델타 (A − B, pt) ==")
    print(f"{'비교':<28}{'micro Δ':>9}{'95% CI':>18}{'sd':>7}{'P(Δ<=0)':>9}   {'k>=2 Δ':>8}{'95% CI':>18}  설명")
    deltas = {}
    for a, b, note in PAIRS:
        if a not in dist or b not in dist:
            continue
        rec = {}
        for m in metrics:
            d = dist[a][m] - dist[b][m]
            lo, hi = ci(d)
            rec[m] = {
                "point_pt": round(100 * (point[a][m] - point[b][m]), 3),
                "ci95_pt": [round(100 * lo, 3), round(100 * hi, 3)],
                "sd_pt": round(100 * float(d.std()), 3),
                "p_sign_flip": round(float((d <= 0).mean() if point[a][m] > point[b][m] else (d >= 0).mean()), 4),
            }
        deltas[f"{a} - {b}"] = {"note": note, **rec}
        r, rk = rec["micro"], rec["micro_k>=2"]
        r_ci = "[%.2f,%.2f]" % tuple(r["ci95_pt"])
        rk_ci = "[%.2f,%.2f]" % tuple(rk["ci95_pt"])
        print(f"{a + ' - ' + b:<28}{r['point_pt']:>9.2f}{r_ci:>18}{r['sd_pt']:>7.2f}"
              f"{r['p_sign_flip']:>9.3f}   {rk['point_pt']:>8.2f}{rk_ci:>18}  {note}")

    payload = {
        "axis": {"split": SPLIT, "dataset": RAW_DS, "n": n, "n_old": n_old, "tau": 0.5},
        "bootstrap": {"B": args.B, "seed": args.seed, "unit": "document", "paired": True},
        "skipped": skipped,
        "single": {
            name: {
                m: {
                    "point_pt": round(100 * point[name][m], 3),
                    "ci95_pt": [round(100 * c, 3) for c in ci(dist[name][m])],
                    "sd_pt": round(100 * float(dist[name][m].std()), 3),
                }
                for m in metrics
            }
            for name in point
        },
        "paired_delta": deltas,
    }
    fp = OUT / "eval_noise_bootstrap.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {fp}")


if __name__ == "__main__":
    main()
