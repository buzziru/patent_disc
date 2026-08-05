"""공개 데모(평가 탐색기)가 읽는 사전 계산 페이로드.

데모는 정적 페이지다 — 모델도 데이터셋도 싣지 않고, 저장된 로짓과 동봉 정답만으로
평가를 재계산해 보여준다. 브라우저에서 로짓 (11,244 × 188)을 다루는 것은 과하므로
τ 스윕을 여기서 미리 돌려 곡선만 넘긴다.

담기는 축 셋:
  ① 채점법  — τ별 멀티라벨 지표(micro F1·P·R·empty)와 top-1 weighted-F1.
               후자는 argmax 기반이라 **τ와 무관하다** — 데모가 보이려는 것이 이 대비다.
  ② top-1 손실 — 문서당 정답 개수 분포. top-1 채점은 문서당 정답 하나만 남긴다.
  ③ 길이 구간 — `final_model_comparison.json`의 구간별 micro와 델타를 그대로 옮긴다.

실행: `uv run python scripts/build_demo_payload.py`
산출: `output/demo_payload.json`, `hf-spaces/patent-eval-demo/payload.js`

`payload.js`는 같은 내용을 `window.PATENT_EVAL`로 감싼 래퍼다. 데모가 fetch 없이
`<script src>`로 읽어, 데이터 재생성이 손으로 쓴 `index.html`을 건드리지 않게 한다.

참조하는 HF 데이터셋은 공개 배포하지 않는다 — 재생성 절차는
`docs/data/data-pipeline.md`「가공 데이터셋은 배포하지 않는다 — 재현 경로」.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import f1_score

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
ROOT = Path(os.environ["DATA_ROOT"])
sys.path.insert(0, str(ROOT / "src"))

from gold_labels import load_gold                        # noqa: E402  (저장소 동봉 정답 축 — 데이터셋 불필요)
from error_analysis import build_gold                    # noqa: E402
from patent_train.metrics import empty_rate, f1_triple   # noqa: E402

OUT = ROOT / "output"
DEMO = ROOT / "hf-spaces" / "patent-eval-demo"
RAW_DS = "ingyoun/patent-clean-text"
SPLIT = "test"
NUM_LABELS = 188
TAU = 0.5
AXIS_NOTE = "정리 test 11,244. 구 test(11,271) 축 로짓은 doc_ids_test.json으로 사영."

# 표시 순서 = 서사 순서(재현 비교선 → 512 → 배포 4096 → 상한 8192).
# `label`은 독자용 이름이며 `final_model_comparison.json`의 내부 표기와 다르다.
MODELS = [
    ("kobert-patent-baseline_len512", "KoBERT (512)",     "재현한 비교선", "baseline"),
    ("modernbert-patent-len512",      "A.X-Encoder (512)", "exp2",         ""),
    ("modernbert-patent-len512-b128",  "A.X-Encoder (512)", "앵커",         "anchor"),
    ("modernbert-patent-len4096-op",   "A.X-Encoder (4096)", "배포 모델",   "target"),
    ("modernbert-patent-len8192",      "A.X-Encoder (8192)", "창 상한",     ""),
]

# 슬라이더 격자. 양 끝은 극단 동작을 보여주되 sigmoid 포화 구간은 피한다.
TAUS = [round(t, 2) for t in np.arange(0.05, 0.9501, 0.01)]

BIN_NAMES = ["<=512", "512-1024", "1024-2048", ">2048"]


def load_logits_clean(tag: str, keep: np.ndarray, n_clean: int) -> np.ndarray:
    """로짓을 정리 test 축(11,244행)으로 맞춰 읽는다. 구 축 파일만 사영한다."""
    z = np.load(OUT / f"logits_{tag}_{SPLIT}.npy")
    if z.shape[0] != n_clean:
        z = z[keep]
    assert z.shape == (n_clean, NUM_LABELS), f"{tag} 로짓 shape가 {z.shape}다"
    return z


def micro_at(z: np.ndarray, Y: np.ndarray, tau: float) -> dict:
    """τ 한 점의 micro 지표. sigmoid(z) >= τ ⟺ z >= log(τ/(1-τ))."""
    P = z >= np.log(tau / (1.0 - tau))
    tp = int(np.count_nonzero(P & Y))
    fp = int(np.count_nonzero(P & ~Y))
    fn = int(np.count_nonzero(~P & Y))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "micro_f1": f1,
        "precision": prec,
        "recall": rec,
        "empty_rate": float((P.sum(1) == 0).mean()),
    }


def main():
    comp = json.loads((OUT / "final_model_comparison.json").read_text(encoding="utf-8"))
    head = comp["headline"]

    gold = load_gold(SPLIT, OUT)
    n = len(gold)
    label_ids = gold["label_ids"]
    length_bin = np.asarray(gold["length_bin"])
    Y = build_gold(label_ids, n, NUM_LABELS)
    assert n == comp["n_docs"], f"정답 축 {n} ≠ 비교표 {comp['n_docs']}"

    # 구 축(11,271) → 정리 축(11,244) 사영 인덱스 — error_analysis_final.py와 같은 방식.
    old_ids = json.loads((OUT / f"doc_ids_{SPLIT}.json").read_text(encoding="utf-8"))
    clean_ids = list(gold["document_id"])
    pos = {d: i for i, d in enumerate(old_ids)}
    keep = np.array([pos[d] for d in clean_ids])
    assert np.all(np.diff(keep) > 0), "사영 인덱스가 단조 증가가 아니다(행 순서 불일치)"

    models = []
    for tag, label, note, role in MODELS:
        z = load_logits_clean(tag, keep, n)

        raw = [micro_at(z, Y, tau) for tau in TAUS]
        sweep = {k: [round(m[k], 5) for m in raw]
                 for k in ("micro_f1", "precision", "recall", "empty_rate")}

        # top-1 weighted-F1 — argmax 기반이라 τ가 들어가지 않는다.
        # 정답도 argmax로 하나만 남으므로 k>=2 문서의 나머지 정답은 채점에서 사라진다.
        anchor_wf1 = float(f1_score(Y.argmax(1), z.argmax(1), average="weighted", zero_division=0))

        # 무결성 — τ=0.5 지점과 앵커 지표가 비교표와 같아야 한다.
        i50 = TAUS.index(TAU)
        ref = head[tag]
        assert abs(sweep["micro_f1"][i50] - ref["micro_f1"]) < 1e-4, \
            f"{tag} micro가 비교표와 다르다: {sweep['micro_f1'][i50]} vs {ref['micro_f1']}"
        assert abs(sweep["empty_rate"][i50] - ref["empty_rate"]) < 1e-4, \
            f"{tag} empty가 비교표와 다르다: {sweep['empty_rate'][i50]} vs {ref['empty_rate']}"
        assert abs(anchor_wf1 - ref["anchor_weighted_f1"]) < 1e-4, \
            f"{tag} 앵커 지표가 비교표와 다르다: {anchor_wf1} vs {ref['anchor_weighted_f1']}"

        # f1_triple/empty_rate(SSOT)와도 대조 — micro를 numpy로 다시 짰으므로 정의 일치를 확인한다.
        P50 = (z >= 0.0).astype(int)
        assert abs(f1_triple(Y.astype(int), P50)["micro_f1"] - raw[i50]["micro_f1"]) < 1e-9, \
            f"{tag} micro 정의가 SSOT와 다르다"
        assert abs(empty_rate(P50) - raw[i50]["empty_rate"]) < 1e-9, \
            f"{tag} empty 정의가 SSOT와 다르다"

        models.append({
            "tag": tag,
            "label": label,
            "note": note,
            "role": role,
            "anchorWeightedF1": round(anchor_wf1, 5),
            "p1": ref["p@1"],
            "macroF1": ref["macro_f1"],
            "sampleF1": ref["sample_f1"],
            "sweep": sweep,
            "byLengthBin": comp["micro_by_length_bin"][tag],
        })
        print(f"  {label:20s} {note:8s}  micro@0.5 {sweep['micro_f1'][i50]:.4f}  top-1 wF1 {anchor_wf1:.4f}")

    # ── ② 문서당 정답 개수 — top-1 채점이 무엇을 버리는지 ──
    k = np.array([len(ids) for ids in label_ids])
    k_dist = {str(int(v)): int(c) for v, c in zip(*np.unique(k, return_counts=True))}
    n_gold = int(k.sum())
    n_multi = int((k >= 2).sum())
    dropped = n_gold - n            # top-1은 문서당 정답 하나만 채점한다
    assert sum(k_dist.values()) == n, "k 분포 합이 문서 수와 다르다"
    assert dropped == int((k - 1).sum()), "버려지는 라벨 수가 k 분포와 맞지 않는다"

    cardinality = {
        "nDocs": n,
        "nGoldLabels": n_gold,
        "scoredByTop1": n,
        "dropped": dropped,
        "droppedRate": round(dropped / n_gold, 4),
        "nMulti": n_multi,
        "multiRate": round(n_multi / n, 4),
        "kDist": k_dist,
    }

    # ── ③ 길이 구간 ──
    bin_counts = {b: int((length_bin == b).sum()) for b in BIN_NAMES}
    assert sum(bin_counts.values()) == n, "구간 문서 수 합이 문서 수와 다르다"

    payload = {
        "meta": {
            "split": SPLIT,
            "n_docs": n,
            "tau": TAU,
            "num_labels": NUM_LABELS,
            "raw_ds": RAW_DS,
            "script": "scripts/build_demo_payload.py",
            "axis_note": AXIS_NOTE,
        },
        "taus": TAUS,
        "tauIndex": TAUS.index(TAU),
        "models": models,
        "cardinality": cardinality,
        "lengthBins": {
            "names": BIN_NAMES,
            "counts": bin_counts,
            "deltaPt": comp["length_bin_delta_pt"],
        },
        "pairedBootstrap": comp["paired_bootstrap"],
    }

    fp = OUT / "demo_payload.json"
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    fp.write_text(body, encoding="utf-8")
    print(f"saved: {fp}  ({len(body) / 1024:.1f} KB)")

    DEMO.mkdir(parents=True, exist_ok=True)
    js = DEMO / "payload.js"
    js.write_text(
        "// 생성물 — scripts/build_demo_payload.py 가 만든다. 직접 편집하지 않는다.\n"
        f"window.PATENT_EVAL = {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))};\n",
        encoding="utf-8",
    )
    print(f"saved: {js}  ({js.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
