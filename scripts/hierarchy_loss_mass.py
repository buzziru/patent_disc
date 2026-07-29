"""계층 손실(MCLoss) 축 — 그룹 항이 닿는 오류 질량과 micro 민감도.

MCLoss를 `y_L = max_{m in L} h_m`(별도 `Lno` 출력 없음)로 축약하면 두 항이 남는다.

  음성 그룹 항  -log(1 - max_{m in L} sigma(h_m))   : 정답 `Lno`가 아닌 그룹의 최상위를 누른다.
  양성 그룹 항  -log(max_{m in L, t=1} sigma(h_m)) : 그룹당 정답 하나가 확신되면 포화된다.

두 항이 **닿는 오류**는 주 지표(멀티라벨 micro)의 FP/FN 중 일부로 정확히 정의된다.

  - 음성 항이 닿는 FP = 그 문서의 정답 `Lno` **밖**에 떨어진 FP. 정답 `Lno` 안의 FP(형제
    오분류)는 그 그룹이 양성이라 음성 항이 걸리지 않는다.
  - 양성 항이 압력을 거는 FN = 그 `Lno` 그룹의 정답을 **하나도 못 맞힌** 문서의 FN. 같은
    그룹의 형제를 이미 맞혔다면 max가 포화돼 압력이 0이다.

이 스크립트는 저장된 로짓(GPU 0)에서 두 질량을 세고, 그 일부가 회수됐을 때의 micro를
민감도로 낸다(오라클 제거·회수이므로 목표치가 아니라 상한이다). k=1/k>=2 슬라이스로 갈라
[ADR-0009](../docs/adr/0009-loss-axis-closure.md)의 FP:FN 부호 뒤집힘과 대면시킨다.

평가 축은 정리 test(11,244). 구 test(11,271) 축의 로짓은 `doc_ids_test.json`에서 위치를
찾아 사영한다(`scripts/hierarchy_conditional.py`와 같은 절차).

실행: `uv run python scripts/hierarchy_loss_mass.py`
산출: `output/hierarchy_loss_mass.json`
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

from datasets import load_dataset                      # noqa: E402  (HF_HOME 설정 뒤 import)
from huggingface_hub import hf_hub_download            # noqa: E402
from error_analysis import LabelSpace, build_gold      # noqa: E402

OUT = ROOT / "output"
RAW_DS = "ingyoun/patent-clean-text"
SPLIT = "test"
NUM_LABELS = 188

MODELS = {
    "KoBERT(512)": "kobert-patent-baseline_len512",
    "exp2(A.X 512)": "modernbert-patent-len512",
    "exp1(A.X 8192)": "modernbert-patent-len8192",
    "11_01(A.X 512)": "modernbert-patent-len512-b128",
}
PRIMARY = "11_01(A.X 512)"          # 계층 손실 arm의 비교 기준선
SHARES = [0.1, 0.2, 0.3, 0.5]       # 표적 질량 중 회수 비율(민감도 축)
# 11_01은 headline_cleaned_test.json에 없어 `docs/experiments/loss-function.md` 실측표로 대조한다
REF_B128 = {"modernbert-patent-len512-b128": 0.8588}


def clean_axis():
    """정리 test의 정답 다중핫과, 구 test 행 → 정리 test 행 사영 인덱스."""
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


def load_logits(tag, keep, n_old, n_clean):
    lg = np.load(OUT / f"logits_{tag}_{SPLIT}.npy")
    if lg.shape[0] == n_old:
        lg = lg[keep]
    assert lg.shape == (n_clean, NUM_LABELS), (tag, lg.shape)
    return lg


def micro(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def counts(P, Y, col_pos_group, grp_hit, sel=None):
    """FP/FN을 그룹 항이 닿는 것과 닿지 않는 것으로 가른다."""
    if sel is None:
        sel = np.ones(len(Y), bool)
    tp, fp, fn = (P & Y)[sel], (P & ~Y)[sel], (~P & Y)[sel]
    cross = fp & ~col_pos_group[sel]           # 정답 Lno 밖 FP — 음성 항의 표적
    missed = fn & ~grp_hit[sel]                # 그룹 전체를 놓친 FN — 양성 항의 압력
    return dict(
        tp=int(tp.sum()), fp=int(fp.sum()), fn=int(fn.sum()),
        fp_cross_lno=int(cross.sum()), fp_within_lno=int((fp & col_pos_group[sel]).sum()),
        fn_group_missed=int(missed.sum()), fn_group_saturated=int((fn & grp_hit[sel]).sum()),
    )


def analyse(z, Y, ls, col_pos_group):
    P = z >= 0.0                                # sigmoid >= 0.5
    tp = P & Y

    # 그룹 적중 마스크 — 열 c가 속한 Lno에서 정답을 하나라도 맞혔는가
    grp_hit = np.zeros_like(Y, dtype=bool)
    for l in range(ls.L):
        cols = np.where(ls.lno_idx == l)[0]
        grp_hit[:, cols] = (tp[:, cols].sum(1) > 0)[:, None]

    k = Y.sum(1)
    total = counts(P, Y, col_pos_group, grp_hit)
    assert total["fp_cross_lno"] + total["fp_within_lno"] == total["fp"]
    assert total["fn_group_missed"] + total["fn_group_saturated"] == total["fn"]

    base = micro(total["tp"], total["fp"], total["fn"])
    sens = {
        "suppress_cross_lno_fp": {
            f"{int(100 * r)}%": round(
                micro(total["tp"], total["fp"] - int(total["fp_cross_lno"] * r), total["fn"]), 4)
            for r in SHARES},
        "recover_group_missed_fn": {
            f"{int(100 * r)}%": round(
                micro(total["tp"] + int(total["fn_group_missed"] * r), total["fp"],
                      total["fn"] - int(total["fn_group_missed"] * r)), 4)
            for r in SHARES},
    }
    return dict(
        micro=round(base, 4), total=total,
        slices={"k=1": counts(P, Y, col_pos_group, grp_hit, k == 1),
                "k>=2": counts(P, Y, col_pos_group, grp_hit, k >= 2)},
        sensitivity_oracle=sens,
    )


def main():
    lm = json.load(open(hf_hub_download(RAW_DS, "label_mappings.json", repo_type="dataset"),
                        encoding="utf-8"))
    ls = LabelSpace(lm["id2mno"], lm["mno2lno"], NUM_LABELS)

    Y, keep, n_old = clean_axis()
    n = len(Y)
    col_pos_group = ls.to_lno(Y)[:, ls.lno_idx]     # (n,C) 열 c의 Lno가 정답 Lno 집합에 있는가

    headline = json.loads((OUT / "headline_cleaned_test.json").read_text(encoding="utf-8"))
    res = {}
    for name, tag in MODELS.items():
        z = load_logits(tag, keep, n_old, n)
        res[name] = analyse(z, Y, ls, col_pos_group)
        # 11_01(b128)은 headline_cleaned_test.json에 없다 — loss-function.md 실측표의 값으로 대조
        ref = (headline["models"][tag]["new"]["micro"] if tag in headline["models"]
               else REF_B128[tag])
        assert abs(res[name]["micro"] - ref) < 1e-3, (name, res[name]["micro"], ref)

    p = res[PRIMARY]
    print(f"정리 test {n:,} · 기준선 {PRIMARY} micro {p['micro']:.4f}")
    print(f"  음성 항 표적(정답 Lno 밖 FP) {p['total']['fp_cross_lno']:,} / FP {p['total']['fp']:,}"
          f" ({p['total']['fp_cross_lno'] / p['total']['fp']:.1%})")
    print(f"  양성 항 표적(그룹 전체 놓친 FN) {p['total']['fn_group_missed']:,} / FN {p['total']['fn']:,}"
          f" ({p['total']['fn_group_missed'] / p['total']['fn']:.1%})")
    for s, v in p["sensitivity_oracle"].items():
        print(f"  {s}: " + " · ".join(
            f"{r} → {m:.4f} ({100 * (m - p['micro']):+.2f}pt)" for r, m in v.items()))
    for name, sl in p["slices"].items():
        print(f"  [{name}] FP {sl['fp']:,} 중 Lno 밖 {sl['fp_cross_lno']:,}"
              f" · FN {sl['fn']:,} 중 그룹 전체 놓침 {sl['fn_group_missed']:,}")

    payload = {
        "note": "계층 손실(MCLoss) 그룹 항이 닿는 오류 질량과 micro 민감도. "
                "음성 항 표적 = 정답 Lno 밖 FP, 양성 항 표적 = 그룹 전체를 놓친 FN. "
                "sensitivity_oracle은 표적 질량의 일부를 오라클로 제거·회수했을 때의 micro이므로 "
                "도달 불가 상한이지 목표치가 아니다. 평가 축 = 정리 test 11,244, tau=0.5.",
        "n": n, "split": SPLIT, "tau": 0.5, "primary": PRIMARY,
        "script": "scripts/hierarchy_loss_mass.py",
        "verify": "FP = cross + within · FN = missed + saturated · "
                  "재계산 micro == output/headline_cleaned_test.json (4/4)",
        "models": res,
    }
    (OUT / "hierarchy_loss_mass.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT / 'hierarchy_loss_mass.json'}")


if __name__ == "__main__":
    main()
