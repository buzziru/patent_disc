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
    "14_01(MCLoss)": "modernbert-patent-len512-mcl",
}
PRIMARY = "11_01(A.X 512)"          # 계층 손실 arm의 비교 기준선
ARM = "14_01(MCLoss)"               # 기준선과 손실만 다른 arm
SHARES = [0.1, 0.2, 0.3, 0.5]       # 표적 질량 중 회수 비율(민감도 축)
# headline_cleaned_test.json에 없는 런은 훈련 잡의 test 지표로 대조한다
REF_EXTRA = {
    "modernbert-patent-len512-b128": 0.8588,    # docs/experiments/loss-function.md 실측표
    "modernbert-patent-len512-mcl": 0.8467,     # notebook_output/14_01_HierLoss_MCLoss.ipynb
}


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


def group_hit_mask(tp, Y, ls):
    """열 c가 속한 `Lno`에서 정답을 하나라도 맞혔는가 — 양성 항의 포화 여부."""
    grp_hit = np.zeros_like(Y, dtype=bool)
    for l in range(ls.L):
        cols = np.where(ls.lno_idx == l)[0]
        grp_hit[:, cols] = (tp[:, cols].sum(1) > 0)[:, None]
    return grp_hit


def analyse(z, Y, ls, col_pos_group):
    P = z >= 0.0                                # sigmoid >= 0.5
    tp = P & Y
    grp_hit = group_hit_mask(tp, Y, ls)

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


def target_movement(base, arm, n):
    """두 런의 표적 질량을 절대량·문서당·share 세 축으로 나란히 낸다.

    share(표적/전체 FP·FN)는 관찰 지표로 쓸 수 없다 — 분모인 전체 FP·FN이 같이 움직이므로
    표적 절대량이 늘어도 share가 내려가고, 줄어도 올라간다. 판정은 절대량·문서당으로 한다.
    """
    out = {"micro": {"base": base["micro"], "arm": arm["micro"],
                     "delta": round(arm["micro"] - base["micro"], 4)}}
    for key, denom in [("fp", "fp"), ("fn", "fn"),
                       ("fp_cross_lno", "fp"), ("fp_within_lno", "fp"),
                       ("fn_group_missed", "fn"), ("fn_group_saturated", "fn")]:
        b, a = base["total"][key], arm["total"][key]
        out[key] = {
            "base": b, "arm": a, "delta": a - b,
            "share_base": round(b / base["total"][denom], 4),
            "share_arm": round(a / arm["total"][denom], 4),
            "per_doc_base": round(b / n, 4), "per_doc_arm": round(a / n, 4),
        }
    return out


def paired_bootstrap(za, zb, Y, n_boot=4000, seed=0):
    """문서 단위 재표본으로 두 런의 micro 델타 분포를 낸다(평가·표본 잡음 성분).

    두 런이 같은 test 문서를 보므로 짝지어 재표본한다 — `eval_noise_bootstrap.py`와 같은 절차.
    """
    rng = np.random.default_rng(seed)
    cells = []
    for z in (za, zb):
        P = z >= 0.0
        cells.append(((P & Y).sum(1), (P & ~Y).sum(1), (~P & Y).sum(1)))
    n = len(Y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        m = [micro(tp[i].sum(), fp[i].sum(), fn[i].sum()) for tp, fp, fn in cells]
        deltas[b] = m[1] - m[0]
    d = deltas * 100
    return {
        "n_boot": n_boot, "seed": seed,
        "delta_pt": round(float(d.mean()), 3),
        "ci95_pt": [round(float(np.percentile(d, 2.5)), 3),
                    round(float(np.percentile(d, 97.5)), 3)],
        "sd_pt": round(float(d.std()), 3),
        "p_delta_ge_0": round(float((d >= 0).mean()), 4),
    }


def cells_at(z, tau, Y, ls, col_pos_group):
    """임계 tau에서 FP/FN을 표적·비표적으로 가른다(작동점 정규화용)."""
    P = z >= np.log(tau / (1 - tau))
    tp, fp, fn = P & Y, P & ~Y, ~P & Y
    grp_hit = group_hit_mask(tp, Y, ls)
    return {
        "tau": round(float(tau), 3),
        "micro": round(float(micro(tp.sum(), fp.sum(), fn.sum())), 4),
        "pred_per_doc": round(float(P.sum(1).mean()), 4),
        "fp": int(fp.sum()), "fp_cross_lno": int((fp & ~col_pos_group).sum()),
        "fp_within_lno": int((fp & col_pos_group).sum()),
        "fn": int(fn.sum()), "fn_group_missed": int((fn & ~grp_hit).sum()),
        "fn_group_saturated": int((fn & grp_hit).sum()),
    }


def matched_operating_point(za, zb, Y, ls, col_pos_group, taus=np.arange(0.50, 0.95, 0.005)):
    """arm의 작동점을 기준선에 맞춘 뒤 표적 질량을 비교한다.

    arm이 기준선보다 많이 예측하면(empty rate 하락) tau=0.5 비교에서 모든 오류 칸이 함께
    늘어 표적의 증감이 작동점 이동에 섞인다. 예측량을 맞추고 재야 두 항이 표적에 남긴
    효과와 순위 자체의 변화가 갈린다. tau는 진단용 정규화이며 임계 정책이 아니다
    (임계값 축은 `PROJECT.md` 「닫힌 갈래」).
    """
    base = cells_at(za, 0.5, Y, ls, col_pos_group)
    rows = {"base": base, "arm_tau0.5": cells_at(zb, 0.5, Y, ls, col_pos_group)}
    for label, key in [("arm_matched_fp", "fp"), ("arm_matched_pred", "pred_per_doc")]:
        cand = [cells_at(zb, t, Y, ls, col_pos_group) for t in taus]
        rows[label] = min(cand, key=lambda c: abs(c[key] - base[key]))
    keys = ["fp", "fp_cross_lno", "fp_within_lno", "fn", "fn_group_missed", "fn_group_saturated"]
    return {
        "rows": rows,
        "delta_vs_base": {name: {**{k: rows[name][k] - base[k] for k in keys},
                                 "micro_pt": round(100 * (rows[name]["micro"] - base["micro"]), 2)}
                          for name in rows if name != "base"},
    }


def main():
    lm = json.load(open(hf_hub_download(RAW_DS, "label_mappings.json", repo_type="dataset"),
                        encoding="utf-8"))
    ls = LabelSpace(lm["id2mno"], lm["mno2lno"], NUM_LABELS)

    Y, keep, n_old = clean_axis()
    n = len(Y)
    col_pos_group = ls.to_lno(Y)[:, ls.lno_idx]     # (n,C) 열 c의 Lno가 정답 Lno 집합에 있는가

    headline = json.loads((OUT / "headline_cleaned_test.json").read_text(encoding="utf-8"))
    res, logits = {}, {}
    for name, tag in MODELS.items():
        z = logits[name] = load_logits(tag, keep, n_old, n)
        res[name] = analyse(z, Y, ls, col_pos_group)
        ref = (headline["models"][tag]["new"]["micro"] if tag in headline["models"]
               else REF_EXTRA[tag])
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

    # 표적 질량의 이동 — 절대량으로 읽는다. share(표적/FP)는 분모가 함께 움직여 방향을 뒤집어 보인다
    movement = target_movement(res[PRIMARY], res[ARM], n)
    print(f"\n{ARM} vs {PRIMARY} — 표적 질량 이동 (n={n:,})")
    for key, v in movement.items():
        if key == "micro":
            print(f"  micro {v['base']:.4f} → {v['arm']:.4f} ({100 * v['delta']:+.2f}pt)")
            continue
        print(f"  {key}: {v['base']:,} → {v['arm']:,} ({v['delta']:+,})"
              f" · share {v['share_base']:.3f} → {v['share_arm']:.3f}"
              f" · 문서당 {v['per_doc_base']:.4f} → {v['per_doc_arm']:.4f}")

    boot = paired_bootstrap(logits[PRIMARY], logits[ARM], Y)
    print(f"\n  paired bootstrap({boot['n_boot']}회): Δmicro {boot['delta_pt']:+.3f}pt"
          f" · CI95 [{boot['ci95_pt'][0]:+.3f}, {boot['ci95_pt'][1]:+.3f}]"
          f" · sd {boot['sd_pt']:.3f}pt · P(Δ≥0)={boot['p_delta_ge_0']:.4f}")

    matched = matched_operating_point(logits[PRIMARY], logits[ARM], Y, ls, col_pos_group)
    print("\n  작동점 정규화 — 예측량을 기준선에 맞춘 뒤 표적 질량")
    for name, r in matched["rows"].items():
        print(f"    {name:>17} τ {r['tau']:.3f} micro {r['micro']:.4f}"
              f" pred/doc {r['pred_per_doc']:.4f} FP {r['fp']:,}"
              f"(cross {r['fp_cross_lno']:,}/within {r['fp_within_lno']:,})"
              f" FN {r['fn']:,}(missed {r['fn_group_missed']:,}/sat {r['fn_group_saturated']:,})")
    for name, dv in matched["delta_vs_base"].items():
        print(f"    {name:>17} Δ cross {dv['fp_cross_lno']:+,} · within {dv['fp_within_lno']:+,}"
              f" · missed {dv['fn_group_missed']:+,} · sat {dv['fn_group_saturated']:+,}"
              f" · micro {dv['micro_pt']:+.2f}pt")

    payload = {
        "note": "계층 손실(MCLoss) 그룹 항이 닿는 오류 질량과 micro 민감도. "
                "음성 항 표적 = 정답 Lno 밖 FP, 양성 항 표적 = 그룹 전체를 놓친 FN. "
                "sensitivity_oracle은 표적 질량의 일부를 오라클로 제거·회수했을 때의 micro이므로 "
                "도달 불가 상한이지 목표치가 아니다. 평가 축 = 정리 test 11,244, tau=0.5.",
        "n": n, "split": SPLIT, "tau": 0.5, "primary": PRIMARY, "arm": ARM,
        "script": "scripts/hierarchy_loss_mass.py",
        "verify": "FP = cross + within · FN = missed + saturated · "
                  "재계산 micro == 훈련 잡 test 지표 (5/5)",
        "models": res,
        "target_movement": movement,
        "movement_note": "share(표적/전체 FP·FN)는 분모가 함께 움직이므로 관찰 지표로 쓸 수 없다. "
                         "표적 절대량과 문서당 값으로 판정한다.",
        "paired_bootstrap": boot,
        "matched_operating_point": matched,
        "matched_note": "arm이 기준선보다 많이 예측해(empty rate 1.17%→0.61%) tau=0.5 비교에는 "
                        "작동점 이동이 섞인다. 예측량을 맞추면 음성 항 표적(cross FP)은 의도한 "
                        "방향으로 움직이고 양성 항 표적(group missed FN)은 모든 작동점에서 "
                        "악화한다. 어느 작동점에서도 micro가 회복되지 않으므로 캘리브레이션 "
                        "차이가 아니라 순위 자체의 손실이다. tau는 진단용 정규화이며 임계 정책이 아니다.",
    }
    (OUT / "hierarchy_loss_mass.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT / 'hierarchy_loss_mass.json'}")


if __name__ == "__main__":
    main()
