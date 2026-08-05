"""계층 확장 축 — 1단계 적중 조건부 정확도·2단계 헤드룸·게이트 비용.

`06_01`의 2단계 추정 P@1(= `Lno` 단계 정확도 × 오라클-`Lno` P@1)은 오라클 항을 **전체 문서**
위에서 재므로 1단계가 이미 틀린 문서를 두 항 모두에서 계상한다. 둘째 항에 들어가야 하는 양은
조건부 정확도 P(2단계 적중 | 1단계 적중)이며, 여기서는 그 조건부 표본에서 직접 재추정한다.

세 가지를 낸다.
  (1) 조건부 재추정 — 조건부 정확도와 그것으로 교정한 2단계 추정. 1단계 적중 문서에서는
      `Lno` 마스크가 flat top-1 열을 반드시 포함해 argmax가 보존되므로 교정값은 flat과
      항등으로 일치한다(assert로 확인). 곧 마스킹 시뮬레이션은 계층 이득을 탐지하지 못한다.
  (2) 1단계 규칙 스윕 — 같은 로짓을 `Lno` 단위로 다르게 집계해 1단계를 고르면 argmax 보존이
      깨지므로 최종 P@1이 flat과 갈린다. 무훈련으로 1단계를 개선할 여지가 있는지 본다.
  (3) 게이트 비용 — 주 지표(micro-F1)에서 top-m `Lno` 게이트가 남기는 도달 가능 recall 상한.

평가 축은 정리 test(11,244). 구 test(11,271) 축의 로짓은 `doc_ids_test.json`에서 위치를 찾아
사영한다(`scripts/eval_noise_bootstrap.py`와 같은 절차).

실행: `uv run python scripts/hierarchy_conditional.py [--B 5000] [--seed 42]`
산출: `output/hierarchy_conditional.json` · `output/hierarchy_stage1_rules.json`

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
sys.path.insert(0, str(ROOT / "src"))

from gold_labels import load_gold                    # noqa: E402  (저장소 동봉 정답 축 — 데이터셋 불필요)
from error_analysis import LabelSpace, build_gold      # noqa: E402

OUT = ROOT / "output"
RAW_DS = "ingyoun/patent-clean-text"
SPLIT = "test"
NUM_LABELS = 188
TAU = 0.5

MODELS = {
    "KoBERT(512)": "kobert-patent-baseline_len512",
    "exp2(A.X 512)": "modernbert-patent-len512",
    "exp1(A.X 8192)": "modernbert-patent-len8192",
    "11_01(A.X 512)": "modernbert-patent-len512-b128",
}
ANCHOR = "exp1(A.X 8192)"
RULES = ["max", "logsumexp", "top2_mean", "noisy_or", "prob_sum", "mean"]


def clean_axis():
    """정리 test의 정답 다중핫과, 구 test 행 → 정리 test 행 사영 인덱스."""
    ds = load_gold(SPLIT, OUT)
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
    """로짓을 정리 test 축으로 얹는다."""
    lg = np.load(OUT / f"logits_{tag}_{SPLIT}.npy")
    if lg.shape[0] == n_old:
        lg = lg[keep]
    assert lg.shape == (n_clean, NUM_LABELS), (tag, lg.shape)
    return lg


def conditional(z, Y, ls):
    """조건부 재추정 — 1단계 적중 표본에서 2단계 정확도를 직접 잰다."""
    n = len(Y)
    rows = np.arange(n)
    gold_col = ls.to_lno(Y)[:, ls.lno_idx]       # (n,C) 열 c의 Lno가 정답 Lno 집합에 있는가
    top1 = z.argmax(1)
    flat = Y[rows, top1]                          # flat top-1 적중
    stage1 = gold_col[rows, top1]                 # 1단계(유도 Lno) 적중
    assert (flat & ~stage1).sum() == 0, "flat 적중인데 1단계 실패 — Mno→Lno 매핑 불일치"

    z_oracle = np.where(gold_col, z, -np.inf)     # 정답 Lno 열 전체로 제한
    orc = Y[rows, z_oracle.argmax(1)]
    same_lno = ls.lno_idx[None, :] == ls.lno_idx[top1][:, None]
    z_gate = np.where(same_lno, z, -np.inf)       # 1단계가 고른 단일 Lno 열로 제한
    gated = Y[rows, z_gate.argmax(1)]

    # 항등의 근거 — 1단계 적중 문서에서는 두 마스크 모두 top1을 보존한다
    assert (z_oracle.argmax(1)[stage1] == top1[stage1]).all()
    assert (z_gate.argmax(1)[stage1] == top1[stage1]).all()

    # sibling 오류(1단계 적중·2단계 실패)에서 정답이 마스크 내 몇 위인가 — 2단계 표적의 폭
    sib = stage1 & ~flat
    order = np.argsort(-z_gate[sib], axis=1)
    gold_sib = Y[sib]
    rank = np.array([np.where(gold_sib[i][order[i]])[0].min() + 1 for i in range(int(sib.sum()))])

    return dict(
        n=n, n_stage1_hit=int(stage1.sum()), n_stage1_miss=int((~stage1).sum()),
        flat_p1=float(flat.mean()), stage1_acc=float(stage1.mean()),
        oracle_p1_all=float(orc.mean()),
        cond_hit=float(orc[stage1].mean()),          # 조건부 정확도(오라클 마스크)
        cond_hit_gate=float(gated[stage1].mean()),   # 조건부 정확도(단일 Lno 마스크)
        cond_miss=float(orc[~stage1].mean()),        # 1단계 실패 문서에서 마스크가 구제한 비율
        two_stage_product=float(stage1.mean() * orc.mean()),   # 결함 추정량 L×O
        two_stage_conditional=float(stage1.mean() * orc[stage1].mean()),
        sibling_rate=float(sib.mean()), cross_lno_rate=float((~stage1).mean()),
        stage2_headroom_pt=float(100 * (stage1.mean() - flat.mean())),
        sib_gold_rank2=float((rank == 2).mean()), sib_gold_rank_le3=float((rank <= 3).mean()),
        sib_mask_size=float(same_lno[sib].sum(1).mean()),
        _flat=flat, _stage1=stage1,
    )


def stage1_rules(z, Y, ls, cols_of):
    """1단계 집계 규칙을 바꿔 최종 P@1을 잰다 — 무훈련으로 1단계가 개선되는가."""
    n = len(Y)
    rows = np.arange(n)
    YL = ls.to_lno(Y)
    p = 1.0 / (1.0 + np.exp(-z))
    flat = Y[rows, z.argmax(1)]

    out = {}
    for kind in RULES:
        S = np.full((n, ls.L), -np.inf)
        for l, cols in enumerate(cols_of):
            zc, pc = z[:, cols], p[:, cols]
            if kind == "max":
                S[:, l] = zc.max(1)
            elif kind == "logsumexp":
                m = zc.max(1)
                S[:, l] = np.log(np.exp(zc - m[:, None]).sum(1)) + m
            elif kind == "top2_mean":
                S[:, l] = np.sort(zc, axis=1)[:, -2:].mean(1)
            elif kind == "noisy_or":
                S[:, l] = -np.log1p(-pc.clip(max=1 - 1e-12)).sum(1)
            elif kind == "prob_sum":
                S[:, l] = pc.sum(1)
            elif kind == "mean":
                S[:, l] = zc.mean(1)
        l_hat = S.argmax(1)
        hit1 = YL[rows, l_hat]
        mask = ls.lno_idx[None, :] == l_hat[:, None]
        final = Y[rows, np.where(mask, z, -np.inf).argmax(1)]
        out[kind] = dict(
            stage1_acc=float(hit1.mean()), cond_acc=float(final[hit1].mean()),
            final_p1=float(final.mean()),
            delta_vs_flat_pt=float(100 * (final.mean() - flat.mean())),
            ceiling_pt=float(100 * (hit1.mean() - flat.mean())),
            _final=final,
        )
    return flat, out


def gate_cost(z, Y, ls, cols_of, m_max=3):
    """주 지표 쪽 비용 — top-m Lno 게이트가 남기는 도달 가능 양성 라벨 비율(recall 상한)."""
    n = len(Y)
    p = 1.0 / (1.0 + np.exp(-z))
    pred = p >= TAU
    tp = int((pred & Y).sum())
    prec, rec = tp / int(pred.sum()), tp / int(Y.sum())

    S = np.stack([z[:, cols].max(1) for cols in cols_of], 1)
    order = np.argsort(-S, 1)
    keep = np.zeros((n, NUM_LABELS), dtype=bool)
    ceilings = {}
    for m in range(1, m_max + 1):
        keep |= ls.lno_idx[None, :] == order[:, m - 1][:, None]
        ceilings[f"top{m}"] = float((Y & keep).sum() / Y.sum())
    return dict(micro_precision=prec, micro_recall=rec,
                micro_f1=2 * prec * rec / (prec + rec), recall_ceiling=ceilings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=5000, help="부트스트랩 재표본 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lm = json.load(open(OUT / "label_mappings.json",
                        encoding="utf-8"))
    ls = LabelSpace(lm["id2mno"], lm["mno2lno"], NUM_LABELS)
    cols_of = [np.where(ls.lno_idx == l)[0] for l in range(ls.L)]

    Y, keep, n_old = clean_axis()
    n = len(Y)
    YL = ls.to_lno(Y)
    print(f"정리 test {n:,} · 정답 Lno 2개 이상 {(YL.sum(1) >= 2).mean():.2%}"
          f" ({int((YL.sum(1) >= 2).sum()):,}건, k>=2 문서 중 {(YL.sum(1) >= 2)[Y.sum(1) >= 2].mean():.2%})")

    cond, rules, gates = {}, {}, {}
    for name, tag in MODELS.items():
        z = load_logits(tag, keep, n_old, n)
        cond[name] = conditional(z, Y, ls)
        _, rules[name] = stage1_rules(z, Y, ls, cols_of)
        gates[name] = gate_cost(z, Y, ls, cols_of)

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.B, n))       # 문서 단위 재표본(모델 간 paired)
    for name, r in cond.items():
        f, s = r["_flat"][idx], r["_stage1"][idx]
        r["ci_cond_hit"] = [float(v) for v in np.percentile(f.sum(1) / s.sum(1), [2.5, 97.5])]
        r["ci_stage2_headroom_pt"] = [float(v) for v in
                                      np.percentile(100 * (s.mean(1) - f.mean(1)), [2.5, 97.5])]
        for kind, rr in rules[name].items():
            d = 100 * (rr["_final"][idx].mean(1) - f.mean(1))
            rr["ci_delta_pt"] = [float(v) for v in np.percentile(d, [2.5, 97.5])]

    names = list(MODELS)
    hdr = f"{'':<26}" + "".join(f"{x:>18}" for x in names)
    print("\n[조건부 재추정]")
    print(hdr); print("-" * len(hdr))
    for label, key in [("flat P@1", "flat_p1"), ("1단계 Lno 정확도", "stage1_acc"),
                       ("오라클 P@1(전체 문서)", "oracle_p1_all"),
                       ("  ├ 1단계 적중 조건부", "cond_hit"), ("  └ 1단계 실패 조건부", "cond_miss"),
                       ("2단계 추정(L×O)", "two_stage_product"),
                       ("2단계 추정(조건부)", "two_stage_conditional"),
                       ("sibling 오류율", "sibling_rate"), ("cross-Lno 오류율", "cross_lno_rate"),
                       ("sibling 정답 rank=2", "sib_gold_rank2"),
                       ("sibling 정답 rank<=3", "sib_gold_rank_le3")]:
        print(f"{label:<26}" + "".join(f"{cond[x][key]:>18.4f}" for x in names))
    print()
    for x in names:
        r = cond[x]
        assert abs(r["two_stage_conditional"] - r["flat_p1"]) < 1e-12, x   # 항등
        print(f"  {x:<18} 조건부 {r['cond_hit']:.4f}"
              f" [{r['ci_cond_hit'][0]:.4f}, {r['ci_cond_hit'][1]:.4f}] (n={r['n_stage1_hit']:,})"
              f" · 2단계 헤드룸 {r['stage2_headroom_pt']:+.2f}pt "
              f"[{r['ci_stage2_headroom_pt'][0]:.2f}, {r['ci_stage2_headroom_pt'][1]:.2f}]")
    print("  verify pass — 조건부 교정 2단계 추정 == flat P@1 (4모델·두 마스크 argmax 보존)")

    print("\n[1단계 규칙 스윕]")
    for x in names:
        print(f"  {x}  flat P@1 {cond[x]['flat_p1']:.4f}")
        print(f"    {'규칙':<12}{'1단계':>9}{'조건부':>9}{'최종 P@1':>10}{'Δ flat':>10}{'2단계 상한':>11}")
        for kind in RULES:
            r = rules[x][kind]
            print(f"    {kind:<12}{r['stage1_acc']:>9.4f}{r['cond_acc']:>9.4f}{r['final_p1']:>10.4f}"
                  f"{r['delta_vs_flat_pt']:>+9.2f}pt{r['ceiling_pt']:>+10.2f}pt"
                  f"   [{r['ci_delta_pt'][0]:+.2f}, {r['ci_delta_pt'][1]:+.2f}]")

    print("\n[게이트 비용 — 주 지표]")
    for x in names:
        g = gates[x]
        c = g["recall_ceiling"]
        print(f"  {x:<18} τ=0.5 micro P {g['micro_precision']:.4f} · R {g['micro_recall']:.4f}"
              f" · F1 {g['micro_f1']:.4f} || recall 상한 top1 {c['top1']:.4f} ·"
              f" top2 {c['top2']:.4f} · top3 {c['top3']:.4f}")

    meta = {"split": SPLIT, "n_docs": n, "raw_ds": RAW_DS, "tau": TAU, "seed": args.seed, "B": args.B,
            "anchor": ANCHOR, "num_labels": NUM_LABELS,
            "multi_lno_doc_rate": float((YL.sum(1) >= 2).mean())}
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    (OUT / "hierarchy_conditional.json").write_text(json.dumps(
        {**meta, "models": {x: {**strip(cond[x]), "gate": gates[x]} for x in names}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "hierarchy_stage1_rules.json").write_text(json.dumps(
        {**meta, "models": {x: {"flat_p1": cond[x]["flat_p1"],
                                "rules": {k: strip(v) for k, v in rules[x].items()}} for x in names}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT / 'hierarchy_conditional.json'} · {OUT / 'hierarchy_stage1_rules.json'}")


if __name__ == "__main__":
    main()
