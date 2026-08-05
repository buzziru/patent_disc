"""KD 게이트 — 앙상블 이득이 단일 student로 옮겨갈 수 있는 구조인가.

`output/kd_gate_ensemble.json`의 헤드룸 게이트는 **"증류할 상한이 존재하는가"**만 물었다
(앙상블 +0.73pt · k>=2 +1.42pt). 이 스크립트는 그 다음 질문을 묻는다 —
**"그 상한이 단일 함수로 근사 가능한 구조인가."** 앙상블 이득은 서로 다른 함수를 평균해
나오는데 student는 함수 하나만 될 수 있으므로, 이득의 성질에 따라 전이량이 갈린다.

  A. 이득의 구성   앙상블이 exp1을 고친 원소와 깨뜨린 원소를 세고, soft target q의 여유
                  |q-0.5|로 결정의 견고함을 잰다. 여유가 없는(knife-edge) 고침은 student가
                  따라가기 어렵고 잡음에 가깝다.
  B. 캘리브레이션 배제  exp1 로짓의 재조정(global tau · per-class tau)만으로 앙상블 이득에
                  닿는가. 닿으면 KD가 아니라 이미 닫힌 임계값·캘리브레이션 축이다
                  (`PROJECT.md` 닫힌 갈래). 오라클과 val→test 일반화분을 함께 낸다.
  C. 특화 vs 분산  teacher가 갈리는 문서에서 "누가 맞는가"가 관측 축(길이 bin x k)으로
                  예측되는가. val에서 적합한 라우팅 규칙을 test에 적용해 이득을 잰다.
                  예측되면 학습 가능한 특화이고, 안 되면 순수 분산 평균이다.
  D. 집중도       고친 원소가 길이 bin·k·`Lno`에 몰려 있는가(관측/기대 비). 몰려 있으면
                  규칙성, 균등하면 우연에 가깝다.

평가 축은 정리 test(11,244)·정리 val(11,132), tau=0.5. 구 축 로짓은 `doc_ids_*.json`으로
사영한다(`hierarchy_loss_mass.py`와 같은 절차). GPU 0 — 덤프된 로짓만 쓴다.

실행: `uv run python scripts/kd_transfer_structure.py`
산출: `output/kd_transfer_structure.json`

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
from error_analysis import LabelSpace, build_gold      # noqa: E402

OUT = ROOT / "output"
RAW_DS = "ingyoun/patent-clean-text"
NUM_LABELS = 188

TEACHERS = {                                            # 게이트가 확정한 teacher 3종
    "exp1": "modernbert-patent-len8192",
    "asl": "modernbert-patent-len512-asl",
    "kobert": "kobert-patent-baseline_len512",
}
W = {"exp1": 0.5, "asl": 0.2, "kobert": 0.3}            # 게이트 val 선택 가중(확률 공간)
PRIMARY = "exp1"                                        # student가 넘어야 하는 최고 단일 teacher
KNIFE = 0.05                                            # |q-0.5| < KNIFE = 여유 없는 결정


# ── 축 · 로짓 ────────────────────────────────────────────────────────────────
def load_axis(split):
    ds = load_gold(split, OUT)
    Y = build_gold(ds["label_ids"], len(ds), NUM_LABELS)
    clean_ids = list(ds["document_id"])
    stored = json.loads((OUT / f"doc_ids_clean_{split}.json").read_text(encoding="utf-8"))
    assert stored == clean_ids, f"doc_ids_clean_{split}이 데이터셋 순서와 다르다(로짓 행 축)"
    old_ids = json.loads((OUT / f"doc_ids_{split}.json").read_text(encoding="utf-8"))
    pos = {d: i for i, d in enumerate(old_ids)}
    keep = np.array([pos[d] for d in clean_ids])
    assert np.all(np.diff(keep) > 0), "사영 인덱스가 단조 증가가 아니다"
    return Y, np.array(ds["length_bin"]), keep, len(old_ids)


def load_probs(split, keep, n_old, n_clean):
    """teacher별 sigmoid 확률 — 앙상블은 확률 공간에서 만든다(게이트와 동일)."""
    P = {}
    for name, tag in TEACHERS.items():
        z = np.load(OUT / f"logits_{tag}_{split}.npy").astype(np.float64)
        if z.shape[0] == n_old:
            z = z[keep]
        assert z.shape == (n_clean, NUM_LABELS), (name, split, z.shape)
        P[name] = 1.0 / (1.0 + np.exp(-z))
    P["ens"] = sum(W[k] * P[k] for k in TEACHERS)
    return P


def micro(pred, Y):
    tp = int((pred & Y).sum())
    return 2 * tp / (2 * tp + int((pred & ~Y).sum()) + int((~pred & Y).sum()))


# ── A. 이득의 구성 — 고친 원소·깨뜨린 원소와 결정 여유 ───────────────────────
def composition(P, Y):
    base, ens = P[PRIMARY] >= 0.5, P["ens"] >= 0.5
    q = P["ens"]
    fixed = (base != Y) & (ens == Y)                    # exp1 틀림 → 앙상블 맞음
    broken = (base == Y) & (ens != Y)                   # 그 반대
    margin = np.abs(q - 0.5)

    # 고침이 몇 대 몇으로 만들어졌나 — 두 약한 teacher가 함께 뒤집었으면 일관된 정정,
    # 한쪽이 겨우 평균을 기울였으면 knife-edge에 가깝다
    agree_pair = ((P["asl"] >= 0.5) == (P["kobert"] >= 0.5)) & ((P["asl"] >= 0.5) != base)

    def band(m):
        return dict(
            n=int(m.sum()),
            knife_edge=int((m & (margin < KNIFE)).sum()),
            knife_edge_share=round(float((margin[m] < KNIFE).mean()), 4) if m.sum() else None,
            median_margin=round(float(np.median(margin[m])), 4) if m.sum() else None,
            two_vs_one=int((m & agree_pair).sum()),
            two_vs_one_share=round(float(agree_pair[m].mean()), 4) if m.sum() else None,
        )

    # 이득이 얼마나 얕은 밴드에 실렸나 — 여유 m 이상에서만 앙상블 결정을 채택하고 나머지는
    # exp1을 그대로 둔다. m을 조금 올렸을 때 이득이 사라지면 결정이 취약하다는 뜻이다.
    by_margin = {}
    for m in (0.0, 0.02, 0.05, 0.10, 0.20):
        take = margin >= m
        hyb = np.where(take, ens, base)
        by_margin[f"m>={m}"] = dict(
            micro=round(micro(hyb, Y), 4),
            pt=round(100 * (micro(hyb, Y) - micro(base, Y)), 2),
            adopted=int(take.sum()))

    return dict(
        micro_base=round(micro(base, Y), 4), micro_ens=round(micro(ens, Y), 4),
        gain_pt=round(100 * (micro(ens, Y) - micro(base, Y)), 2),
        fixed=band(fixed), broken=band(broken),
        net_elements=int(fixed.sum() - broken.sum()),
        two_vs_one_precision=round(
            float(comp_2v1 := fixed[agree_pair].sum() / max(1, (fixed | broken)[agree_pair].sum())), 4),
        # 대조군 — 앙상블이 원래 맞히던 원소의 여유(고침 밴드가 얼마나 얕은지 읽는 기준)
        reference_all_correct_median_margin=round(float(np.median(margin[ens == Y])), 4),
        gain_by_margin_band=by_margin,
        knife_note=f"knife-edge = |q-0.5| < {KNIFE}",
        margin_note="gain_by_margin_band = 여유 m 이상에서만 앙상블 결정을 채택했을 때의 micro. "
                    "m을 조금만 올려도 이득이 사라지면 이득이 임계 근처의 얕은 뒤집기에 실렸다는 뜻.",
    ), fixed, broken


# ── B. 캘리브레이션 배제 — exp1 재조정만으로 닿는가 ──────────────────────────
def per_class_tau(p, Y, grid):
    """클래스별로 micro 기여(2tp - fp - fn 축)를 최대화하는 tau를 고른다."""
    taus = np.empty(NUM_LABELS)
    for c in range(NUM_LABELS):
        pc, yc = p[:, c], Y[:, c]
        best, bt = -1e18, 0.5
        for t in grid:
            pred = pc >= t
            # micro의 분자·분모에 대한 이 클래스의 기여를 결정하는 양
            s = 2 * int((pred & yc).sum()) - int((pred & ~yc).sum()) - int((~pred & yc).sum())
            if s > best:
                best, bt = s, t
        taus[c] = bt
    return taus


def calibration_reach(Pv, Yv, Pt, Yt, grid=np.arange(0.05, 0.96, 0.01)):
    """exp1 로짓의 재조정(global tau · per-class tau)이 앙상블 이득에 닿는가.

    오라클(test 직접 튜닝)과 일반화분(val 적합 → test 적용)을 함께 낸다. 오라클은 도달
    불가 상한이고, 판정은 일반화분으로 한다.
    """
    pv, pt = Pv[PRIMARY], Pt[PRIMARY]
    base_t = micro(pt >= 0.5, Yt)
    ens_t = micro(Pt["ens"] >= 0.5, Yt)

    g_oracle = max((micro(pt >= t, Yt), t) for t in grid)
    g_val = max((micro(pv >= t, Yv), t) for t in grid)[1]
    pc_oracle = per_class_tau(pt, Yt, grid)
    pc_val = per_class_tau(pv, Yv, grid)
    return dict(
        base_test=round(base_t, 4), ensemble_test=round(ens_t, 4),
        gain_pt=round(100 * (ens_t - base_t), 2),
        global_tau_oracle={"tau": round(float(g_oracle[1]), 3), "micro": round(g_oracle[0], 4),
                           "pt": round(100 * (g_oracle[0] - base_t), 2)},
        global_tau_val_fit={"tau": round(float(g_val), 3),
                            "micro": round(micro(pt >= g_val, Yt), 4),
                            "pt": round(100 * (micro(pt >= g_val, Yt) - base_t), 2)},
        per_class_tau_oracle={"micro": round(micro(pt >= pc_oracle, Yt), 4),
                              "pt": round(100 * (micro(pt >= pc_oracle, Yt) - base_t), 2)},
        per_class_tau_val_fit={"micro": round(micro(pt >= pc_val, Yt), 4),
                               "pt": round(100 * (micro(pt >= pc_val, Yt) - base_t), 2)},
        note="재조정이 앙상블 이득에 닿으면 KD가 아니라 닫힌 캘리브레이션 축이다. "
             "오라클은 도달 불가 상한이며 판정은 val 적합분으로 한다.",
    )


# ── C. 특화 vs 분산 — 라우팅 규칙이 val→test로 전이되는가 ────────────────────
def routing(Pv, Yv, binv, Pt, Yt, bint):
    """teacher가 갈리는 문서에서 '누가 맞는가'가 관측 축으로 예측되는가.

    val에서 칸마다 top-1 정확도가 최고인 teacher를 고르고 그 규칙을 test에 적용한다. 이득이
    0에 가까우면 승자가 문서마다 무작위 = 분산이고, 유의하면 학습 가능한 특화다. 상한은
    oracle-any(셋 중 하나라도 적중). 특징 집합을 둘로 두어 강한 쪽으로 판정한다 —
    (1) 길이 bin x k: KD 문서가 teacher 역할로 주장한 두 축.
    (2) + exp1 확신도 4분위: "주 teacher가 흔들리는 문서를 남에게 넘긴다"는 가장 자연스러운 규칙.
    """
    def axes(P, Y, lb, edges=None):
        top1 = {k: P[k].argmax(1) for k in TEACHERS}
        ok = {k: Y[np.arange(len(Y)), v] for k, v in top1.items()}
        disagree = ~np.all([top1[k] == top1[PRIMARY] for k in TEACHERS], axis=0)
        k = Y.sum(1)
        s = np.sort(P[PRIMARY], axis=1)
        conf = s[:, -1] - s[:, -2]                      # exp1 top-1 여유 = 확신도
        if edges is None:
            edges = np.quantile(conf, [0.25, 0.5, 0.75])
        cq = np.searchsorted(edges, conf)
        coarse = np.array([f"{b}|{min(int(x), 2)}" for b, x in zip(lb, k)])
        fine = np.array([f"{c}|q{q}" for c, q in zip(coarse, cq)])
        return ok, disagree, coarse, fine, edges

    okv, dv, cv, fv, edges = axes(Pv, Yv, binv)
    okt, dt, ct, ft, _ = axes(Pt, Yt, bint, edges)
    names = list(TEACHERS)
    sub = dt                                            # 판정은 불일치 부분집합에서 한다
    oracle = np.any([okt[k] for k in names], axis=0)
    gap = oracle[sub].mean() - okt[PRIMARY][sub].mean()

    out = {
        "n_disagree_test": int(sub.sum()), "disagree_share": round(float(sub.mean()), 4),
        "on_disagree": {f"{k}_p@1": round(float(okt[k][sub].mean()), 4) for k in names},
        "oracle_any_p@1": round(float(oracle[sub].mean()), 4),
        "oracle_gap_pt": round(100 * float(gap), 2),
        "rules": {},
        "note": "라우팅 이득이 0 근처면 승자가 문서별 무작위(분산 평균)이고, 크면 관측 축으로 "
                "예측되는 특화다. 상한은 oracle-any. 판정은 두 특징 집합 중 강한 쪽으로 한다.",
    }
    for label, (cell_v, cell_t) in {"length_bin x k": (cv, ct),
                                    "+ exp1 confidence quartile": (fv, ft)}.items():
        rule = {}
        for cell in sorted(set(cell_v[dv]) | set(cell_t[dt])):
            sel = dv & (cell_v == cell)
            rule[cell] = (PRIMARY if sel.sum() < 30                 # 표본이 얇은 칸은 기본 teacher
                          else max(names, key=lambda k: okv[k][sel].mean()))
        routed = np.array([okt[rule.get(c, PRIMARY)][i] for i, c in enumerate(cell_t)])
        out["rules"][label] = {
            "routed_p@1": round(float(routed[sub].mean()), 4),
            "delta_vs_primary_pt": round(100 * float(routed[sub].mean()
                                                     - okt[PRIMARY][sub].mean()), 2),
            "recovery_of_oracle_gap": round(float(
                (routed[sub].mean() - okt[PRIMARY][sub].mean()) / gap), 4),
            "n_cells": len(rule),
            "cells_not_primary": sum(1 for v in rule.values() if v != PRIMARY),
        }
    return out


# ── D. 집중도 — 고친 원소가 어디에 몰려 있나 ─────────────────────────────────
def concentration(fixed, broken, Y, lb, ls):
    """관측/기대 비 — 기대는 그 축이 가진 '틀릴 기회'(정답+예측 원소) 비례로 둔다."""
    base_err = fixed | broken                           # 두 함수가 갈린 원소 전체
    out = {}

    def by(name, keys, masks):
        tot_f, tot_b = int(fixed.sum()), int(broken.sum())
        rows = {}
        for key, m in zip(keys, masks):
            share_err = float(base_err[m].sum() / base_err.sum())
            f = int(fixed[m].sum())
            rows[key] = dict(
                fixed=f, broken=int(broken[m].sum()), net=f - int(broken[m].sum()),
                fixed_share=round(f / tot_f, 4), split_share=round(share_err, 4),
                obs_over_exp=round((f / tot_f) / share_err, 3) if share_err else None)
        out[name] = rows

    bins = sorted(set(lb))
    by("length_bin", bins, [np.repeat((lb == b)[:, None], NUM_LABELS, 1) for b in bins])
    k = Y.sum(1)
    kk = ["k=1", "k>=2"]
    by("cardinality", kk, [np.repeat((k == 1)[:, None], NUM_LABELS, 1),
                           np.repeat((k >= 2)[:, None], NUM_LABELS, 1)])
    lno = np.zeros((len(Y), NUM_LABELS), dtype=int) + ls.lno_idx[None, :]
    by("lno", [f"L{i}" for i in range(ls.L)], [lno == i for i in range(ls.L)])

    # 클래스 축 집중도 — 상위 몇 클래스가 순이득을 쥐고 있나
    net_c = fixed.sum(0).astype(int) - broken.sum(0).astype(int)
    order = np.argsort(-net_c)
    tot = int(net_c.sum())
    out["class_top10_net_share"] = round(float(net_c[order[:10]].sum() / tot), 4) if tot else None
    out["classes_with_positive_net"] = int((net_c > 0).sum())
    out["note"] = ("obs_over_exp = (그 축이 가진 고침 몫) / (그 축이 가진 갈림 몫). "
                   "1.0 근처면 균등(우연), 크게 벗어나면 집중(규칙성).")
    return out


def main():
    lm = json.load(open(OUT / "label_mappings.json",
                        encoding="utf-8"))
    ls = LabelSpace(lm["id2mno"], lm["mno2lno"], NUM_LABELS)

    Yv, binv, keepv, n_old_v = load_axis("val")
    Yt, bint, keept, n_old_t = load_axis("test")
    Pv = load_probs("val", keepv, n_old_v, len(Yv))
    Pt = load_probs("test", keept, n_old_t, len(Yt))

    # verify — 게이트가 기록한 단일·앙상블 micro가 재현되는가
    gate = json.loads((OUT / "kd_gate_ensemble.json").read_text(encoding="utf-8"))
    for k in TEACHERS:
        got, ref = micro(Pt[k] >= 0.5, Yt), gate["singles"][k]["micro"]
        assert abs(got - ref) < 1e-3, (k, got, ref)
    assert abs(micro(Pt["ens"] >= 0.5, Yt) - gate["ensembles"]["ens_best"]["micro"]) < 1e-3

    comp, fixed, broken = composition(Pt, Yt)
    calib = calibration_reach(Pv, Yv, Pt, Yt)
    route = routing(Pv, Yv, binv, Pt, Yt, bint)
    conc = concentration(fixed, broken, Yt, bint, ls)

    print(f"정리 test {len(Yt):,} · val {len(Yv):,} · tau=0.5 · 가중 {W}")
    print(f"\nA. 이득의 구성 — exp1 {comp['micro_base']:.4f} → 앙상블 {comp['micro_ens']:.4f}"
          f" ({comp['gain_pt']:+.2f}pt)")
    for band in ("fixed", "broken"):
        b = comp[band]
        print(f"  {band:>6} {b['n']:,}개 · knife-edge(|q-0.5|<{KNIFE}) {b['knife_edge']:,}"
              f"({b['knife_edge_share']:.1%}) · 중앙 여유 {b['median_margin']:.4f}"
              f" · 2대1 정정 {b['two_vs_one']:,}({b['two_vs_one_share']:.1%})")
    print(f"  순 원소 {comp['net_elements']:+,} · 2대1 정정의 정확도"
          f" {comp['two_vs_one_precision']:.1%} · 대조(앙상블 정답 원소) 중앙 여유"
          f" {comp['reference_all_correct_median_margin']:.4f}")
    print("  여유 밴드별 이득(그 밴드에서만 앙상블 결정 채택): " + " · ".join(
        f"{k} {v['pt']:+.2f}pt" for k, v in comp["gain_by_margin_band"].items()))

    print(f"\nB. 캘리브레이션 배제 — 앙상블 이득 {calib['gain_pt']:+.2f}pt에 재조정이 닿는가")
    for k in ("global_tau_oracle", "global_tau_val_fit",
              "per_class_tau_oracle", "per_class_tau_val_fit"):
        v = calib[k]
        print(f"  {k:>22} micro {v['micro']:.4f} ({v['pt']:+.2f}pt)")

    print(f"\nC. 특화 vs 분산 — 불일치 문서 {route['n_disagree_test']:,}"
          f"({route['disagree_share']:.1%})에서")
    for k, v in route["on_disagree"].items():
        print(f"  {k:>26} {v:.4f}")
    print(f"  {'oracle_any_p@1':>26} {route['oracle_any_p@1']:.4f}"
          f" (gap {route['oracle_gap_pt']:+.2f}pt)")
    for label, r in route["rules"].items():
        print(f"  [{label}] routed {r['routed_p@1']:.4f} ({r['delta_vs_primary_pt']:+.2f}pt)"
              f" · gap 회수율 {r['recovery_of_oracle_gap']:+.1%}"
              f" · 칸 {r['n_cells']}개 중 exp1 아닌 칸 {r['cells_not_primary']}개")

    print("\nD. 집중도 — 관측/기대")
    for axis in ("length_bin", "cardinality"):
        print(f"  [{axis}] " + " · ".join(
            f"{k} {v['obs_over_exp']:.2f}(net {v['net']:+,})" for k, v in conc[axis].items()))
    lno_ratios = [v["obs_over_exp"] for v in conc["lno"].values() if v["obs_over_exp"]]
    print(f"  [lno] 관측/기대 {min(lno_ratios):.2f}~{max(lno_ratios):.2f}"
          f" · 클래스 상위10 순이득 몫 {conc['class_top10_net_share']:.1%}"
          f" · 순이득 양수 클래스 {conc['classes_with_positive_net']}/188")

    payload = {
        "gate": "kd_transfer_structure",
        "question": "앙상블 이득(+0.73pt)이 단일 student로 옮겨갈 수 있는 구조인가 — "
                    "헤드룸 게이트가 묻지 않은 두 번째 질문.",
        "test_n": len(Yt), "val_n": len(Yv), "tau": 0.5,
        "teachers": TEACHERS, "weights": W, "primary": PRIMARY,
        "script": "scripts/kd_transfer_structure.py",
        "verify": "teacher 3종·앙상블 test micro == output/kd_gate_ensemble.json · "
                  "로짓 행 축 == doc_ids_clean_{val,test}.json",
        "composition": comp,
        "calibration_reach": calib,
        "routing": route,
        "concentration": conc,
    }
    (OUT / "kd_transfer_structure.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT / 'kd_transfer_structure.json'}")


if __name__ == "__main__":
    main()
