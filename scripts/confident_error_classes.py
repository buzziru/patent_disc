"""확신 오답의 전원 동조(5/5)·앵커 단독(0/5) 상세 분류 — 보고·재라벨링 산출물.

[confident_error_diagnosis.py](confident_error_diagnosis.py)가 확신 오답 1,461개를 교차 모델
합의로 갈랐다(전원 동조 60.5% · 앵커 단독 1.6%). 이 스크립트는 그 두 극단을 **클래스 축으로
분해해** 성능 보고에 그대로 실을 수 있는 형태로 만든다.

  A. 5/5 확신 FP — 클래스 쌍   (정답 라벨, 확신 예측 라벨) 쌍을 빈도순으로 세고 lift·`Lno`
                              동일 여부를 붙인다. 특정 쌍에 몰려 있으면 재라벨링 요청의 단위가
                              그 쌍이 되고, 흩어져 있으면 전면적 라벨 잡음이다.
  B. 5/5 확신 FN — 클래스      모든 런이 부정하는 정답 라벨을 세고, 그 라벨이 문서의 다른 정답과
                              붙는 정도(lift)와 test per-class recall을 붙인다.
  C. 0/5 앵커 단독            시드 쌍둥이(`11_04`)의 확률로 이것이 훈련 잡음인지 본다. 쌍둥이가
                              맞히면 재현되지 않는 개별 런의 흔들림이다.
  D. 클리닝 연루 검정          [ADR-0010]이 제거한 라벨 충돌 42그룹의 연루 라벨과 위 5/5 집합이
                              겹치는가. 겹치면 확신 오답이 **이미 실측된 잡음과 같은 자리**에서
                              난다는 직접 증거다(제거는 입력 동일 케이스만 잡은 하한이므로,
                              같은 클래스의 잔여 충돌이 test에 남아 있을 수 있다).
  E. 재라벨링 후보 목록        위 판정을 문서 단위로 펼쳐 `output/relabel_candidates.csv`로 낸다.
                              실무에서 어노테이터에게 넘길 수 있는 형태다.

평가 축·모델·확신 밴드 정의는 `confident_error_diagnosis.py`와 같다. GPU 0.

실행: `uv run python scripts/confident_error_classes.py`
산출: `output/confident_error_classes.json` · `output/relabel_candidates.csv`

참조하는 HF 데이터셋은 공개 배포하지 않는다 — 재생성 절차는
`docs/data/data-pipeline.md`「가공 데이터셋은 배포하지 않는다 — 재현 경로」.
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
ROOT = Path(os.environ["DATA_ROOT"])
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
sys.path.insert(0, str(ROOT / "src"))

from gold_labels import load_gold                      # noqa: E402  (저장소 동봉 정답 축)
from error_analysis import LabelSpace, build_gold      # noqa: E402

OUT = ROOT / "output"
RAW_DS = "ingyoun/patent-clean-text"
NUM_LABELS = 188
CONF_LO, CONF_HI, TAU = 0.1, 0.9, 0.5
ANCHOR = "11_01"
TWIN = "11_04"
MODELS = {
    "11_01": "modernbert-patent-len512-b128",
    "11_04": "modernbert-patent-seed153",
    "exp2": "modernbert-patent-len512",
    "exp1": "modernbert-patent-len8192",
    "ASL": "modernbert-patent-len512-asl",
    "KoBERT": "kobert-patent-baseline_len512",
}
OTHERS = [m for m in MODELS if m != ANCHOR]
TOP_N = 15


def load_all():
    ds = load_gold("test", OUT)
    Y = build_gold(ds["label_ids"], len(ds), NUM_LABELS)
    clean_ids = list(ds["document_id"])
    assert json.loads((OUT / "doc_ids_clean_test.json").read_text(encoding="utf-8")) == clean_ids
    old_ids = json.loads((OUT / "doc_ids_test.json").read_text(encoding="utf-8"))
    pos = {d: i for i, d in enumerate(old_ids)}
    keep = np.array([pos[d] for d in clean_ids])
    P = {}
    for n, tag in MODELS.items():
        z = np.load(OUT / f"logits_{tag}_test.npy").astype(np.float64)
        if z.shape[0] == len(old_ids):
            z = z[keep]
        assert z.shape == (len(Y), NUM_LABELS), (n, z.shape)
        P[n] = 1.0 / (1.0 + np.exp(-z))
    return ds, Y, P


def train_lift():
    ids_list = load_gold("train", OUT)["label_ids"]
    n = len(ids_list)
    cnt = np.zeros(NUM_LABELS)
    co = np.zeros((NUM_LABELS, NUM_LABELS))
    for ids in ids_list:
        cnt[ids] += 1
        for a in ids:
            co[a, ids] += 1
    lift = co * n / np.outer(cnt, cnt)
    np.fill_diagonal(lift, 1.0)
    return lift, co, cnt, n


def conflict_labels():
    """[ADR-0010]이 제거한 라벨 충돌 42그룹의 연루 라벨 — 실측된 잡음의 클래스 축."""
    groups = json.loads((OUT / "label_conflict_docs.json").read_text(encoding="utf-8"))["conflict_groups"]
    involved, disputed, by_label = Counter(), Counter(), defaultdict(list)
    for gi, grp in enumerate(groups):
        sets = [set(json.loads(e.split(":", 2)[2])) for e in grp]
        shared = set.intersection(*sets)
        union = set().union(*sets)
        for c in union:
            involved[c] += 1
            by_label[c].append(gi)
        for c in union - shared:
            disputed[c] += 1
    return involved, disputed, by_label, groups


def main():
    lm = json.load(open(OUT / "label_mappings.json", encoding="utf-8"))
    ls = LabelSpace(lm["id2mno"], lm["mno2lno"], NUM_LABELS)
    mno = ls.mno_of_col
    ds, Y, P = load_all()
    Yb = Y.astype(bool)
    lift, co, cnt_tr, n_tr = train_lift()
    involved, disputed, by_label, groups = conflict_labels()

    pa = P[ANCHOR]
    conf = np.where(Yb, pa < CONF_LO, pa > CONF_HI)
    wrong = {m: np.where(Yb, P[m] < TAU, P[m] >= TAU) for m in OTHERS}
    n_agree = np.sum([wrong[m] for m in OTHERS], axis=0)
    gold_rows = [np.where(Yb[i])[0] for i in range(len(Y))]

    uni_fp = conf & ~Yb & (n_agree == len(OTHERS))
    uni_fn = conf & Yb & (n_agree == len(OTHERS))
    solo = conf & (n_agree == 0)
    print(f"[집합] 전원 동조 FP {int(uni_fp.sum()):,} · 전원 동조 FN {int(uni_fn.sum()):,}"
          f" · 앵커 단독 {int(solo.sum()):,} (확신 오답 {int(conf.sum()):,} 중)")

    # ── A. 5/5 확신 FP — 클래스 쌍 ──────────────────────────────────────────
    rows, cols = np.where(uni_fp)
    pairs = Counter()
    pair_docs = defaultdict(list)
    for i, c in zip(rows, cols):
        g = max(gold_rows[i], key=lambda x: lift[x, c])      # 가장 강하게 붙는 정답 라벨에 귀속
        pairs[(int(g), int(c))] += 1
        pair_docs[(int(g), int(c))].append(int(i))
    tot_fp = sum(pairs.values())
    top_pairs = []
    for (g, c), n in pairs.most_common(TOP_N):
        rev = pairs.get((c, g), 0)
        top_pairs.append({
            "gold": mno[g], "predicted": mno[c], "n": n,
            "lift": round(float(lift[g, c]), 2),
            "same_lno": bool(ls.lno_idx[g] == ls.lno_idx[c]),
            "cooccur_docs_train": int(co[g, c]),
            "reverse_n": rev,
            "gold_in_conflict_groups": involved.get(g, 0),
            "pred_in_conflict_groups": involved.get(c, 0),
        })
    fp_summary = {
        "n_elements": tot_fp, "n_distinct_pairs": len(pairs),
        "top10_share": round(sum(n for _, n in pairs.most_common(10)) / tot_fp, 4),
        "same_lno_share": round(float(np.mean([ls.lno_idx[g] == ls.lno_idx[c]
                                               for (g, c), n in pairs.items()
                                               for _ in range(n)])), 4),
        "top_pairs": top_pairs,
    }
    print(f"\nA. 5/5 확신 FP — {tot_fp:,}개가 {len(pairs)}개 쌍에 분포"
          f"(상위 10쌍 {fp_summary['top10_share']:.1%} · 같은 Lno {fp_summary['same_lno_share']:.1%})")
    print(f"  {'정답→예측':<16}{'n':>4}{'lift':>8}{'train 공기':>10}{'역방향':>7}{'Lno':>6}{'충돌그룹':>8}")
    for r in top_pairs[:10]:
        print(f"  {r['gold']}→{r['predicted']:<10}{r['n']:>4}{r['lift']:>8.1f}"
              f"{r['cooccur_docs_train']:>10,}{r['reverse_n']:>7}"
              f"{'같음' if r['same_lno'] else '다름':>6}"
              f"{r['gold_in_conflict_groups']:>4}/{r['pred_in_conflict_groups']:<3}")

    # ── B. 5/5 확신 FN — 클래스 ────────────────────────────────────────────
    rows, cols = np.where(uni_fn)
    fn_by_label = Counter(int(c) for c in cols)
    fn_docs = defaultdict(list)
    fn_lifts = defaultdict(list)
    for i, c in zip(rows, cols):
        fn_docs[int(c)].append(int(i))
        sib = [g for g in gold_rows[i] if g != c]
        fn_lifts[int(c)].append(max([lift[g, c] for g in sib], default=float("nan")))
    pred_all = pa >= TAU
    recall = {c: float((pred_all[:, c] & Yb[:, c]).sum() / max(int(Yb[:, c].sum()), 1))
              for c in fn_by_label}
    top_fn = []
    for c, n in fn_by_label.most_common(TOP_N):
        lv = [v for v in fn_lifts[c] if v == v]
        top_fn.append({
            "label": mno[c], "n": n,
            "test_support": int(Yb[:, c].sum()),
            "share_of_class_positives": round(n / max(int(Yb[:, c].sum()), 1), 4),
            "test_recall": round(recall[c], 4),
            "train_support": int(cnt_tr[c]),
            "median_lift_to_sibling_gold": round(float(np.median(lv)), 2) if lv else None,
            "n_k1_docs": sum(1 for i in fn_docs[c] if len(gold_rows[i]) == 1),
            "in_conflict_groups": involved.get(c, 0),
        })
    fn_summary = {
        "n_elements": int(uni_fn.sum()), "n_distinct_labels": len(fn_by_label),
        "top10_share": round(sum(n for _, n in fn_by_label.most_common(10)) / int(uni_fn.sum()), 4),
        "top_labels": top_fn,
    }
    print(f"\nB. 5/5 확신 FN — {int(uni_fn.sum()):,}개가 {len(fn_by_label)}개 클래스에 분포"
          f"(상위 10클래스 {fn_summary['top10_share']:.1%})")
    print(f"  {'라벨':<8}{'n':>4}{'test 양성':>9}{'그 중':>7}{'recall':>8}{'형제 lift':>10}{'k=1':>5}{'충돌':>5}")
    for r in top_fn[:10]:
        ml = f"{r['median_lift_to_sibling_gold']:.2f}" if r["median_lift_to_sibling_gold"] is not None else "—"
        print(f"  {r['label']:<8}{r['n']:>4}{r['test_support']:>9}{r['share_of_class_positives']:>7.1%}"
              f"{r['test_recall']:>8.3f}{ml:>10}{r['n_k1_docs']:>5}{r['in_conflict_groups']:>5}")

    # ── C. 0/5 앵커 단독 — 훈련 잡음인가 ───────────────────────────────────
    rows, cols = np.where(solo)
    twin_p = P[TWIN][rows, cols]
    anchor_p = pa[rows, cols]
    is_pos = Yb[rows, cols]
    solo_rec = {
        "n": int(solo.sum()),
        "fp": int((~is_pos).sum()), "fn": int(is_pos.sum()),
        "twin_correct_share": round(float(np.where(is_pos, twin_p >= TAU, twin_p < TAU).mean()), 4),
        "twin_confident_correct_share": round(float(np.where(is_pos, twin_p >= CONF_HI,
                                                             twin_p < CONF_LO).mean()), 4),
        "median_anchor_p": round(float(np.median(anchor_p)), 4),
        "median_twin_p": round(float(np.median(twin_p)), 4),
        "mass_share_of_confident": None,      # 아래에서 채운다
        "examples": [{"document_id": ds["document_id"][int(i)], "label": mno[int(c)],
                      "gold": bool(Yb[i, c]), "anchor_p": round(float(pa[i, c]), 4),
                      **{m: round(float(P[m][i, c]), 4) for m in OTHERS}}
                     for i, c in zip(rows[:10], cols[:10])],
        "note": "앵커만 확신을 갖고 틀린 원소. 시드 쌍둥이(11_04)가 맞히면 재현되지 않는 "
                "개별 런의 흔들림이며, 모델 계열의 성질이 아니다.",
    }
    ptt = np.where(Yb, pa, 1 - pa)
    fl = 0.25 * (1 - ptt) ** 2 * -np.log(np.clip(ptt, 1e-12, 1))
    solo_rec["mass_share_of_confident"] = round(float(fl[solo].sum() / fl[conf].sum()), 4)
    print(f"\nC. 0/5 앵커 단독 {solo_rec['n']}개(FP {solo_rec['fp']} · FN {solo_rec['fn']})"
          f" — 확신 오답 손실 질량의 {solo_rec['mass_share_of_confident']:.1%}")
    print(f"  시드 쌍둥이가 맞히는 비율 {solo_rec['twin_correct_share']:.1%}"
          f"(확신까지 {solo_rec['twin_confident_correct_share']:.1%})"
          f" · 중앙 확률 앵커 {solo_rec['median_anchor_p']:.3f} 대 쌍둥이 {solo_rec['median_twin_p']:.3f}")

    # ── D. 클리닝 연루 검정 ────────────────────────────────────────────────
    conf_lbl = set(involved)
    base = float(Yb[:, sorted(conf_lbl)].sum() / Yb.sum())          # test 양성 중 연루 클래스 몫
    uni_fn_lbl = [int(c) for c in np.where(uni_fn)[1]]
    uni_fp_lbl = [int(c) for c in np.where(uni_fp)[1]]
    obs_fn = float(np.mean([c in conf_lbl for c in uni_fn_lbl]))
    obs_fp = float(np.mean([c in conf_lbl for c in uni_fp_lbl]))
    ea = ls.mno_of_col.index("EA04") if "EA04" in ls.mno_of_col else None
    ea04 = {
        "in_conflict_groups": involved.get(ea, 0),
        "as_disputed_label": disputed.get(ea, 0),
        "groups": [groups[g] for g in by_label.get(ea, [])],
        "uni_fn_elements": int(sum(1 for c in uni_fn_lbl if c == ea)),
        "test_support": int(Yb[:, ea].sum()), "train_support": int(cnt_tr[ea]),
        "test_recall": round(float((pred_all[:, ea] & Yb[:, ea]).sum()
                                   / max(int(Yb[:, ea].sum()), 1)), 4),
    } if ea is not None else None
    clean_rec = {
        "n_labels_in_conflict_groups": len(conf_lbl),
        "base_rate_test_positives": round(base, 4),
        "share_uni_fn_in_conflict_labels": round(obs_fn, 4),
        "enrichment_fn": round(obs_fn / base, 2) if base else None,
        "share_uni_fp_pred_in_conflict_labels": round(obs_fp, 4),
        "enrichment_fp": round(obs_fp / base, 2) if base else None,
        "EA04": ea04,
        "note": "[ADR-0010]의 제거는 **입력이 완전히 동일한** 케이스만 잡은 하한이다. 같은 "
                "클래스의 잔여 충돌이 test에 남아 있다면 확신 오답이 그 클래스에 몰려야 한다.",
    }
    print(f"\nD. 클리닝 연루 검정 — 충돌 42그룹 연루 라벨 {len(conf_lbl)}개"
          f"(test 양성의 {base:.1%})")
    print(f"  5/5 확신 FN 중 연루 클래스 {obs_fn:.1%}(관측/기대 {clean_rec['enrichment_fn']:.2f})"
          f" · 5/5 확신 FP 예측 라벨 중 {obs_fp:.1%}({clean_rec['enrichment_fp']:.2f})")
    if ea04:
        print(f"  EA04 — 충돌 그룹 연루 {ea04['in_conflict_groups']}회"
              f"(그 중 갈린 라벨로 {ea04['as_disputed_label']}회)"
              f" · 5/5 확신 FN {ea04['uni_fn_elements']}개"
              f" · test 양성 {ea04['test_support']} · recall {ea04['test_recall']:.3f}")
        for grp in ea04["groups"]:
            print(f"    그룹: {grp}")

    # ── E. 재라벨링 후보 목록 ──────────────────────────────────────────────
    cand = []
    for i, c in zip(*np.where(uni_fp)):
        g = max(gold_rows[i], key=lambda x: lift[x, c])
        cand.append({
            "type": "missing_label(누락 후보)", "document_id": ds["document_id"][int(i)],
            "invention_title": "", "ipc_main": "",   # 원문 필드는 동봉하지 않는다(데이터셋 재생성 시 채워진다)
            "gold_mno": " ".join(mno[x] for x in gold_rows[i]),
            "candidate_mno": mno[int(c)], "anchor_p": round(float(pa[i, c]), 3),
            # 합의의 최약 고리 — FP는 가장 낮은 확률(가장 소극적인 런)
            "weakest_model_p": round(float(min(P[m][i, c] for m in MODELS)), 3),
            "lift_to_gold": round(float(lift[g, c]), 2),
            "same_lno": int(ls.lno_idx[g] == ls.lno_idx[int(c)]),
        })
    for i, c in zip(*np.where(uni_fn)):
        sib = [g for g in gold_rows[i] if g != c]
        cand.append({
            "type": "spurious_label(오부착 후보)", "document_id": ds["document_id"][int(i)],
            "invention_title": "", "ipc_main": "",   # 원문 필드는 동봉하지 않는다(데이터셋 재생성 시 채워진다)
            "gold_mno": " ".join(mno[x] for x in gold_rows[i]),
            "candidate_mno": mno[int(c)], "anchor_p": round(float(pa[i, c]), 3),
            # FN은 가장 높은 확률(가장 적극적인 런) — 어느 쪽이든 합의를 가장 약하게 지지하는 값
            "weakest_model_p": round(float(max(P[m][i, c] for m in MODELS)), 3),
            "lift_to_gold": round(float(max([lift[g, c] for g in sib], default=float("nan"))), 2),
            "same_lno": "",
        })
    # 검토 순서 = 신호가 강한 순. 누락 후보는 lift 높은 순, 오부착 후보는 낮은 순(NaN = k=1 문서는 뒤로)
    def sort_key(r):
        v = r["lift_to_gold"]
        if r["type"].startswith("missing"):
            return (0, -(v if v == v else -1))
        return (1, (v if v == v else float("inf")))
    cand.sort(key=sort_key)

    csv_path = OUT / "relabel_candidates.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cand[0]))
        w.writeheader()
        w.writerows(cand)
    print(f"\nE. 재라벨링 후보 {len(cand):,}행 → {csv_path.name}"
          f"(누락 {int(uni_fp.sum()):,} · 오부착 {int(uni_fn.sum()):,},"
          f" 문서 {len({r['document_id'] for r in cand}):,}건)")

    payload = {
        "question": "확신 오답의 두 극단(전원 동조 5/5 · 앵커 단독 0/5)은 클래스 축에서 어떤 모양인가.",
        "axis": {"split": "test", "n_docs": len(Y), "n_labels": NUM_LABELS,
                 "confident_band": [CONF_LO, CONF_HI], "tau": TAU,
                 "models": MODELS, "anchor": ANCHOR},
        "script": "scripts/confident_error_classes.py",
        "pair_attribution": "확신 FP의 (정답, 예측) 쌍은 문서의 정답 라벨 중 lift가 최대인 것에 귀속한다.",
        "universal_fp_pairs": fp_summary,
        "universal_fn_labels": fn_summary,
        "anchor_solo": solo_rec,
        "cleaning_overlap": clean_rec,
        "relabel_candidates": {"path": "output/relabel_candidates.csv", "n_rows": len(cand),
                               "n_docs": len({r["document_id"] for r in cand})},
    }
    path = OUT / "confident_error_classes.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
