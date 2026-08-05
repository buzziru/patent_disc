"""다중레이블 문서의 계층 형상 — `Lno`가 갈리는가, 같은 `Lno` 안에서 갈리는가.

계층 구조를 설계한다면 두 단계 각각을 단일 선택으로 둘 수 있는지가 먼저 정해져야 한다.
그 답은 모델이 아니라 **라벨 분포**에 있다. 여기서 재는 것은 세 가지다.

  (1) 형상 분해 — k>=2 문서를 순수 cross-`Lno`(`Lno`당 `Mno` 1개) / 순수 within-`Lno`
      (`Lno` 1개, 형제 복수) / 혼합으로 가른다. 각 설계 제약이 표현하지 못하는 문서를 센다.
  (2) 설계 상한 — 1단계를 단일 `Lno`로 두거나 2단계를 `Lno`당 단일 `Mno`로 둘 때
      완벽 예측을 가정해도 남는 양성 라벨 비율(recall 상한).
  (3) 결손의 소재 — test 양성 라벨을 '같은 문서에 같은 `Lno` 형제가 있는 라벨'과
      '그 `Lno`에 혼자인 라벨'로 갈라 recall을 재, 다중레이블 결손이 형제 쪽인지
      cross-`Lno` 쪽인지 본다.

형상 통계는 정리본 전 split(223,992문서), recall 분해는 test(11,244) 로짓으로 낸다.

실행: `uv run python scripts/multilabel_shape.py`
산출: `output/multilabel_shape.json`

참조하는 HF 데이터셋은 공개 배포하지 않는다 — 재생성 절차는
`docs/data/data-pipeline.md`「가공 데이터셋은 배포하지 않는다 — 재현 경로」.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
ROOT = Path(os.environ["DATA_ROOT"])
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset                      # noqa: E402
from huggingface_hub import hf_hub_download            # noqa: E402
from error_analysis import LabelSpace, build_gold      # noqa: E402

OUT = ROOT / "output"
RAW_DS = "ingyoun/patent-clean-text"
SPLITS = ["train", "val", "test"]
NUM_LABELS = 188

MODELS = {                                  # recall 분해 대상(정리 축에 얹을 수 있는 로짓)
    "exp1(8192)": "modernbert-patent-len8192",
    "11_01(512)": "modernbert-patent-len512-b128",
}


def shape(recs, ls):
    """문서별 라벨 목록 → 계층 형상 통계."""
    n = len(recs)
    k = np.array([len(r) for r in recs])
    n_lno = np.array([len(set(ls.lno_idx[r])) for r in recs])
    max_per_lno = np.array([max(Counter(ls.lno_idx[r]).values()) for r in recs])
    multi = k >= 2

    pure_cross = multi & (max_per_lno == 1)      # Lno 여러 개, Lno당 Mno 1개
    pure_within = multi & (n_lno == 1)           # Lno 1개, 형제 복수
    mixed = multi & (max_per_lno >= 2) & (n_lno >= 2)
    need_multi_mno = pure_within | mixed         # 'Lno당 단일 Mno' 설계로 표현 불가
    assert int((pure_cross | pure_within | mixed).sum()) == int(multi.sum())

    dist = lambda a: {int(v): int(c) for v, c in zip(*np.unique(a, return_counts=True))}
    return {
        "n_docs": n, "n_positive_labels": int(k.sum()),
        "multi_rate": float(multi.mean()), "n_multi": int(multi.sum()),
        "k_dist": dist(k),
        "multi_n_lno_dist": dist(n_lno[multi]),
        "multi_max_mno_per_lno_dist": dist(max_per_lno[multi]),
        "pure_cross_lno": int(pure_cross.sum()),
        "pure_within_lno": int(pure_within.sum()),
        "mixed": int(mixed.sum()),
        "need_multi_mno_per_lno": int(need_multi_mno.sum()),
        "need_multi_mno_share_of_multi": float(need_multi_mno.sum() / multi.sum()),
        "need_multi_mno_share_of_all": float(need_multi_mno.mean()),
        # Lno당 1개만 허용하면 남는 양성 라벨 = 문서별 서로 다른 Lno 수의 합
        "recall_ceiling_one_mno_per_lno": float(n_lno.sum() / k.sum()),
    }


def cooccurrence(recs, ls):
    """문서 안에서 함께 오는 Lno 쌍 · 같은 Lno 안의 형제 Mno 쌍 · Lno별 형제 복수 비율."""
    mno_of = np.array(ls.mno_of_col)
    lno_pairs, sib_pairs, sib_docs, lno_docs = Counter(), Counter(), Counter(), Counter()
    pair_mno_count = Counter()
    for r in recs:
        ids = sorted(set(ls.lno_idx[r]))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                lno_pairs[f"{ls.lnos[ids[i]]}+{ls.lnos[ids[j]]}"] += 1
        by_lno = {}
        for c in r:
            by_lno.setdefault(ls.lno_idx[c], []).append(c)
        for l, cs in by_lno.items():
            lno_docs[ls.lnos[l]] += 1
            pair_mno_count[len(cs)] += 1
            if len(cs) >= 2:
                sib_docs[ls.lnos[l]] += 1
                cs = sorted(cs)
                for i in range(len(cs)):
                    for j in range(i + 1, len(cs)):
                        sib_pairs[f"{mno_of[cs[i]]}+{mno_of[cs[j]]}"] += 1
    n_pairs = sum(pair_mno_count.values())
    return {
        "n_doc_lno_pairs": n_pairs,
        "mno_per_doc_lno_dist": {int(k): {"n": v, "share": round(v / n_pairs, 6)}
                                 for k, v in sorted(pair_mno_count.items())},
        "n_lno_pair_types": len(lno_pairs), "n_sibling_pair_types": len(sib_pairs),
        "top_lno_pairs": lno_pairs.most_common(10),
        "top_sibling_pairs": sib_pairs.most_common(10),
        "sibling_top10_share": round(sum(c for _, c in sib_pairs.most_common(10))
                                     / max(sum(sib_pairs.values()), 1), 4),
        "sibling_rate_by_lno": {l: round(sib_docs[l] / lno_docs[l], 4)
                                for l in sorted(lno_docs, key=lambda x: -sib_docs[x] / lno_docs[x])},
    }


def recall_split(ls):
    """test 양성 라벨을 형제 동반 / 단독-Lno로 갈라 recall을 잰다 + 유도 다중 Lno 게이트 상한."""
    ds = load_dataset(RAW_DS, split="test")
    n = len(ds)
    Y = build_gold(ds["label_ids"], n, NUM_LABELS)
    YL = ls.to_lno(Y)
    old_ids = json.loads((OUT / "doc_ids_test.json").read_text(encoding="utf-8"))
    pos = {d: i for i, d in enumerate(old_ids)}
    keep = np.array([pos[d] for d in ds["document_id"]])

    sib_lab = np.zeros((n, NUM_LABELS), dtype=bool)   # 같은 문서에 같은 Lno 형제가 있는 양성 라벨
    for i, r in enumerate(ds["label_ids"]):
        cnt = Counter(ls.lno_idx[list(r)])
        for c in r:
            if cnt[ls.lno_idx[c]] >= 2:
                sib_lab[i, c] = True
    solo = Y & ~sib_lab
    k = Y.sum(1)
    multi = k >= 2

    out = {"n_positive": int(Y.sum()), "n_sibling_accompanied": int(sib_lab.sum()),
           "n_solo_lno": int(solo.sum()), "n_solo_in_multi": int(solo[multi].sum()),
           "models": {}}
    for name, tag in MODELS.items():
        z = np.load(OUT / f"logits_{tag}_test.npy")
        if z.shape[0] == len(old_ids):
            z = z[keep]
        pred = z >= 0                                  # τ=0.5 ⟺ logit >= 0
        fn = Y & ~pred
        predL = ls.to_lno(pred)                        # 유도 Lno 다중레이블 예측
        empty = predL.sum(1) == 0
        predL[empty, ls.lno_idx[z.argmax(1)][empty]] = True     # 빈 예측은 top-1 Lno로 백오프
        gate = predL[:, ls.lno_idx]
        out["models"][name] = {
            "recall_all": float((pred & Y).sum() / Y.sum()),
            "recall_sibling_accompanied": float((pred & sib_lab).sum() / sib_lab.sum()),
            "recall_solo_lno": float((pred & solo).sum() / solo.sum()),
            "recall_solo_in_k1": float((pred[~multi] & solo[~multi]).sum() / solo[~multi].sum()),
            "recall_solo_in_multi": float((pred[multi] & solo[multi]).sum() / solo[multi].sum()),
            "fn_total": int(fn.sum()),
            "fn_sibling_accompanied": int((fn & sib_lab).sum()),
            "fn_solo_in_multi": int((fn[multi] & solo[multi]).sum()),
            "fn_solo_in_k1": int((fn[~multi] & solo[~multi]).sum()),
            "induced_multi_lno_gate": {
                "mean_lno_per_doc": float(predL.sum(1).mean()),
                "column_share": float(gate.mean()),
                "recall_ceiling": float((Y & gate).sum() / Y.sum()),
                "lno_level_recall": float((YL & predL).sum() / YL.sum()),
            },
        }
    return out


def main():
    lm = json.load(open(hf_hub_download(RAW_DS, "label_mappings.json", repo_type="dataset"),
                        encoding="utf-8"))
    ls = LabelSpace(lm["id2mno"], lm["mno2lno"], NUM_LABELS)

    per_split = {s: [list(x) for x in load_dataset(RAW_DS, split=s)["label_ids"]] for s in SPLITS}
    per_split["ALL"] = [r for s in SPLITS for r in per_split[s]]

    shapes = {s: shape(recs, ls) for s, recs in per_split.items()}
    co = cooccurrence(per_split["ALL"], ls)
    rec = recall_split(ls)

    a = shapes["ALL"]
    print(f"[형상] 문서 {a['n_docs']:,} · 양성 라벨 {a['n_positive_labels']:,}"
          f" · k>=2 {a['multi_rate']:.2%} ({a['n_multi']:,})")
    print(f"  k>=2의 Lno 개수 분포 {a['multi_n_lno_dist']}")
    print(f"  k>=2의 Lno당 최대 Mno 분포 {a['multi_max_mno_per_lno_dist']}")
    for lab, key in [("순수 cross-Lno (Lno당 Mno 1개)", "pure_cross_lno"),
                     ("순수 within-Lno (Lno 1개·형제 복수)", "pure_within_lno"),
                     ("혼합", "mixed"), ("→ Lno당 단일 Mno로 표현 불가", "need_multi_mno_per_lno")]:
        v = a[key]
        print(f"    {lab:<34}{v:>8,}  k>=2의 {v/a['n_multi']:>6.1%}  전체의 {v/a['n_docs']:>6.2%}")
    print(f"  recall 상한 — 'Lno당 단일 Mno' 제약 {a['recall_ceiling_one_mno_per_lno']:.4f}")
    print(f"  split별 k>=2 " + " · ".join(f"{s} {shapes[s]['multi_rate']:.2%}" for s in SPLITS))

    print(f"\n[동시 출현] (문서,Lno) 쌍 {co['n_doc_lno_pairs']:,} · 그 안의 Mno 개수 "
          + ", ".join(f"{k}개 {v['share']:.2%}" for k, v in co["mno_per_doc_lno_dist"].items()))
    print("  상위 Lno 쌍: " + ", ".join(f"{p} {c:,}" for p, c in co["top_lno_pairs"][:6]))
    print("  상위 형제 Mno 쌍: " + ", ".join(f"{p} {c:,}" for p, c in co["top_sibling_pairs"][:6])
          + f" (쌍 {co['n_sibling_pair_types']:,}종·상위10 {co['sibling_top10_share']:.1%})")
    print("  Lno별 형제 복수 비율 상위: "
          + ", ".join(f"{l} {v:.1%}" for l, v in list(co["sibling_rate_by_lno"].items())[:6]))

    print(f"\n[결손의 소재] test 양성 {rec['n_positive']:,} = 형제 동반 {rec['n_sibling_accompanied']:,}"
          f" + 단독-Lno {rec['n_solo_lno']:,}(그중 k>=2 문서 {rec['n_solo_in_multi']:,})")
    for name, r in rec["models"].items():
        g = r["induced_multi_lno_gate"]
        print(f"  {name}: recall 전체 {r['recall_all']:.4f} || k=1 {r['recall_solo_in_k1']:.4f}"
              f" · k>=2 형제 동반 {r['recall_sibling_accompanied']:.4f}"
              f" · k>=2 단독-Lno {r['recall_solo_in_multi']:.4f}")
        print(f"      FN {r['fn_total']:,} = 형제 {r['fn_sibling_accompanied']:,}"
              f" + k>=2 단독-Lno {r['fn_solo_in_multi']:,} + k=1 {r['fn_solo_in_k1']:,}")
        print(f"      유도 다중 Lno 게이트: 평균 {g['mean_lno_per_doc']:.2f}개"
              f" · 열 {g['column_share']:.1%} 잔존 · recall 상한 {g['recall_ceiling']:.4f}")

    (OUT / "multilabel_shape.json").write_text(json.dumps(
        {"raw_ds": RAW_DS, "num_labels": NUM_LABELS, "splits": SPLITS,
         "shape": shapes, "cooccurrence": co, "recall_decomposition": rec},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT / 'multilabel_shape.json'}")


if __name__ == "__main__":
    main()
