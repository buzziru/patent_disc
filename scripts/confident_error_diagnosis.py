"""확신 오답 진단 — 손실 질량의 73~82%를 무는 원소 1,500~2,000개의 정체.

`docs/experiments/training-curves.md`가 val loss 상승의 정체를 **소수 확신 오답으로의 질량
집중**으로 규정했다(양성인데 p<0.1 · 음성인데 p>0.9). 이 스크립트는 그 집합이
**라벨 누락**인지 **본질적 모호성**인지 **모델 고유의 실패**인지를 가른다. 성능 레버가 아니라
산출물의 한계 기술을 정하는 진단이다(`NEXT_SESSION.md` 「확신 오답 진단」).

다섯 갈래를 잰다. 전부 덤프된 로짓과 train 라벨만 쓴다 — GPU 0.

  A. 교차 모델 합의   앵커의 확신 오답을 다른 5런이 **같은 문서·같은 라벨에서 같은 방향으로**
                     틀리는가. 시드 쌍둥이(`11_04`) → 같은 아키텍처 다른 길이·손실 →
                     다른 아키텍처·토크나이저(KoBERT)로 사다리를 놓아 합의의 독립성을 잰다.
                     대조군은 같은 앵커의 **경계 오답**(틀렸으나 확신 없음)이다 — 모델들이
                     원래 서로 닮았기 때문에, 확신 오답의 합의율은 이 대조 위에서만 읽힌다.
  B. 공기 검정        확신 FP의 (정답 라벨, 예측 라벨) 쌍이 train 코퍼스에서 우연 이상으로
                     함께 붙는가(lift). 붙으면 그 예측은 오답이 아니라 **누락된 정답**일
                     개연성이 있다. 바닥은 무작위 비정답 라벨, 천장은 실제 정답끼리의 공기다.
                     확신 FN은 방향을 뒤집어 잰다 — 그 정답 라벨이 같은 문서의 다른 정답과
                     안 붙으면 **오부착** 후보다.
  C. 위치 조건화      확신 오답이 정답 `Lno` 안(형제)인가 밖인가 · k · 길이 bin.
  D. focal 귀속       "focal이 hard sample에 확신을 키운다"는 해석의 직접 검정 —
                     같은 레시피(eff128/lr4.8e-4)에서 γ만 뺀 BCE 런과 확신 오답 규모·집합을
                     대조한다. 집합이 크게 겹치면 확신 오답은 손실이 만든 것이 아니라
                     데이터가 정한 것이다.
  E. 종합 분류        A×B를 교차해 확신 오답을 세 갈래로 나누고 각 갈래가 문 손실 질량을 낸다.

평가 축은 정리 test(11,244). 구 test(11,271) 축의 로짓은 `doc_ids_test.json`으로 사영한다
(`scripts/loss_mass_decomposition.py`와 같은 절차).

실행: `uv run python scripts/confident_error_diagnosis.py`
산출: `output/confident_error_diagnosis.json`

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

from datasets import load_dataset                      # noqa: E402  (HF_HOME 설정 뒤 import)
from huggingface_hub import hf_hub_download            # noqa: E402
from error_analysis import LabelSpace, build_gold      # noqa: E402

OUT = ROOT / "output"
RAW_DS = "ingyoun/patent-clean-text"
SPLIT = "test"
NUM_LABELS = 188
ALPHA, GAMMA = 0.25, 2                  # 프로젝트 확정 focal 하이퍼파라미터
CONF_LO, CONF_HI = 0.1, 0.9             # 확신 오답 정의(training-curves.md와 동일)
TAU = 0.5

ANCHOR = "11_01"
MODELS = {                              # 앵커 + 합의 사다리 5종
    "11_01": "modernbert-patent-len512-b128",     # 현행 앵커(정리 데이터·신 레시피)
    "11_04": "modernbert-patent-seed153",         # 시드 쌍둥이 — 같은 모든 것, 시드만 다름
    "exp2": "modernbert-patent-len512",           # 같은 아키텍처, 구 레시피·구 데이터
    "exp1": "modernbert-patent-len8192",          # 같은 아키텍처, 16배 컨텍스트
    "ASL": "modernbert-patent-len512-asl",        # 다른 손실(확률 스케일이 다름 — 결정축으로만 읽는다)
    "KoBERT": "kobert-patent-baseline_len512",    # 다른 아키텍처·토크나이저
}
OTHERS = [m for m in MODELS if m != ANCHOR]
BCE_TAG = "modernbert-patent-len512-bce"          # D. focal 귀속 대조군(γ만 제거, 같은 레시피)
RNG = np.random.default_rng(0)


# ── 축 · 로짓 ────────────────────────────────────────────────────────────────
def load_axis():
    ds = load_dataset(RAW_DS, split=SPLIT)
    Y = build_gold(ds["label_ids"], len(ds), NUM_LABELS)
    clean_ids = list(ds["document_id"])
    stored = json.loads((OUT / f"doc_ids_clean_{SPLIT}.json").read_text(encoding="utf-8"))
    assert stored == clean_ids, "doc_ids_clean이 데이터셋 순서와 다르다(정리 축 로짓의 행 축)"
    old_ids = json.loads((OUT / f"doc_ids_{SPLIT}.json").read_text(encoding="utf-8"))
    pos = {d: i for i, d in enumerate(old_ids)}
    keep = np.array([pos[d] for d in clean_ids])
    assert np.all(np.diff(keep) > 0), "사영 인덱스가 단조 증가가 아니다(행 순서 불일치)"
    return Y, np.array(ds["length_bin"]), clean_ids, keep, len(old_ids)


def load_prob(tag, keep, n_old, n_clean):
    z = np.load(OUT / f"logits_{tag}_{SPLIT}.npy").astype(np.float64)
    if z.shape[0] == n_old:
        z = z[keep]
    assert z.shape == (n_clean, NUM_LABELS), (tag, z.shape)
    return 1.0 / (1.0 + np.exp(-z))


def wrong_at_tau(p, Y):
    """τ=0.5에서 틀린 원소 — 스케일이 다른 손실(ASL)에도 성립하는 결정축 정의."""
    return np.where(Y, p < TAU, p >= TAU)


def confidently_wrong(p, Y):
    """확신 오답 — 양성인데 p<0.1 · 음성인데 p>0.9."""
    return np.where(Y, p < CONF_LO, p > CONF_HI)


def focal_elements(p, Y):
    """원소별 focal 손실(훈련 손실과 같은 정의, 리덕션 전)."""
    pt = np.where(Y, p, 1 - p)
    return ALPHA * (1 - pt) ** GAMMA * -np.log(np.clip(pt, 1e-12, 1))


# ── A. 교차 모델 합의 ────────────────────────────────────────────────────────
def consensus(P, Y):
    """앵커 확신 오답에 대한 다른 런들의 동조율 — 대조군은 같은 앵커의 경계 오답."""
    pa = P[ANCHOR]
    conf = confidently_wrong(pa, Y)
    err = wrong_at_tau(pa, Y)
    border = err & ~conf                                # 틀렸으나 확신 없음 = 대조군
    correct = ~err

    wrong_o = {m: wrong_at_tau(P[m], Y) for m in OTHERS}
    conf_o = {m: confidently_wrong(P[m], Y) for m in OTHERS}
    n_agree = np.sum([wrong_o[m] for m in OTHERS], axis=0)     # 0~5

    def profile(mask):
        n = int(mask.sum())
        return {
            "n": n,
            "per_model_agree": {m: round(float(wrong_o[m][mask].mean()), 4) for m in OTHERS},
            "per_model_agree_confident": {m: round(float(conf_o[m][mask].mean()), 4)
                                          for m in OTHERS},
            "n_agree_hist": {str(k): int((n_agree[mask] == k).sum()) for k in range(len(OTHERS) + 1)},
            "mean_n_agree": round(float(n_agree[mask].mean()), 3),
            "unique_share": round(float((n_agree[mask] == 0).mean()), 4),
            "universal_share": round(float((n_agree[mask] == len(OTHERS)).mean()), 4),
        }

    Yb = Y.astype(bool)
    return {
        "definition": f"확신 오답 = 양성 p<{CONF_LO} · 음성 p>{CONF_HI}. "
                      f"동조 = τ={TAU}에서 같은 원소를 같은 방향으로 틀림(결정축 — ASL 스케일 무관).",
        "anchor": ANCHOR,
        "confident_errors": profile(conf),
        "confident_fp": profile(conf & ~Yb),
        "confident_fn": profile(conf & Yb),
        "borderline_errors": profile(border),
        "reference_correct_elements": {
            "n": int(correct.sum()),
            "per_model_agree": {m: round(float(wrong_o[m][correct].mean()), 5) for m in OTHERS},
            "note": "앵커가 맞힌 원소에서 다른 런이 틀린 비율 — 원소 축의 기저 불일치율.",
        },
        "ladder_note": "11_04(시드만 다름) → exp2·exp1(같은 아키텍처) → ASL(다른 손실) → "
                       "KoBERT(다른 아키텍처·토크나이저). 사다리를 내려가도 동조율이 유지되면 "
                       "그 오답은 모델이 아니라 데이터가 정한 것이다.",
    }, conf, border, n_agree


# ── B. 공기 검정 ─────────────────────────────────────────────────────────────
def train_cooccurrence():
    """train 201,616문서의 라벨 공기 통계 — lift와 실현된 라벨 조합 목록."""
    ds = load_dataset(RAW_DS, split="train")
    ids_list = ds["label_ids"]
    n = len(ids_list)
    cnt = np.zeros(NUM_LABELS)
    co = np.zeros((NUM_LABELS, NUM_LABELS))
    combos = Counter()
    for ids in ids_list:
        cnt[ids] += 1
        for a in ids:
            co[a, ids] += 1
        combos[frozenset(ids)] += 1
    lift = co * n / np.outer(cnt, cnt)                  # 1.0 = 우연 수준
    np.fill_diagonal(lift, 1.0)
    return lift, co, cnt, n, set(combos)


def max_lift(lift, cols, target):
    """target 라벨과 cols(문서의 다른 라벨) 사이 lift의 최댓값."""
    return float(lift[cols, target].max()) if len(cols) else 0.0


def known_noise_reference(lift, groups):
    """입력이 완전히 동일한데 라벨이 갈린 42개 그룹(ADR-0010) — 사람 라벨링 불일치의 실측 표본.

    같은 문서에 어떤 주석자는 붙이고 어떤 주석자는 안 붙인 라벨이 **공유 라벨과 얼마나 붙는지**가
    "누락된 정답"의 lift 분포다. 확신 FP의 분포를 이것과 대면시키면 공기 검정의 판정선이
    임의값이 아니라 실측 잡음 표본이 된다.
    """
    vals = []
    for grp in groups:
        sets = [set(json.loads(e.split(":", 2)[2])) for e in grp]
        shared = set.intersection(*sets)
        for c in set().union(*sets) - shared:
            base = shared or (set().union(*sets) - {c})
            if base:
                vals.append(max(float(lift[b, c]) for b in base))
    return np.array(vals)


def cooccurrence_test(P, Y, lift, co, combos, conf, n_agree, gold_rows):
    """확신 FP는 '누락된 정답'인가 — lift 분포를 바닥·천장 사이에 놓고 읽는다."""
    pa = P[ANCHOR]
    fp_all = (~Y.astype(bool)) & (pa >= TAU)
    conf_fp = conf & ~Y.astype(bool)
    conf_fn = conf & Y.astype(bool)

    def lifts(mask):
        rows, cols = np.where(mask)
        return np.array([max_lift(lift, gold_rows[i], c) for i, c in zip(rows, cols)]), rows, cols

    l_conf_fp, rows_cfp, cols_cfp = lifts(conf_fp)
    l_border_fp, _, _ = lifts(fp_all & ~conf)

    # 바닥 — 같은 문서에서 무작위로 뽑은 비정답·비예측 라벨
    floor = []
    for i in np.unique(rows_cfp):
        pool = np.where(~Y[i].astype(bool) & (pa[i] < TAU))[0]
        for c in RNG.choice(pool, size=min(5, len(pool)), replace=False):
            floor.append(max_lift(lift, gold_rows[i], int(c)))
    floor = np.array(floor)

    # 천장 — 실제 정답끼리의 공기(k>=2 문서)
    ceil = []
    for i in np.where(Y.sum(1) >= 2)[0]:
        g = gold_rows[i]
        for c in g:
            ceil.append(max_lift(lift, g[g != c], int(c)))
    ceil = np.array(ceil)

    thr = float(np.quantile(floor, 0.90)) if len(floor) else 1.0   # 바닥 90분위 = 판정선

    def dist(v, name):
        return {"set": name, "n": int(v.size),
                "median_lift": round(float(np.median(v)), 3) if v.size else None,
                "mean_lift": round(float(v.mean()), 3) if v.size else None,
                "share_above_floor_q90": round(float((v >= thr).mean()), 4) if v.size else None,
                "share_lift_ge_5": round(float((v >= 5).mean()), 4) if v.size else None,
                "share_never_cooccur": round(float((v == 0).mean()), 4) if v.size else None}

    # 합의 여부로 갈라 본다 — 합의가 곧 누락이면 두 분포가 갈려야 한다
    uni = n_agree[rows_cfp, cols_cfp] == len(OTHERS)
    solo = n_agree[rows_cfp, cols_cfp] == 0

    # 실현된 조합 — 정답 집합에 예측 라벨을 더한 집합이 train에 실제로 존재하는가
    realized = np.array([frozenset(gold_rows[i].tolist() + [int(c)]) in combos
                         for i, c in zip(rows_cfp, cols_cfp)])
    gold_realized = np.array([frozenset(gold_rows[i].tolist()) in combos
                             for i in np.unique(rows_cfp)])

    # 확신 FN — 방향을 뒤집는다. 그 정답 라벨이 같은 문서의 다른 정답과 안 붙으면 오부착 후보.
    # 비교는 같은 k>=2 모집단 안에서 짝지어 한다(놓친 정답 · 경계 FN · 맞힌 정답).
    multi_doc = Y.sum(1) >= 2

    def gold_lift(mask):
        rows, cols = np.where(mask & multi_doc[:, None])
        v = np.array([max_lift(lift, gold_rows[i][gold_rows[i] != c], c)
                      for i, c in zip(rows, cols)])
        return v, rows, cols

    Yb = Y.astype(bool)
    fn_all = Yb & (pa < TAU)
    l_conf_fn, rows_cfn, cols_cfn = gold_lift(conf_fn)
    l_border_fn, _, _ = gold_lift(fn_all & ~conf)
    l_hit_gold, _, _ = gold_lift(Yb & (pa >= TAU))

    noise = known_noise_reference(lift, json.loads(
        (OUT / "label_conflict_docs.json").read_text(encoding="utf-8"))["conflict_groups"])

    return {
        "definition": "lift = P(a,b)/(P(a)P(b)) — train 201,616문서 기준. 1.0이 우연 수준. "
                      "확신 FP의 lift는 (문서의 정답 라벨, 확신 예측 라벨) 쌍의 최댓값이다.",
        "confound_note": "lift가 높다는 것만으로 '누락된 정답'이 되지는 않는다 — 모델이 코퍼스의 "
                         "공기 구조를 학습했으므로 그냥 틀린 예측도 lift가 높다(경계 FP가 그 대조군). "
                         "판정력은 두 곳에서 나온다: (1) 합의 여부로 가른 분포 차이, "
                         "(2) 입력 동일 라벨 충돌 42그룹(ADR-0010)이라는 실측 잡음 표본과의 대면.",
        "floor_q90_threshold": round(thr, 3),
        "distributions": [
            dist(floor, "바닥 — 무작위 비정답 라벨"),
            dist(l_border_fp, "경계 FP(확신 아님)"),
            dist(l_conf_fp, "확신 FP"),
            dist(l_conf_fp[uni], "확신 FP · 6런 전원 동조"),
            dist(l_conf_fp[solo], "확신 FP · 앵커 단독"),
            dist(noise, "실측 잡음 — 입력 동일 라벨 충돌(ADR-0010)"),
            dist(ceil, "천장 — 실제 정답끼리(k>=2)"),
        ],
        "fn_matched_within_k2": {
            "확신 FN": dist(l_conf_fn, "확신 FN"),
            "경계 FN": dist(l_border_fn, "경계 FN"),
            "맞힌 정답": dist(l_hit_gold, "맞힌 정답"),
            "note": "같은 k>=2 모집단에서 정답 라벨이 같은 문서의 다른 정답과 붙는 정도. "
                    "확신 FN이 맞힌 정답보다 크게 낮으면 그 라벨은 문서의 나머지와 겉도는 "
                    "이질 라벨 = 오부착 후보다.",
        },
        "realized_combination": {
            "conf_fp_gold_plus_pred_in_train": round(float(realized.mean()), 4),
            "conf_fp_gold_set_in_train": round(float(gold_realized.mean()), 4),
            "note": "정답 집합에 확신 예측 라벨을 더한 조합이 train에 실제로 존재하는 비율. "
                    "대조는 그 문서들의 정답 집합 자체가 train에 존재하는 비율.",
        },
        "confident_fn_shape": {
            "n_total": int(conf_fn.sum()),
            "n_k1_docs": int((conf_fn & ~multi_doc[:, None]).sum()),
            "n_k2_docs": int(l_conf_fn.size),
            "note": "k=1 문서의 확신 FN은 문서 안에 대조할 다른 정답이 없어 공기 검정 밖이다.",
        },
    }, conf_fp, conf_fn, rows_cfp, cols_cfp, l_conf_fp, thr, (l_conf_fn, rows_cfn, cols_cfn,
                                                              float(np.median(l_hit_gold)))


# ── C. 위치 조건화 ───────────────────────────────────────────────────────────
def conditioning(P, Y, conf, ls, length_bin):
    pa = P[ANCHOR]
    Yb = Y.astype(bool)
    pred = pa >= TAU
    gold_col = ls.to_lno(Yb)[:, ls.lno_idx]             # (N, C) 열 c가 정답 Lno 집합에 속하는가
    fp_all, fn_all = pred & ~Yb, Yb & ~pred
    conf_fp, conf_fn = conf & ~Yb, conf & Yb
    k = Yb.sum(1)

    def share_within(mask):
        n = int(mask.sum())
        return {"n": n, "within_gold_lno": round(float(gold_col[mask].mean()), 4) if n else None}

    pred_col = ls.to_lno(pred)[:, ls.lno_idx]
    doc_conf = conf.sum(1)
    return {
        "lno_position": {
            "전체 FP": share_within(fp_all),
            "확신 FP": share_within(conf_fp),
            "전체 FN": {"n": int(fn_all.sum()),
                        "within_pred_lno": round(float(pred_col[fn_all].mean()), 4)},
            "확신 FN": {"n": int(conf_fn.sum()),
                        "within_pred_lno": round(float(pred_col[conf_fn].mean()), 4)},
            "note": "FP는 예측 라벨이 정답 Lno 안(형제)인지, FN은 놓친 정답이 예측 Lno 안인지.",
        },
        "by_cardinality": {
            f"k={kk}": {
                "n_docs": int((k == kk).sum() if kk != "2+" else (k >= 2).sum()),
                "conf_err_per_doc": round(float(doc_conf[(k == kk) if kk != "2+" else (k >= 2)].mean()), 4),
            } for kk in (1, "2+")
        },
        "by_length_bin": {
            b: {"n_docs": int((length_bin == b).sum()),
                "conf_err_per_doc": round(float(doc_conf[length_bin == b].mean()), 4)}
            for b in sorted(set(length_bin))
        },
        "doc_concentration": {
            "n_docs_with_conf_err": int((doc_conf > 0).sum()),
            "share_of_docs": round(float((doc_conf > 0).mean()), 4),
            "max_per_doc": int(doc_conf.max()),
            "note": "확신 오답 원소가 몇 문서에 몰려 있는가.",
        },
    }


# ── D. focal 귀속 — γ만 뺀 BCE 대조 ──────────────────────────────────────────
def focal_attribution(P, Y, keep, n_old, n_clean):
    """"focal이 오답 확신을 키운다"는 해석의 직접 검정."""
    p_bce = load_prob(BCE_TAG, keep, n_old, n_clean)
    Yb = Y.astype(bool)
    sets = {"11_01(focal·신 레시피)": P["11_01"], "BCE(γ 제거·같은 레시피)": p_bce,
            "exp2(focal·구 레시피)": P["exp2"]}
    rows = {}
    for name, p in sets.items():
        c = confidently_wrong(p, Yb)
        fl = focal_elements(p, Yb)
        rows[name] = {
            "n_confident_errors": int(c.sum()),
            "mass_share_of_focal_loss": round(float(fl[c].sum() / fl.sum()), 4),
            "pos_saturation_ge_0.9": round(float((p[Yb] >= 0.9).mean()), 4),
        }
    a = confidently_wrong(P["11_01"], Yb)
    b = confidently_wrong(p_bce, Yb)
    rows["overlap(11_01 ∩ BCE)"] = {
        "n_intersection": int((a & b).sum()),
        "jaccard": round(float((a & b).sum() / (a | b).sum()), 4),
        "share_of_11_01": round(float((a & b).sum() / a.sum()), 4),
    }
    rows["note"] = ("BCE 런은 eff128/lr4.8e-4로 `11_01`과 레시피가 같고 γ·α만 없다. 다만 구 데이터 "
                    "훈련이라 데이터 축이 0.14%(336문서)만큼 겹치지 않는다 — 구 레시피 focal(exp2)을 "
                    "함께 놓아 괄호를 친다. 집합이 크게 겹치면 확신 오답은 손실이 만든 것이 아니다.")
    return rows


# ── E. 종합 분류 ─────────────────────────────────────────────────────────────
def verdict(P, Y, conf, n_agree, rows_cfp, cols_cfp, l_conf_fp, thr, fn_pack):
    """A×B 교차 — 확신 오답 전체를 갈래로 나누고 각 갈래가 문 손실 질량을 낸다."""
    Yb = Y.astype(bool)
    fl = focal_elements(P[ANCHOR], Yb)
    total_conf_mass = float(fl[conf].sum())
    total_mass = float(fl.sum())
    CONSENSUS = 3                                        # 5런 중 3런 이상 동조 = 합의

    l_conf_fn, rows_cfn, cols_cfn, hit_median = fn_pack
    conf_fn = conf & Yb
    k1_fn = conf_fn & (Y.sum(1) < 2)[:, None]            # 공기 검정 밖(문서에 다른 정답 없음)

    agree_fp, mass_fp = n_agree[rows_cfp, cols_cfp], fl[rows_cfp, cols_cfp]
    agree_fn, mass_fn = n_agree[rows_cfn, cols_cfn], fl[rows_cfn, cols_cfn]

    groups = [
        ("FP · 모델 고유 실패(합의 없음)", agree_fp < CONSENSUS, mass_fp),
        ("FP · 합의 + 공기 높음(라벨 누락 후보)", (agree_fp >= CONSENSUS) & (l_conf_fp >= thr), mass_fp),
        ("FP · 합의 + 공기 낮음(본질적 모호성)", (agree_fp >= CONSENSUS) & (l_conf_fp < thr), mass_fp),
        ("FN · 모델 고유 실패(합의 없음)", agree_fn < CONSENSUS, mass_fn),
        ("FN · 합의 + 공기 없음(오부착 후보)", (agree_fn >= CONSENSUS) & (l_conf_fn < thr), mass_fn),
        ("FN · 합의 + 공기 있음(본질적 난이도)", (agree_fn >= CONSENSUS) & (l_conf_fn >= thr), mass_fn),
    ]
    out = {name: {"n": int(g.sum()),
                  "mass_share_of_all_confident": round(float(m[g].sum() / total_conf_mass), 4)}
           for name, g, m in groups}
    out["FN · k=1 문서(공기 검정 밖)"] = {
        "n": int(k1_fn.sum()),
        "mass_share_of_all_confident": round(float(fl[k1_fn].sum() / total_conf_mass), 4),
        "agree_ge_consensus": round(float((n_agree[k1_fn] >= CONSENSUS).mean()), 4),
    }
    out["_totals"] = {
        "n_confident_errors": int(conf.sum()),
        "confident_mass_share_of_total_loss": round(total_conf_mass / total_mass, 4),
        "consensus_rule": f"5런 중 {CONSENSUS}런 이상 동조",
        "cooccurrence_rule": f"lift 판정선 = 바닥 90분위 {thr:.2f}",
        "fn_soft_rule_ref": {"hit_gold_median_lift": round(hit_median, 3),
                             "conf_fn_below_hit_median": round(float((l_conf_fn < hit_median).mean()), 4)},
    }
    return out


# ── F. 측정 편향 — 라벨 잡음 가정 하의 micro-F1 ─────────────────────────────
def measurement_bias(P, Y, n_agree, rows_cfp, cols_cfp, l_conf_fp, thr, fn_pack):
    """E의 잡음 후보가 전부 실제 잡음이라면 측정 micro-F1은 얼마나 하향 편향돼 있나.

    가정적 상한이며 판정선이 아니다 — 확신 밴드 안에서만 계산하므로 경계 오답에 섞인 잡음은
    빠져 있고, 후보가 전부 잡음이라는 가정도 도달 불가다.
    """
    Yb = Y.astype(bool)
    l_conf_fn, rows_cfn, cols_cfn, _ = fn_pack
    CONSENSUS = 3
    miss = (n_agree[rows_cfp, cols_cfp] >= CONSENSUS) & (l_conf_fp >= thr)
    spur = (n_agree[rows_cfn, cols_cfn] >= CONSENSUS) & (l_conf_fn < thr)

    def micro(pred, gold):
        tp = int((pred & gold).sum())
        return 2 * tp / (2 * tp + int((pred & ~gold).sum()) + int((~pred & gold).sum()))

    out = {}
    for name in (ANCHOR, "exp1"):
        pred = P[name] >= TAU
        adj = Yb.copy()
        adj[rows_cfp[miss], cols_cfp[miss]] = True                  # 누락된 정답으로 간주
        adj[rows_cfn[spur], cols_cfn[spur]] = False                 # 오부착으로 간주
        base, corr = micro(pred, Yb), micro(pred, adj)
        out[name] = {"micro_measured": round(base, 4), "micro_noise_adjusted": round(corr, 4),
                     "bias_pt": round(100 * (corr - base), 2)}
    docs = np.unique(np.concatenate([rows_cfp[miss], rows_cfn[spur]]))
    out["_scope"] = {
        "n_missing_label_candidates": int(miss.sum()),
        "n_spurious_label_candidates": int(spur.sum()),
        "n_docs_touched": int(docs.size),
        "share_of_test_docs": round(float(docs.size / len(Y)), 4),
        "note": "후보가 전부 실제 잡음이라는 가정 하의 가정적 값이다. 확신 밴드 밖(경계 오답)의 "
                "잡음은 포함되지 않으므로 상한도 하한도 아니다 — 편향의 규모 감각을 주는 값이다. "
                "잡음 조정 라벨로 학습·평가를 다시 하지 않는다(정답 재작성 아님).",
    }
    return out


def main():
    lm = json.load(open(hf_hub_download(RAW_DS, "label_mappings.json", repo_type="dataset"),
                        encoding="utf-8"))
    ls = LabelSpace(lm["id2mno"], lm["mno2lno"], NUM_LABELS)

    Y, length_bin, doc_ids, keep, n_old = load_axis()
    n_clean = len(Y)
    Yb = Y.astype(bool)
    P = {name: load_prob(tag, keep, n_old, n_clean) for name, tag in MODELS.items()}
    gold_rows = [np.where(Yb[i])[0] for i in range(n_clean)]

    # verify — 앵커 확신 오답 개수가 loss_mass_decomposition.json과 일치하는가
    lmd = json.loads((OUT / "loss_mass_decomposition.json").read_text(encoding="utf-8"))
    ref = lmd["models"]["11_01(A.X 512 b128)"]["confident_errors"]["n"]
    got = int(confidently_wrong(P[ANCHOR], Yb).sum())
    assert got == ref, (got, ref)
    print(f"[verify] 앵커 확신 오답 {got:,}개 == loss_mass_decomposition.json")
    print(f"[axis] 정리 {SPLIT} {n_clean:,}행 × {NUM_LABELS} · 양성 {int(Yb.sum()):,}\n")

    cons, conf, border, n_agree = consensus(P, Yb)
    lift, co, cnt, n_train, combos = train_cooccurrence()
    cooc, conf_fp, conf_fn, rows_cfp, cols_cfp, l_conf_fp, thr, fn_pack = cooccurrence_test(
        P, Y, lift, co, combos, conf, n_agree, gold_rows)
    cond = conditioning(P, Y, conf, ls, length_bin)
    attr = focal_attribution(P, Y, keep, n_old, n_clean)
    verd = verdict(P, Y, conf, n_agree, rows_cfp, cols_cfp, l_conf_fp, thr, fn_pack)
    bias = measurement_bias(P, Y, n_agree, rows_cfp, cols_cfp, l_conf_fp, thr, fn_pack)

    # ── 출력 ────────────────────────────────────────────────────────────────
    ce, be = cons["confident_errors"], cons["borderline_errors"]
    print(f"A. 교차 모델 합의 — 앵커 {ANCHOR}의 확신 오답 {ce['n']:,}개"
          f"(대조: 경계 오답 {be['n']:,}개)")
    print(f"  {'런':<8}{'확신 오답 동조':>14}{'경계 오답 동조':>16}{'확신까지 동조':>15}")
    for m in OTHERS:
        print(f"  {m:<8}{ce['per_model_agree'][m]:>13.1%}{be['per_model_agree'][m]:>15.1%}"
              f"{ce['per_model_agree_confident'][m]:>14.1%}")
    print(f"  평균 동조 런 수 {ce['mean_n_agree']:.2f}/5(경계 {be['mean_n_agree']:.2f})"
          f" · 전원 동조 {ce['universal_share']:.1%}(경계 {be['universal_share']:.1%})"
          f" · 앵커 단독 {ce['unique_share']:.1%}(경계 {be['unique_share']:.1%})")
    for key, label in (("confident_fp", "확신 FP"), ("confident_fn", "확신 FN")):
        d = cons[key]
        print(f"  [{label} {d['n']:,}개] 평균 동조 {d['mean_n_agree']:.2f}/5"
              f" · 전원 {d['universal_share']:.1%} · 단독 {d['unique_share']:.1%}"
              f" · KoBERT 동조 {d['per_model_agree']['KoBERT']:.1%}")

    print(f"\nB. 공기 검정 — 판정선(바닥 90분위) lift {thr:.2f}")
    print(f"  {'집합':<28}{'n':>8}{'중앙 lift':>11}{'판정선 위':>10}{'공기 0':>9}")
    for d in cooc["distributions"]:
        print(f"  {d['set']:<28}{d['n']:>8,}{d['median_lift']:>11.2f}"
              f"{d['share_above_floor_q90']:>10.1%}{d['share_never_cooccur']:>9.1%}")
    r = cooc["realized_combination"]
    print(f"  정답∪확신예측 조합이 train에 실재 {r['conf_fp_gold_plus_pred_in_train']:.1%}"
          f" (대조: 정답 집합 자체 {r['conf_fp_gold_set_in_train']:.1%})")
    print("  [확신 FN 역방향 — 같은 k>=2 모집단에서 짝지어]")
    for label in ("확신 FN", "경계 FN", "맞힌 정답"):
        d = cooc["fn_matched_within_k2"][label]
        print(f"    {label:<8}{d['n']:>7,}개 · 중앙 lift {d['median_lift']:>6.2f}"
              f" · 공기 0 {d['share_never_cooccur']:.1%}")
    s = cooc["confident_fn_shape"]
    print(f"    확신 FN {s['n_total']:,}개 중 k=1 문서 {s['n_k1_docs']:,}개는 공기 검정 밖")

    print("\nC. 위치 조건화")
    lp = cond["lno_position"]
    print(f"  정답 Lno 안 비율 — 전체 FP {lp['전체 FP']['within_gold_lno']:.1%}"
          f" · 확신 FP {lp['확신 FP']['within_gold_lno']:.1%}"
          f" | 예측 Lno 안 — 전체 FN {lp['전체 FN']['within_pred_lno']:.1%}"
          f" · 확신 FN {lp['확신 FN']['within_pred_lno']:.1%}")
    print("  문서당 확신 오답 — " + " · ".join(
        f"{k} {v['conf_err_per_doc']:.3f}" for k, v in cond["by_cardinality"].items())
        + " | " + " · ".join(f"{k} {v['conf_err_per_doc']:.3f}"
                             for k, v in cond["by_length_bin"].items()))
    dc = cond["doc_concentration"]
    print(f"  확신 오답을 가진 문서 {dc['n_docs_with_conf_err']:,}건({dc['share_of_docs']:.1%})"
          f" · 문서당 최대 {dc['max_per_doc']}개")

    print("\nD. focal 귀속 — γ만 뺀 BCE 대조")
    for name, v in attr.items():
        if name == "note" or "overlap" in name:
            continue
        print(f"  {name:<24} 확신 오답 {v['n_confident_errors']:>6,}"
              f" · 손실 질량 {v['mass_share_of_focal_loss']:.1%}"
              f" · 양성 포화 {v['pos_saturation_ge_0.9']:.1%}")
    ov = attr["overlap(11_01 ∩ BCE)"]
    print(f"  집합 겹침 — 교집합 {ov['n_intersection']:,} · Jaccard {ov['jaccard']:.3f}"
          f" · 앵커 확신 오답 중 {ov['share_of_11_01']:.1%}가 BCE에서도 확신 오답")

    print("\nE. 종합 분류 — 확신 오답을 A×B로 가른다")
    for name, v in verd.items():
        if name.startswith("_"):
            continue
        print(f"  {name:<36}{v['n']:>6,}개"
              f" · 확신 오답 손실 질량의 {v['mass_share_of_all_confident']:>5.1%}")
    t = verd["_totals"]
    print(f"  총 확신 오답 {t['n_confident_errors']:,}개가 전체 손실의"
          f" {t['confident_mass_share_of_total_loss']:.1%} · 규칙: {t['consensus_rule']} · {t['cooccurrence_rule']}")

    sc = bias["_scope"]
    print(f"\nF. 측정 편향 — 잡음 후보 {sc['n_missing_label_candidates']}(누락)"
          f"+{sc['n_spurious_label_candidates']}(오부착)가 전부 실제 잡음이라면"
          f" (문서 {sc['n_docs_touched']:,}건 · {sc['share_of_test_docs']:.1%})")
    for name in (ANCHOR, "exp1"):
        v = bias[name]
        print(f"  {name:<8} 측정 {v['micro_measured']:.4f} → 잡음 조정"
              f" {v['micro_noise_adjusted']:.4f} ({v['bias_pt']:+.2f}pt)")

    payload = {
        "question": "손실 질량의 73~82%를 무는 확신 오답 1,500~2,000개는 라벨 누락인가, "
                    "본질적 모호성인가, 모델 고유의 실패인가.",
        "axis": {"split": SPLIT, "n_docs": n_clean, "n_labels": NUM_LABELS,
                 "n_positive": int(Yb.sum()), "tau": TAU,
                 "confident_band": [CONF_LO, CONF_HI], "n_train_docs": n_train},
        "models": MODELS, "anchor": ANCHOR,
        "script": "scripts/confident_error_diagnosis.py",
        "verify": "앵커 확신 오답 개수 == output/loss_mass_decomposition.json · "
                  "로짓 행 축 == output/doc_ids_clean_test.json",
        "consensus": cons,
        "cooccurrence": cooc,
        "conditioning": cond,
        "focal_attribution": attr,
        "verdict": verd,
        "measurement_bias": bias,
    }
    path = OUT / "confident_error_diagnosis.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
