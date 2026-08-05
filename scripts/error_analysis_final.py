"""배포 모델(`16_01`, A.X 4096) 상세 오류 분석 — `error_analysis` 하니스 기반.

`16_01`은 `11_01`에서 `max_len`만 4096으로 바꾼 단일 변수 런이다. 이 스크립트는 회수된
로짓(`output/logits_modernbert-patent-len4096-op_test.npy`)을 정리 test(11,244) 축에 얹어
`error_analysis`의 기법 레지스트리를 전부 돌리고, 비교선 세 런과 문서 단위로 대조한다.

비교선
  - `11_01`(A.X 512, 정리 데이터·같은 레시피) — 창 크기만 다른 앵커. 길이 이득의 측정선.
  - exp1(A.X 8192, 정리 이전 데이터·구 레시피) — 창 상한. 4096의 길이 부채 확인선.
  - KoBERT(512, 정리 이전 데이터) — 벤더 baseline 재현.

축 정렬: 정리 축(11,244)에서 훈련·평가된 런(`16_01`·`11_01`)의 로짓은 그대로 얹고,
구 test(11,271) 축의 로짓은 `doc_ids_test.json`에서 위치를 찾아 사영한다.

`16_01`의 `{tag}_metrics.json`은 유실됐다 — 노트북 출력에 남은 test 지표(`SSOT_TEST`)를
로짓 재계산값과 대조해 행 축과 수치를 함께 증명한다.

실행: `uv run python scripts/error_analysis_final.py`
산출: `output/error_analysis_modernbert-patent-len4096-op.json`
      `output/final_model_comparison.json`

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
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset                          # noqa: E402  (HF_HOME 설정 뒤 import)
from huggingface_hub import hf_hub_download                # noqa: E402
from error_analysis import ErrorAnalysis                   # noqa: E402
from patent_train.metrics import empty_rate, f1_triple     # noqa: E402

OUT = ROOT / "output"
RAW_DS = "ingyoun/patent-clean-text"
SPLIT = "test"
NUM_LABELS = 188
TAU = 0.5
SEED = 42

TARGET = "modernbert-patent-len4096-op"          # 16_01 — 배포 모델
ANCHOR = "modernbert-patent-len512-b128"         # 11_01 — 창 크기만 다른 앵커
MODELS = [                                       # (표시명, 로짓 tag, 훈련 데이터)
    ("KoBERT(512)",   "kobert-patent-baseline_len512", "구"),
    ("exp2(A.X 512)", "modernbert-patent-len512",      "구"),
    ("exp1(A.X 8192)", "modernbert-patent-len8192",    "구"),
    ("11_01(A.X 512)", ANCHOR,                          "정리"),
    ("16_01(A.X 4096)", TARGET,                         "정리"),
]
TAGS = [t for _, t, _ in MODELS]

# 16_01 훈련 잡이 출력한 test 지표(`{tag}_metrics.json` 유실 — notebook_output/16_01_Model_4096.ipynb)
SSOT_TEST = {
    "micro_f1": 0.8660344890711383,
    "macro_f1": 0.8637787673728732,
    "sample_f1": 0.8835061135354624,
    "empty_rate": 0.008271077908217716,
    "anchor_weighted_f1": 0.825135139892277,
}


def micro_from_cells(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def paired_bootstrap(za, zb, Y, n_boot=4000, seed=SEED):
    """문서 단위 재표본으로 두 런의 micro 델타 분포를 낸다(`scripts/hierarchy_loss_mass.py`와 같은 절차)."""
    rng = np.random.default_rng(seed)
    cells = []
    for z in (za, zb):
        P = z >= 0.0                                  # sigmoid τ=0.5 ⟺ logit≥0
        cells.append(((P & Y).sum(1), (P & ~Y).sum(1), (~P & Y).sum(1)))
    n = len(Y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        m = [micro_from_cells(tp[i].sum(), fp[i].sum(), fn[i].sum()) for tp, fp, fn in cells]
        deltas[b] = m[1] - m[0]
    d = deltas * 100
    return {
        "n_boot": n_boot, "seed": seed,
        "delta_pt": round(float(d.mean()), 3),
        "ci95_pt": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
        "sd_pt": round(float(d.std()), 3),
        "p_delta_ge_0": round(float((d >= 0).mean()), 4),
    }


def micro_by_bin(z, Y, length_bin, bins):
    """길이 bin별 micro-F1 — 창 확장 이득이 어느 bin에서 오는지 본다."""
    P = z >= 0.0
    return {b: round(float(f1_score(Y[length_bin == b], P[length_bin == b],
                                    average="micro", zero_division=0)), 4) for b in bins}


def main():
    # ── 축 · 라벨 공간 ────────────────────────────────────────────────────
    label_mapping = json.load(open(
        hf_hub_download(repo_id=RAW_DS, filename="label_mappings.json", repo_type="dataset"),
        encoding="utf-8"))
    EA = ErrorAnalysis(label_mapping, num_labels=NUM_LABELS, tau=TAU)

    ds = load_dataset(RAW_DS, split=SPLIT)
    EA.set_data(ds)
    Y, length_bin, k_gold, BINS, N = EA.Y, EA.length_bin, EA.k_gold, EA.bins, EA.n

    clean_ids = list(ds["document_id"])
    stored = json.loads((OUT / f"doc_ids_clean_{SPLIT}.json").read_text(encoding="utf-8"))
    assert stored == clean_ids, "doc_ids_clean이 정리 데이터셋 순서와 다르다(정리 축 로짓의 행 축)"
    old_ids = json.loads((OUT / f"doc_ids_{SPLIT}.json").read_text(encoding="utf-8"))
    pos = {d: i for i, d in enumerate(old_ids)}
    KEEP = np.array([pos[d] for d in clean_ids])
    assert np.all(np.diff(KEEP) > 0), "사영 인덱스가 단조 증가가 아니다(행 순서 불일치)"

    print(f"평가 축 — 정리 {SPLIT} N={N:,} · 구 {SPLIT} {len(old_ids):,} → 사영 {len(KEEP):,}")
    print(f"라벨 개수 k=1~5 {np.bincount(k_gold)[1:6].tolist()} · k≥2 {(k_gold >= 2).mean():.2%}")
    print("길이 bin  " + "  ".join(f"{b} {int((length_bin == b).sum()):,}" for b in BINS))

    # ── 로짓 등록 ─────────────────────────────────────────────────────────
    logits = {}
    for _, tag, _ in MODELS:
        z = np.load(OUT / f"logits_{tag}_{SPLIT}.npy")
        if z.shape[0] == len(old_ids):
            z = z[KEEP]
        assert z.shape == (N, NUM_LABELS), (tag, z.shape)
        logits[tag] = z
        EA.add_logits(tag, z)
    rec = EA.records

    # ── verify(축·수치) — 재계산 헤드라인이 훈련 잡 출력과 일치 ──────────
    P_t = logits[TARGET] >= 0.0
    head_t = {**f1_triple(Y, P_t), "empty_rate": empty_rate(P_t)}
    head_t["anchor_weighted_f1"] = float(f1_score(Y.argmax(1), logits[TARGET].argmax(1),
                                                  average="weighted", zero_division=0))
    for k, v in SSOT_TEST.items():
        assert abs(head_t[k] - v) < 1e-9, (k, head_t[k], v)
    print(f"\nverify(축·수치) pass — 로짓 재계산 == 16_01 훈련 잡 test 출력 5/5 (유실된 "
          f"{TARGET}_metrics.json을 로짓으로 복원)")

    # ── 헤드라인 ─────────────────────────────────────────────────────────
    head = {}
    for name, tag, data in MODELS:
        P = logits[tag] >= 0.0
        head[tag] = {**f1_triple(Y, P), "empty_rate": empty_rate(P),
                     "p@1": rec[tag]["anchor_error"]["p@1"],
                     "anchor_weighted_f1": float(f1_score(Y.argmax(1), logits[tag].argmax(1),
                                                          average="weighted", zero_division=0))}

    print(f"\n[헤드라인] 정리 test {N:,} · τ={TAU}")
    print(f"{'모델':<18}{'데이터':>6}{'micro':>9}{'macro':>9}{'sample':>9}{'empty':>9}{'P@1':>9}{'앵커wF1':>10}")
    for name, tag, data in MODELS:
        h = head[tag]
        print(f"{name:<18}{data:>6}{h['micro_f1']:>9.4f}{h['macro_f1']:>9.4f}{h['sample_f1']:>9.4f}"
              f"{h['empty_rate']:>9.2%}{h['p@1']:>9.4f}{h['anchor_weighted_f1']:>10.4f}")
    for base_name, base_tag in [("11_01", ANCHOR), ("exp1", "modernbert-patent-len8192"),
                                ("KoBERT", "kobert-patent-baseline_len512")]:
        d = {k: head[TARGET][k] - head[base_tag][k]
             for k in ("micro_f1", "macro_f1", "sample_f1", "empty_rate", "p@1", "anchor_weighted_f1")}
        print(f"{'Δ(16_01 - ' + base_name + ', pt)':<24}{100*d['micro_f1']:>9.2f}{100*d['macro_f1']:>9.2f}"
              f"{100*d['sample_f1']:>9.2f}{100*d['empty_rate']:>9.2f}{100*d['p@1']:>9.2f}"
              f"{100*d['anchor_weighted_f1']:>10.2f}")

    # ── paired bootstrap ────────────────────────────────────────────────
    print(f"\n[paired bootstrap] 문서 단위 재표본 · Δmicro(16_01 - 기준)")
    boot = {}
    for name, tag, _ in MODELS:
        if tag == TARGET:
            continue
        r = paired_bootstrap(logits[tag], logits[TARGET], Y)
        boot[tag] = r
        print(f"  vs {name:<16} Δ {r['delta_pt']:>+6.3f}pt · CI95 [{r['ci95_pt'][0]:+.3f}, "
              f"{r['ci95_pt'][1]:+.3f}] · sd {r['sd_pt']:.3f} · P(Δ≥0)={r['p_delta_ge_0']:.4f}")

    # ── 앵커(top-1) 오류 분해 ────────────────────────────────────────────
    print(f"\n[앵커 오류 분해] top-1 오답을 sibling(대분류 적중) / cross-Lno(대분류 이탈)로")
    print(f"{'모델':<18}{'P@1':>8}{'오류':>8}{'sibling':>17}{'cross-Lno':>17}{'우연':>7}{'배수':>7}")
    for name, tag, _ in MODELS:
        r = rec[tag]["anchor_error"]
        sib = f"{r['sibling']:,} ({r['sibling_ratio']:.1%})"
        cro = f"{r['cross_lno']:,} ({1 - r['sibling_ratio']:.1%})"
        print(f"{name:<18}{r['p@1']:>8.4f}{r['n_error']:>8,}{sib:>17}{cro:>17}"
              f"{r['chance_sibling_ratio']:>7.1%}{r['sibling_enrichment']:>6.1f}x")

    # ── 멀티라벨(τ) 오류 분해 ────────────────────────────────────────────
    print(f"\n[멀티라벨 오류 분해] τ={TAU}")
    print(f"{'모델':<18}{'FP':>8}{'FP sib':>17}{'FN':>8}{'FN sib':>17}{'empty':>9}")
    for name, tag, _ in MODELS:
        r = rec[tag]["multilabel_error"]
        fps = f"{r['fp_sibling']:,} ({r['fp_sibling_ratio']:.1%})"
        fns = f"{r['fn_sibling']:,} ({r['fn_sibling_ratio']:.1%})"
        print(f"{name:<18}{r['fp']:>8,}{fps:>17}{r['fn']:>8,}{fns:>17}{r['empty_rate']:>9.2%}")

    # ── 라벨 개수 bin · 카디널리티 ───────────────────────────────────────
    card_t = rec[TARGET]["cardinality"]
    print(f"\n[라벨 개수 bin] 양성 라벨 인스턴스 {int(Y.sum()):,} · k≥2 점유율 {card_t['pos_share_k>=2']:.1%}")
    print(f"{'모델':<18}{'bin':>6}{'n':>8}{'micro':>9}{'sample':>9}{'R-Prec':>9}{'FP':>8}{'FN':>8}{'FP:FN':>8}")
    for name, tag, _ in MODELS:
        r = rec[tag]["label_count_bins"]
        for b in ["k=1", "k>=2"]:
            v = r[b]
            print(f"{name if b == 'k=1' else '':<18}{b:>6}{v['n']:>8,}{v['micro_f1']:>9.4f}"
                  f"{v['sample_f1']:>9.4f}{v['r_precision']:>9.4f}{v['fp']:>8,}{v['fn']:>8,}"
                  f"{v['fp_fn_ratio']:>8.2f}")

    print(f"\n[카디널리티 헤드룸] k≥2 문서에 오라클 k(정답 개수)를 준 micro 상한")
    print(f"{'모델':<18}{'micro':>9}{'+오라클k':>11}{'이득':>9}{'k≥2 과소예측':>15}{'예측/정답':>13}")
    for name, tag, _ in MODELS:
        r = rec[tag]["cardinality"]
        kp = f"{r['mean_k_pred_k>=2']:.2f}/{r['mean_k_gold_k>=2']:.2f}"
        print(f"{name:<18}{r['micro']:>9.4f}{r['micro_oracle_k_on_multi']:>11.4f}"
              f"{r['oracle_k_gain_pt']:>+9.2f}{r['under_predict_rate_k>=2']:>15.1%}{kp:>13}")

    # ── 길이 bin ────────────────────────────────────────────────────────
    bin_micro = {tag: micro_by_bin(logits[tag], Y, length_bin, BINS) for tag in TAGS}
    print(f"\n[길이 bin × micro-F1] 창 확장 이득의 소재")
    print(f"{'모델':<18}" + "".join(f"{b:>12}" for b in BINS))
    for name, tag, _ in MODELS:
        print(f"{name:<18}" + "".join(f"{bin_micro[tag][b]:>12.4f}" for b in BINS))
    for base_name, base_tag in [("11_01", ANCHOR), ("exp1", "modernbert-patent-len8192")]:
        print(f"{'Δ(16_01 - ' + base_name + ', pt)':<18}"
              + "".join(f"{100*(bin_micro[TARGET][b] - bin_micro[base_tag][b]):>+12.2f}" for b in BINS))

    print(f"\n[길이 bin × 오류 유형] 16_01")
    print(f"  {'bin':<12}{'n':>7}{'앵커오류율':>11}{'sibling비':>11}{'FP/문서':>10}{'FN/문서':>10}")
    for b in BINS:
        v = rec[TARGET]["length_bin_error"][b]
        print(f"  {b:<12}{v['n']:>7,}{v['anchor_error_rate']:>11.2%}{v['sibling_ratio']:>11.1%}"
              f"{v['fp_per_doc']:>10.3f}{v['fn_per_doc']:>10.3f}")

    # ── 대분류(Lno) 유도 성능 ───────────────────────────────────────────
    print(f"\n[대분류(Lno) 유도 성능] Mno 예측을 M2L로 사영 — 별도 헤드 없음")
    print(f"{'모델':<18}{'micro':>9}{'macro':>9}{'sample':>9}{'P@1':>9}{'오라클Lno':>11}{'2단계추정':>11}{'Δflat':>9}")
    for name, tag, _ in MODELS:
        r = rec[tag]["lno_metrics"]
        print(f"{name:<18}{r['micro_f1']:>9.4f}{r['macro_f1']:>9.4f}{r['sample_f1']:>9.4f}"
              f"{r['p@1']:>9.4f}{r['oracle_lno_p@1']:>11.4f}{r['two_stage_p@1_est']:>11.4f}"
              f"{r['delta_vs_flat']:>+9.4f}")
    verdict = EA.hierarchy_verdict(TARGET)
    print(f"  계층 확장 판정({TARGET}): {verdict['decision']}")

    # ── cross-Lno 혼동 쌍 ───────────────────────────────────────────────
    pr = rec[TARGET]["pair_analysis"]
    print(f"\n[cross-Lno 혼동 쌍] off-diagonal {pr['off_diagonal_total']:,} · "
          f"top5 {pr['top5_share']:.1%} · top10 {pr['top10_share']:.1%}")
    print(f"  {'쌍':<16}{'합':>7}{'a→b':>7}{'b→a':>7}{'대칭도':>8}")
    for p in pr["top5_pairs"]:
        print(f"  {p['pair']:<16}{p['total']:>7,}{p['ab']:>7,}{p['ba']:>7,}{p['symmetry']:>8.2f}")
    print(f"  국소 처리 상한(정답 Lno로 제한 재선택): "
          + " · ".join(f"top{n} +{pr['oracle_gain'][f'top{n}']['p@1_gain_pt']:.2f}pt" for n in (1, 5, 10)))

    # ── 오류 차집합 · hard core ─────────────────────────────────────────
    print(f"\n[오류 차집합] 앵커 오류가 어디서 고쳐지고(fixed) 어디서 깨지는가(broken)")
    pairs = []
    for base_name, base_tag, comp in [("11_01", ANCHOR, "max_len 512→4096"),
                                      ("exp1", "modernbert-patent-len8192", "data+recipe+len")]:
        c = EA.compare(base_tag, TARGET, comp)
        pairs.append(c)
        print(f"  [{comp}] {base_name} → 16_01")
        print(f"    {'':<9}{'n':>7}{'k>=2':>7}{'sibling':>9}{'cross':>8}   " + "".join(f"{b:>11}" for b in BINS))
        for nm in ["fixed", "broken"]:
            v = c[nm]
            print(f"    {nm:<9}{v['n']:>7,}{v['k>=2']:>7,}{v['sibling']:>9,}{v['cross_lno']:>8,}   "
                  + "".join(f"{v['by_length_bin'][b]:>11,}" for b in BINS))
        print(f"    {'순이득':<7}{c['net_gain']:>+7,}{'':>24}   "
              + "".join(f"{c['net_by_bin'][b]:>+11,}" for b in BINS))
        print(f"    {'교정률':<7}{'':>7}{'':>24}   "
              + "".join(f"{c['fix_rate_by_bin'][b]:>11.1%}" for b in BINS))

    hc = EA.hard_core()
    print(f"\n  hard core(5개 런 공통 오류) {int(hc.sum()):,}건 ({hc.mean():.2%}) · "
          + " ".join(f"{b} {int((hc & (length_bin == b)).sum()):,}" for b in BINS))

    # ── per-class 대조(vs 앵커) ─────────────────────────────────────────
    pc_t, pc_a = rec[TARGET]["per_class_f1"], rec[ANCHOR]["per_class_f1"]
    MNOS = EA.ls.mno_of_col
    delta = np.array([pc_t[m]["f1"] - pc_a[m]["f1"] for m in MNOS])
    support = np.array([pc_t[m]["support"] for m in MNOS])
    f1_t = np.array([pc_t[m]["f1"] for m in MNOS])
    order = np.argsort(delta)
    print(f"\n[per-class F1] 16_01 vs 11_01 — 개선 {int((delta > 0).sum())}/188 클래스 · "
          f"평균 Δ {100*delta.mean():+.2f}pt · support 상관 {np.corrcoef(support, f1_t)[0,1]:+.3f}")
    print(f"  {'하락 5':<8}" + "".join(f"{MNOS[i]}({100*delta[i]:+.1f})  " for i in order[:5]))
    print(f"  {'개선 5':<8}" + "".join(f"{MNOS[i]}({100*delta[i]:+.1f})  " for i in order[-5:][::-1]))
    worst = np.argsort(f1_t)[:5]
    print(f"  {'최저 F1 5':<8}" + "".join(f"{MNOS[i]}({f1_t[i]:.3f}/n={support[i]})  " for i in worst))

    # ── verify(per-class) ───────────────────────────────────────────────
    for tag in TAGS:
        pcf = rec[tag]["per_class_f1"]
        assert sum(v["support"] for v in pcf.values()) == int(Y.sum())
        macro = float(np.mean([v["f1"] for v in pcf.values()]))
        assert abs(macro - head[tag]["macro_f1"]) < 5e-4, (tag, macro, head[tag]["macro_f1"])
    print("\nverify(per-class) pass — support 합 == 양성 인스턴스 · per-class 평균 == macro-F1 (5/5)")

    # ── 저장 ────────────────────────────────────────────────────────────
    meta = {
        "split": SPLIT, "n_docs": int(N), "tau": TAU, "num_labels": NUM_LABELS,
        "num_lno": EA.ls.L, "raw_ds": RAW_DS, "script": "scripts/error_analysis_final.py",
        "axis_note": "정리 test 11,244. 구 test(11,271) 축 로짓은 doc_ids_test.json으로 사영.",
    }
    record = {**meta, "tag": TARGET, "arch": "modernbert", "recipe": "eff128/lr4.8e-4/len4096",
              "data": "clean", "run": "16_01",
              "ssot_test": SSOT_TEST,
              "ssot_note": f"{TARGET}_metrics.json 유실 — 노트북 출력값이며 로짓 재계산과 1e-9 내 일치",
              **rec[TARGET]}
    fp = OUT / f"error_analysis_{TARGET}.json"
    fp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {fp}")

    comparison = {
        **meta,
        "target": TARGET, "anchor": ANCHOR,
        "models": {tag: name for name, tag, _ in MODELS},
        "headline": {tag: {k: round(v, 4) for k, v in h.items()} for tag, h in head.items()},
        "paired_bootstrap": boot,
        "micro_by_length_bin": bin_micro,
        "length_bin_delta_pt": {
            base: {b: round(100 * (bin_micro[TARGET][b] - bin_micro[base][b]), 3) for b in BINS}
            for base in (ANCHOR, "modernbert-patent-len8192")},
        "error_diff": pairs,
        "hard_core": {"n": int(hc.sum()),
                      "by_length_bin": {b: int((hc & (length_bin == b)).sum()) for b in BINS}},
        "hierarchy_verdict": verdict,
        "per_class_delta_vs_anchor_pt": {m: round(100 * float(d), 2) for m, d in zip(MNOS, delta)},
    }
    fp = OUT / "final_model_comparison.json"
    fp.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {fp}")


if __name__ == "__main__":
    main()
