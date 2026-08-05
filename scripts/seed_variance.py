"""훈련 잡음 — 같은 설정을 시드만 바꿔 재훈련했을 때의 run-to-run 변동.

[eval_noise_bootstrap.py](eval_noise_bootstrap.py)가 재는 **평가 표본 잡음**의 짝 성분이다.
저기서는 모델을 고정한 채 test 표본을 다시 뽑고, 여기서는 test를 고정한 채 훈련을 다시 돌린다.

`11_01`(seed 42)과 `11_04`(seed 153)은 시드 외 전 설정이 같다 — axenc·len512·focal(α=.25,γ=2)·
eff_batch 128·lr 4.8e-4·12 epoch·정리 데이터. 두 런 모두 24회 평가를 끝까지 돌고 마지막 체크포인트가
선택됐다. 따라서 두 값의 차이는 **시드(초기화·데이터 순서·dropout)만의 효과**다.

**관측된 시드 델타는 훈련 잡음의 추정치가 아니다.** 고정 test에서 잰 델타에도 평가 표본 잡음이
실려 있다 — 두 모델이 서로 다른 문서에서 틀리는 만큼 델타가 흔들린다. 분산이 더해지므로

    D² = 2σ_훈련² + σ_표본²      →      σ_훈련 = sqrt(max(0, D² − σ_표본²) / 2)

σ_표본은 시드 쌍의 paired bootstrap에서 온다(`eval_noise_bootstrap.json`의 시드 쌍 항목). 이 뺄셈을
빠뜨리고 D를 그대로 훈련 잡음으로 읽으면 잡음을 이중으로 세게 된다.

표본이 2개(자유도 1)이므로 σ_훈련은 점추정으로만 쓴다. D² < σ_표본²이면 훈련 잡음이 평가 잡음 바닥
아래라는 뜻이고, 이때 얻는 것은 추정치가 아니라 **상한**이다.

실행: `uv run python scripts/seed_variance.py`
산출: `output/seed_variance.json`

참조하는 HF 데이터셋은 공개 배포하지 않는다 — 재생성 절차는
`docs/data/data-pipeline.md`「가공 데이터셋은 배포하지 않는다 — 재현 경로」.
"""

import json
import os
import re
import sys
from html import unescape
from math import pi, sqrt
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
ROOT = Path(os.environ["DATA_ROOT"])
OUT = ROOT / "output"
NB = ROOT / "notebook_output"

PAIR = {
    "seed42": {"metrics": "modernbert-patent-len512-b128_metrics.json", "nb": "11_01_CleanData_Recipe.ipynb"},
    "seed153": {"metrics": "modernbert-patent-seed153_metrics.json", "nb": "11_04_Seed153.ipynb"},
}
METRICS = ["micro_f1", "macro_f1", "sample_f1", "anchor_weighted_f1", "empty_rate"]
SEED_KEY = "11_04(seed153) - 11_01(b128)"      # bootstrap의 시드 쌍 항목
BOOT_METRICS = ["micro", "macro", "sample", "micro_k>=2"]
# 설정이 다른 두 모델의 비교 — 여기에 훈련 잡음을 얹어 재판정한다
DELTAS = [
    "exp1(8192) - exp2(512)",
    "exp2(512) - KoBERT",
    "11_01(b128) - exp2(512)",
    "11_01(b128) - BCE",
    "11_04(seed153) - BCE",
    "11_01(b128) - ZLPR",
    "11_01(b128) - ASL",
]


def val_curve(nb_path: Path) -> list[dict]:
    """훈련 로그 테이블(HTML)에서 평가 시점별 val 지표를 뽑는다."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    tables = [
        "".join(o["data"]["text/html"])
        for c in nb["cells"]
        for o in c.get("outputs", [])
        if "data" in o and "text/html" in o["data"]
    ]
    best: list[dict] = []
    for tbl in tables:  # 노트북에는 업로드 위젯 등 다른 표도 있다 — 평가 행이 가장 많은 표를 고른다
        parsed = []
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.S):
            cells = [unescape(re.sub("<[^>]+>", "", x)).strip()
                     for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
            if len(cells) == 8 and cells[0].isdigit():
                step, _tr, _vl, mi, ma, sa, em, an = cells
                parsed.append({"step": int(step), "micro_f1": float(mi), "macro_f1": float(ma),
                               "sample_f1": float(sa), "empty_rate": float(em), "anchor_weighted_f1": float(an)})
        if len(parsed) > len(best):
            best = parsed
    assert best, f"{nb_path.name}에서 평가 로그 표를 찾지 못했다"
    return best


runs = {}
for name, spec in PAIR.items():
    test = json.loads((OUT / spec["metrics"]).read_text(encoding="utf-8"))["test"]
    runs[name] = {"test": {m: test[f"test_{m}"] for m in METRICS}, "curve": val_curve(NB / spec["nb"])}

a, b = runs["seed42"], runs["seed153"]
assert [r["step"] for r in a["curve"]] == [r["step"] for r in b["curve"]], "두 런의 평가 스텝이 다르다"

boot = json.loads((OUT / "eval_noise_bootstrap.json").read_text(encoding="utf-8"))["paired_delta"]
assert SEED_KEY in boot, f"bootstrap 산출에 시드 쌍 {SEED_KEY}이 없다 — eval_noise_bootstrap.py를 먼저 돌린다"

# ── 관측 시드 델타 (표본 잡음 포함, 기술 통계) ──────────────────────────────────
observed = {
    m: {"seed42_pt": round(a["test"][m] * 100, 3), "seed153_pt": round(b["test"][m] * 100, 3),
        "delta_pt": round((b["test"][m] - a["test"][m]) * 100, 3)}
    for m in METRICS
}

# ── 분산 분해: D² = 2σ_훈련² + σ_표본² ──────────────────────────────────────────
decomp = {}
for m in BOOT_METRICS:
    r = boot[SEED_KEY][m]
    d, sd_s = r["point_pt"], r["sd_pt"]
    var_train_delta = max(0.0, d ** 2 - sd_s ** 2)      # 2σ_훈련²
    decomp[m] = {
        "delta_pt": d,
        "ci95_pt": r["ci95_pt"],
        "sd_sample_pt": sd_s,
        "expected_abs_delta_if_no_train_noise_pt": round(sd_s * sqrt(2 / pi), 3),
        "var_train_delta": round(d ** 2 - sd_s ** 2, 4),      # 음수면 분해 불가(바닥 아래)
        "sigma_train_per_run_pt": round(sqrt(var_train_delta / 2), 3),
        "resolved": d ** 2 > sd_s ** 2,                       # 훈련 잡음이 표본 바닥 위로 올라왔나
        "ci_excludes_zero": r["ci95_pt"][0] > 0 or r["ci95_pt"][1] < 0,
    }

# ── 학습 궤적 위의 시드 스프레드 ────────────────────────────────────────────────
trajectory = [
    {"step": ra["step"], **{f"{m}_delta_pt": round((rb[m] - ra[m]) * 100, 3) for m in ["micro_f1", "empty_rate"]}}
    for ra, rb in zip(a["curve"], b["curve"])
]

# ── 두 잡음 합성으로 기존 델타 재판정 ────────────────────────────────────────────
# 각 비교의 자체 표본 sd에 훈련 잡음 분산(2σ_훈련²)만 더한다 — 시드 델타 D를 통째로 더하면
# 표본 잡음이 두 번 세어진다.
var_train = max(0.0, decomp["micro"]["var_train_delta"])
combined = {}
for key in DELTAS:
    src = boot[key]["micro"]
    sd_total = sqrt(src["sd_pt"] ** 2 + var_train)
    point = src["point_pt"]
    combined[key] = {
        "point_pt": point,
        "sd_sample_pt": src["sd_pt"],
        "sd_total_pt": round(sd_total, 3),
        "ci95_sample_only_pt": src["ci95_pt"],
        "ci95_total_pt": [round(point - 1.96 * sd_total, 3), round(point + 1.96 * sd_total, 3)],
        "survives": abs(point) > 1.96 * sd_total,
    }

result = {
    "axis": {"split": "test", "dataset": "ingyoun/patent-clean-text", "n": 11244, "tau": 0.5,
             "pair": ["11_01(b128) seed 42", "11_04 seed 153"], "n_runs": 2, "dof": 1},
    "observed_seed_delta": observed,
    "variance_decomposition": decomp,
    "trajectory": trajectory,
    "combined_noise": combined,
}
(OUT / "seed_variance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

print("[관측] 시드 델타 (seed153 − seed42, pt) — 표본 잡음 포함")
for m, v in observed.items():
    print(f"  {m:<20} {v['seed42_pt']:>7.3f} → {v['seed153_pt']:>7.3f}   Δ {v['delta_pt']:+.3f}")

print("\n[분해] D² = 2σ_훈련² + σ_표본²")
for m, v in decomp.items():
    verdict = f"σ_훈련 {v['sigma_train_per_run_pt']:.3f}pt" if v["resolved"] else "분해 불가 — 훈련 잡음이 표본 바닥 아래"
    print(f"  {m:<12} D {v['delta_pt']:+.3f}  표본 sd {v['sd_sample_pt']:.3f}  "
          f"CI {v['ci95_pt']}  순수표본 기대 |D| {v['expected_abs_delta_if_no_train_noise_pt']:.3f}  → {verdict}")

print(f"\n[합성] micro 델타 재판정 (2σ_훈련² = {var_train:.4f})")
for k, v in combined.items():
    print(f"  {k:<28} {v['point_pt']:+.2f}  sd {v['sd_sample_pt']:.2f}→{v['sd_total_pt']:.2f}  "
          f"CI [{v['ci95_total_pt'][0]:+.2f}, {v['ci95_total_pt'][1]:+.2f}]  "
          f"{'유지' if v['survives'] else '무너짐'}")
print(f"\n[save] {OUT / 'seed_variance.json'}")
