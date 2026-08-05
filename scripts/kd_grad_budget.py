"""KD 게이트 — 혼합 손실의 기울기 예산이 종점에서 어디로 가는가.

`knowledge-distillation.md`의 주 런 손실은 `L = (1-lam)*focal(z,y) + lam*BCE(z,q)`, lam=0.5다.
두 항의 소멸 속도가 다르다 — focal은 `alpha*(1-p_t)^gamma`로 자기소멸하고 BCE는 잔차에 선형
으로만 반응해 소멸하지 않는다. [ADR-0014]가 계층 손실에서 실측한 예산 포획과 **같은 형태**이며,
거기서는 초기화 값 일치로 고정한 lam이 종점에 기울기 67.9%를 가져갔다.

여기에 teacher 포화가 겹친다 — 앙상블 `q`는 하드 라벨 `y`와 거의 같으므로(문서당 중간대
라벨 2.01개/188), `BCE(z,q)`가 사실상 `BCE(z,y)`로 축퇴하면 종점 목적함수는 **하드 라벨 위의
BCE**가 된다. BCE는 focal 대비 −0.62pt로 이미 측정돼 있다([ADR-0009]).

  A. 예산 드리프트  초기화(스케일 스윕)와 종점에서 distill 항의 기울기 몫. lam=0.5가 종점에
                   무엇이 되는가.
  B. 축퇴도        `BCE(z,q)`의 기울기가 `BCE(z,y)`와 얼마나 같은가(코사인·L1 차). 1에 가까우면
                   증류가 아니라 하드 라벨 재학습이다.
  C. 온도 T        `q_T = sigmoid(logit(q)/T)`가 중간대·축퇴도·기울기 몫을 어떻게 바꾸는가.
                   T는 단조 변환이라 tau=0.5 결정을 바꾸지 않는다(가르치는 결정은 그대로).
  D. 설계 lam      종점 기울기 몫을 목표치로 두면 lam이 얼마여야 하는가.

student 종점은 아직 없으므로 **수렴한 focal 런의 종점 로짓**을 대역으로 쓴다(`11_01` 512 ·
exp1 8192). `q`는 test에서 만든 값이라 상한이다 — train 위 `q`는 teacher가 암기한 만큼 더
포화되므로 축퇴도는 여기서 잰 것보다 심해진다.

실행: `uv run python scripts/kd_grad_budget.py`
산출: `output/kd_grad_budget.json`

참조하는 HF 데이터셋은 공개 배포하지 않는다 — 재생성 절차는
`docs/data/data-pipeline.md`「가공 데이터셋은 배포하지 않는다 — 재현 경로」.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
ALPHA, GAMMA = 0.25, 2                                 # 운영 focal(= L_hard) 설정
LAM = 0.5                                              # knowledge-distillation.md 주 런 값
TEACHERS = {"exp1": "modernbert-patent-len8192",
            "asl": "modernbert-patent-len512-asl",
            "kobert": "kobert-patent-baseline_len512"}
W = {"exp1": 0.5, "asl": 0.2, "kobert": 0.3}
# student 종점 대역 — 수렴한 focal 런의 로짓
ENDPOINTS = {"11_01(512 focal)": "modernbert-patent-len512-b128",
             "exp1(8192 focal)": "modernbert-patent-len8192"}
MID = (0.05, 0.95)                                     # 하드 라벨과 구분되는 중간대


def load_axis():
    ds = load_gold(SPLIT, OUT)
    Y = build_gold(ds["label_ids"], len(ds), NUM_LABELS)
    clean_ids = list(ds["document_id"])
    stored = json.loads((OUT / f"doc_ids_clean_{SPLIT}.json").read_text(encoding="utf-8"))
    assert stored == clean_ids, "doc_ids_clean이 데이터셋 순서와 다르다(로짓 행 축)"
    old_ids = json.loads((OUT / f"doc_ids_{SPLIT}.json").read_text(encoding="utf-8"))
    pos = {d: i for i, d in enumerate(old_ids)}
    keep = np.array([pos[d] for d in clean_ids])
    assert np.all(np.diff(keep) > 0), "사영 인덱스가 단조 증가가 아니다"
    return Y, keep, len(old_ids)


def load_logits(tag, keep, n_old, n_clean):
    z = np.load(OUT / f"logits_{tag}_{SPLIT}.npy").astype(np.float64)
    if z.shape[0] == n_old:
        z = z[keep]
    assert z.shape == (n_clean, NUM_LABELS), (tag, z.shape)
    return z


def focal_loss(z, t):
    bce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
    pt = torch.exp(-bce)
    return (ALPHA * (1 - pt) ** GAMMA * bce).mean()


def soft_bce(z, q):
    return F.binary_cross_entropy_with_logits(z, q, reduction="mean")


def grad_of(fn, Zn, t):
    z = torch.tensor(Zn, requires_grad=True)
    v = fn(z, t)
    g = torch.autograd.grad(v, z)[0]
    return float(v.detach()), float(g.abs().sum()), g.numpy()


def temper(q, T):
    """q_T = sigmoid(logit(q)/T) — 단조 변환이라 tau=0.5 결정은 불변이다."""
    if T == 1.0:
        return q
    lg = np.log(np.clip(q, 1e-12, 1 - 1e-12) / np.clip(1 - q, 1e-12, 1 - 1e-12))
    return 1.0 / (1.0 + np.exp(-lg / T))


def mid_per_doc(q):
    return float(((q >= MID[0]) & (q <= MID[1])).sum(1).mean())


def shares(Zn, Yt, Qt):
    """lam=0.5에서 두 항의 손실 값 비·기울기 비·distill 몫."""
    vh, gh, Gh = grad_of(focal_loss, Zn, Yt)
    vd, gd, Gd = grad_of(soft_bce, Zn, Qt)
    return dict(
        focal=round(vh, 6), distill=round(vd, 6),
        value_ratio=round(LAM * vd / ((1 - LAM) * vh), 3),
        grad_ratio=round(LAM * gd / ((1 - LAM) * gh), 3),
        distill_grad_share=round(LAM * gd / ((1 - LAM) * gh + LAM * gd), 4),
    ), (gh, gd, Gh, Gd)


def signal_localization(Zn, Y, q, Gd, gd):
    """distill 기울기가 어디에 실리는가 — 이득을 만드는 밴드에 닿는가.

    Gate 1이 확인한 대로 앙상블 이득(+0.73pt)은 앙상블과 exp1의 결정이 갈리는 716개 원소
    (고침 449 · 깨뜨림 267)에서만 나온다. distill 항의 기울기 중 그 밴드의 몫을 재면, KD가
    이득의 원천을 가르치는 데 쓰는 예산의 크기가 나온다.
    """
    a = np.abs(Gd)
    contra = (q >= 0.5) != Y                            # teacher가 정답과 반대로 가르치는 원소
    diverge = (q >= 0.5) != (Zn >= 0.0)                 # 앙상블과 이 종점 모델의 결정이 갈리는 원소
    return dict(
        n_contradict=int(contra.sum()),
        grad_share_on_contradict=round(float(a[contra].sum() / a.sum()), 4),
        n_diverge=int(diverge.sum()),
        grad_share_on_diverge=round(float(a[diverge].sum() / a.sum()), 4),
        note="diverge = 앙상블과 종점 모델의 tau=0.5 결정이 갈리는 원소. 앙상블 이득은 "
             "전부 이 밴드에서 나오므로, 여기에 실리는 기울기 몫이 KD가 이득을 가르치는 데 "
             "쓰는 예산이다.",
    )


def degeneracy(Zn, Yt, Qt):
    """BCE(z,q)의 기울기가 BCE(z,y)와 얼마나 같은가 — 1이면 증류가 아니라 하드 라벨 재학습."""
    _, gq, Gq = grad_of(soft_bce, Zn, Qt)
    _, gy, Gy = grad_of(soft_bce, Zn, Yt)
    cos = float((Gq * Gy).sum() / (np.linalg.norm(Gq) * np.linalg.norm(Gy)))
    return dict(
        cosine_grad_q_vs_y=round(cos, 5),
        l1_diff_over_l1_y=round(float(np.abs(Gq - Gy).sum() / gy), 4),
        note="q의 기울기가 y의 기울기와 같은 방향·크기면 distill 항은 하드 라벨 BCE다.",
    )


def designed_lambda(gh, gd, targets=(0.2, 0.3, 0.5)):
    """종점 distill 기울기 몫을 목표치로 두는 lam — lam*gd/((1-lam)*gh + lam*gd) = s."""
    return {f"share={s}": round(float(s * gh / (gd * (1 - s) + s * gh)), 4) for s in targets}


def main():
    lm = json.load(open(OUT / "label_mappings.json",
                        encoding="utf-8"))
    LabelSpace(lm["id2mno"], lm["mno2lno"], NUM_LABELS)          # 라벨 축 존재 확인

    Y, keep, n_old = load_axis()
    n = len(Y)
    Yt = torch.as_tensor(Y.astype(np.float64))

    q = sum(W[k] * (1.0 / (1.0 + np.exp(-load_logits(t, keep, n_old, n))))
            for k, t in TEACHERS.items())
    gate = json.loads((OUT / "kd_gate_ensemble.json").read_text(encoding="utf-8"))
    P = q >= 0.5
    tp = int((P & Y).sum())
    ens_micro = 2 * tp / (2 * tp + int((P & ~Y).sum()) + int((~P & Y).sum()))
    assert abs(ens_micro - gate["ensembles"]["ens_best"]["micro"]) < 1e-3, ens_micro

    Z = {k: load_logits(t, keep, n_old, n) for k, t in ENDPOINTS.items()}

    print(f"정리 test {n:,} · lam {LAM} · 앙상블 q micro {ens_micro:.4f}"
          f" · 중간대 문서당 {mid_per_doc(q):.2f}/188")

    # A. 예산 드리프트 — 초기화(스케일 스윕)와 종점
    Qt = torch.as_tensor(q)
    rows = {}
    for s in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3):
        g = torch.Generator().manual_seed(0)
        zi = (torch.randn(4000, NUM_LABELS, generator=g, dtype=torch.float64) * s).numpy()
        r, _ = shares(zi, Yt[:4000], Qt[:4000])
        rows[f"init_s{s}"] = r
    ends = {}
    for name, Zn in Z.items():
        rows[name], ends[name] = shares(Zn, Yt, Qt)

    init_share = [v["distill_grad_share"] for k, v in rows.items() if k.startswith("init")]
    print(f"\nA. 예산 드리프트 — lam={LAM}에서 distill 항의 기울기 몫")
    print(f"  초기화(스케일 스윕) {min(init_share):.1%}~{max(init_share):.1%}")
    for name in ENDPOINTS:
        r = rows[name]
        print(f"  {name:>18} 종점 {r['distill_grad_share']:.1%}"
              f" (기울기 비 {r['grad_ratio']:.1f}x · 손실 값 비 {r['value_ratio']:.2f}x"
              f" · focal {r['focal']:.6f} / distill {r['distill']:.6f})")

    # B. 축퇴도
    deg = {name: degeneracy(Zn, Yt, Qt) for name, Zn in Z.items()}
    print("\nB. 축퇴도 — BCE(z,q) 기울기가 BCE(z,y)와 같은가")
    for name, d in deg.items():
        print(f"  {name:>18} cos {d['cosine_grad_q_vs_y']:.5f}"
              f" · |Δgrad|₁/|grad_y|₁ {d['l1_diff_over_l1_y']:.1%}")

    # B-2. 기울기가 실리는 자리 — 이득을 만드는 밴드에 닿는가
    loc = {name: signal_localization(Z[name], Y, q, ends[name][3], ends[name][1])
           for name in ENDPOINTS}
    print("\nB-2. distill 기울기가 실리는 자리")
    for name, v in loc.items():
        print(f"  {name:>18} 결정 갈림 {v['n_diverge']:,}개에 기울기"
              f" {v['grad_share_on_diverge']:.1%} · 정답과 반대 {v['n_contradict']:,}개에"
              f" {v['grad_share_on_contradict']:.1%}")

    # C. 온도
    temp = {}
    for T in (1.0, 1.5, 2.0, 3.0, 5.0):
        qT = temper(q, T)
        QT = torch.as_tensor(qT)
        row = {"mid_per_doc": round(mid_per_doc(qT), 3),
               "l1_to_hard": round(float(np.abs(qT - Y).mean()), 5)}
        for name, Zn in Z.items():
            r, _ = shares(Zn, Yt, QT)
            row[f"{name}_distill_share"] = r["distill_grad_share"]
            row[f"{name}_cos_q_vs_y"] = degeneracy(Zn, Yt, QT)["cosine_grad_q_vs_y"]
        temp[f"T={T}"] = row
    print("\nC. 온도 T (단조 변환이라 tau=0.5 결정 불변)")
    for k, v in temp.items():
        print(f"  {k:>6} 중간대 {v['mid_per_doc']:6.2f}/188 · 하드와 L1 {v['l1_to_hard']:.5f}"
              + "".join(f" · {name} 몫 {v[f'{name}_distill_share']:.1%}"
                        f"(cos {v[f'{name}_cos_q_vs_y']:.4f})" for name in ENDPOINTS))

    # D. 설계 lam
    design = {name: designed_lambda(ends[name][0], ends[name][1]) for name in ENDPOINTS}
    print("\nD. 종점 기울기 몫을 설계값으로 두는 lam")
    for name, d in design.items():
        print(f"  {name:>18} " + " · ".join(f"{k} → lam {v:.4f}" for k, v in d.items()))

    payload = {
        "gate": "kd_grad_budget",
        "question": "KD 혼합 손실의 lam=0.5가 종점에서 무엇이 되는가 — [ADR-0014]가 계층 손실에서 "
                    "실측한 예산 포획이 KD 항에서 반복되는가.",
        "n": n, "split": SPLIT, "lambda": LAM, "alpha": ALPHA, "gamma": GAMMA,
        "teachers": TEACHERS, "weights": W, "endpoints": ENDPOINTS,
        "ensemble_micro": round(ens_micro, 4), "mid_band": MID,
        "mid_per_doc_T1": round(mid_per_doc(q), 3),
        "script": "scripts/kd_grad_budget.py",
        "verify": "앙상블 test micro == output/kd_gate_ensemble.json · "
                  "로짓 행 축 == doc_ids_clean_test.json · "
                  "focal 구현 == scripts/hierarchy_loss_grad_budget.py",
        "budget": rows,
        "degeneracy": deg,
        "signal_localization": loc,
        "temperature": temp,
        "designed_lambda": design,
        "caveat": "q는 test에서 만든 값이라 상한이다 — train 위 q는 teacher가 암기한 만큼 더 "
                  "포화되므로 축퇴도는 여기서 잰 것보다 심해진다. student 종점은 아직 없어 "
                  "수렴한 focal 런의 종점 로짓을 대역으로 썼다.",
    }
    (OUT / "kd_grad_budget.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT / 'kd_grad_budget.json'}")


if __name__ == "__main__":
    main()
