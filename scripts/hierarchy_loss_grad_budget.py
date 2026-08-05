"""계층 손실(MCLoss) 실패 기제 — 기울기 예산·포화도·순위 손실의 국소화, 그리고 세 갈래 판정.

`hierarchy_loss_mass.py`가 "무엇이 얼마나 움직였나"(표적 질량·작동점)를 소유한다면 이 스크립트는
**왜 움직였나**를 소유한다. `14_01`(focal + lambda*group)이 앵커 `11_01`(focal) 대비 −1.20pt로
떨어진 원인을 덤프된 로짓만으로(GPU 0) 분해하고, 남은 세 갈래를 각각 수치로 친다.

  A. 기울기 예산   lambda는 손실 '값'을 초기화 시점에 맞췄다. 기울기 비는 그 시점에 최소이고
                  종점에서 뒤집힌다 — focal은 alpha*(1-p)^gamma/188로 자기소멸하는 반면 그룹
                  항은 max 위의 평범한 로그손실 1/17이라 소멸하지 않는다.
  B. 포화도        그룹 목적(`Lno` 축)은 훈련 전에 이미 대부분 충족돼 있다. 남은 기울기의 절반은
                  어떤 결정도 바꾸지 못하는 확신 부풀리기다.
  C. 로짓 지오메트리 정답 +1.5 / `Lno` 밖 −3.8로 축을 재편했고, 두 항 어디에도 닿지 않는 형제
                  밴드가 임계 쪽으로 떠올랐다(시드 쌍둥이 `11_04` 대조로 잡음과 분리).
  D. 순위 손실     P@1 하락이 `Lno` 축이 아니라 조건부 within-`Lno` 축에 실린다(작동점 무관).
  E. 세 갈래       (1) BCE 기반 교체 (2) lambda 재설정 (3) 종결 — 앞의 둘이 산술로 닫힌다.
                  lambda 축은 훈련 곡선이 대리 실험이다(초반 = 그룹 항 기울기 몫이 낮은 국면).

실행: `uv run python scripts/hierarchy_loss_grad_budget.py`
산출: `output/hierarchy_loss_grad_budget.json`

참조하는 HF 데이터셋은 공개 배포하지 않는다 — 재생성 절차는
`docs/data/data-pipeline.md`「가공 데이터셋은 배포하지 않는다 — 재현 경로」.
"""

import json
import os
import re
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
sys.path.insert(0, str(ROOT / "scripts"))

from gold_labels import load_gold                    # noqa: E402  (저장소 동봉 정답 축 — 데이터셋 불필요)
from error_analysis import LabelSpace, build_gold      # noqa: E402

OUT = ROOT / "output"
NB = ROOT / "notebook_output"
RAW_DS = "ingyoun/patent-clean-text"
SPLIT = "test"
NUM_LABELS, N_GROUPS = 188, 17
ALPHA, GAMMA = 0.25, 2                                 # 운영 focal(=앵커) 설정
LAM = 0.044393                                         # 14_01이 고정한 lambda(초기화 1배치 값 일치)
L_FOCAL_INIT, L_GROUP_INIT = 0.065730, 1.480618        # 그 산출에 쓰인 두 항의 원시 크기

TAGS = {
    "11_01(focal)": "modernbert-patent-len512-b128",
    "14_01(MCL)": "modernbert-patent-len512-mcl",
    "11_04(seed153)": "modernbert-patent-seed153",
    "BCE": "modernbert-patent-len512-bce",
}
ANCHOR, ARM, SEED, BCE = "11_01(focal)", "14_01(MCL)", "11_04(seed153)", "BCE"


# ── 축 · 로짓 ────────────────────────────────────────────────────────────────
def load_axis():
    ds = load_gold(SPLIT, OUT)
    Y = build_gold(ds["label_ids"], len(ds), NUM_LABELS)
    clean_ids = list(ds["document_id"])
    stored = json.loads((OUT / f"doc_ids_clean_{SPLIT}.json").read_text(encoding="utf-8"))
    assert stored == clean_ids, "doc_ids_clean이 정리 데이터셋 순서와 다르다(로짓 행 축)"
    old_ids = json.loads((OUT / f"doc_ids_{SPLIT}.json").read_text(encoding="utf-8"))
    pos = {d: i for i, d in enumerate(old_ids)}
    keep = np.array([pos[d] for d in clean_ids])
    assert np.all(np.diff(keep) > 0), "사영 인덱스가 단조 증가가 아니다"
    return Y, keep, len(old_ids)


def load_logits(tag, keep, n_old, n_clean):
    z = np.load(OUT / f"logits_{tag}_{SPLIT}.npy")
    if z.shape[0] == n_old:
        z = z[keep]
    assert z.shape == (n_clean, NUM_LABELS), (tag, z.shape)
    return z.astype(np.float64)


# ── 손실 세 종 — 훈련에 쓰인 구현과 동일한 리덕션 ────────────────────────────
def focal_loss(z, t):
    bce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
    pt = torch.exp(-bce)
    return (ALPHA * (1 - pt) ** GAMMA * bce).mean()


def bce_loss(z, t):
    return F.binary_cross_entropy_with_logits(z, t, reduction="mean")


def group_parts(z, t, lno):
    """MclFocalLoss.group_loss(`14_01` 5번 셀)와 항등 — 그룹별 손실·max 로짓·양성 여부."""
    n, neg = z.shape[0], torch.finfo(z.dtype).min
    idx = lno.expand(n, -1)
    gz = z.new_full((n, N_GROUPS), neg).scatter_reduce(1, idx, z, "amax")
    gz_pos = z.new_full((n, N_GROUPS), neg).scatter_reduce(
        1, idx, z.masked_fill(t <= 0, neg), "amax")
    gt = z.new_zeros((n, N_GROUPS)).scatter_reduce(1, idx, t, "amax")
    p = gt > 0
    h = torch.where(p, gz_pos, gz)
    return torch.where(p, -F.logsigmoid(h), -F.logsigmoid(-h)), h, p


def make_group_loss(lno):
    return lambda z, t: group_parts(z, t, lno)[0].mean()


def value_and_gradnorm(fn, Zn, t):
    """손실 값과 로짓 기울기의 L1 총량."""
    z = torch.tensor(Zn, requires_grad=True)
    g = torch.autograd.grad(fn(z, t), z)[0]
    return float(fn(torch.as_tensor(Zn), t)), float(g.abs().sum()), g.numpy()


# ── A. 기울기 예산 ───────────────────────────────────────────────────────────
def budget(Z, T, lno, group_loss):
    """초기화(모사 스윕)와 각 종점에서 group/base의 손실 값 비·기울기 비.

    초기화 로짓의 실제 분포는 알 수 없으므로 스케일 s를 훑는다 — 비가 s에 둔감하다는 것이
    결론이며, 그 값이 종점의 비와 얼마나 벌어지는지가 판정 대상이다.
    """
    rows, ti = {}, T[:4000]
    lam_bce_candidates = []
    for s in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
        g = torch.Generator().manual_seed(0)
        zi = (torch.randn(4000, NUM_LABELS, generator=g, dtype=torch.float64) * s).numpy()
        vf, gf, _ = value_and_gradnorm(focal_loss, zi, ti)
        vb, gb, _ = value_and_gradnorm(bce_loss, zi, ti)
        vg, gg, _ = value_and_gradnorm(group_loss, zi, ti)
        lam_bce_candidates.append(vb / vg)
        rows[f"init_s{s}"] = dict(
            focal=round(vf, 6), bce=round(vb, 6), group=round(vg, 6),
            grad_ratio_focal_base=round(LAM * gg / gf, 3),
            value_ratio_focal_base=round(LAM * vg / vf, 3))
    lam_bce = float(np.mean(lam_bce_candidates))       # 같은 규칙(초기화 값 일치)의 BCE판 lambda
    for s in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
        g = torch.Generator().manual_seed(0)
        zi = (torch.randn(4000, NUM_LABELS, generator=g, dtype=torch.float64) * s).numpy()
        _, gb, _ = value_and_gradnorm(bce_loss, zi, ti)
        _, gg, _ = value_and_gradnorm(group_loss, zi, ti)
        rows[f"init_s{s}"]["grad_ratio_bce_base"] = round(lam_bce * gg / gb, 3)

    for name, Zn in Z.items():
        vf, gf, _ = value_and_gradnorm(focal_loss, Zn, T)
        vb, gb, _ = value_and_gradnorm(bce_loss, Zn, T)
        vg, gg, _ = value_and_gradnorm(group_loss, Zn, T)
        rows[name] = dict(
            focal=round(vf, 6), bce=round(vb, 6), group=round(vg, 6),
            value_ratio_focal_base=round(LAM * vg / vf, 3),
            grad_ratio_focal_base=round(LAM * gg / gf, 3),
            grad_share_focal_base=round(LAM * gg / (gf + LAM * gg), 4),
            grad_ratio_bce_base=round(lam_bce * gg / gb, 3),
            grad_share_bce_base=round(lam_bce * gg / (gb + lam_bce * gg), 4))
    return rows, lam_bce


def touched(Zn, T, lno, group_loss):
    """그룹 항이 기울기를 주는 로짓(문서당 17개)에서 두 항의 평균 기울기 크기."""
    _, gf, GF = value_and_gradnorm(focal_loss, Zn, T)
    _, gg, GG = value_and_gradnorm(group_loss, Zn, T)
    GG = LAM * GG
    z = torch.as_tensor(Zn)
    _, _, p = group_parts(z, T, lno)
    big = torch.finfo(z.dtype).min
    cols = []
    for g in range(N_GROUPS):
        c = torch.where(lno == g)[0]
        am = torch.where(p[:, g], z[:, c].masked_fill(T[:, c] <= 0, big).argmax(1),
                         z[:, c].argmax(1))
        cols.append(c[am])
    cols = torch.stack(cols, 1).numpy()
    pn = p.numpy()
    out = {}
    for name, sel in [("all", np.ones_like(pn, bool)), ("pos_group", pn), ("neg_group", ~pn)]:
        m = np.zeros_like(Zn, bool)
        r, g = np.where(sel)
        m[r, cols[r, g]] = True
        out[name] = dict(n=int(m.sum()), focal=float(f"{np.abs(GF[m]).mean():.4g}"),
                         lam_group=float(f"{np.abs(GG[m]).mean():.4g}"),
                         ratio=round(float(np.abs(GG[m]).mean() / np.abs(GF[m]).mean()), 2))
    return out


# ── B. 그룹 목적의 포화도 ────────────────────────────────────────────────────
def saturation(Zn, T, lno):
    per, h, p = group_parts(torch.as_tensor(Zn), T, lno)
    sig, pn, per = torch.sigmoid(h).numpy(), p.numpy(), per.numpy()
    ok_pos, ok_neg = pn & (sig > 0.5), (~pn) & (sig < 0.5)
    gmag = np.where(pn, 1 - sig, sig)                  # |dl/dh| — 그룹 단위 기울기 크기
    return dict(
        pos_groups=int(pn.sum()), pos_already_satisfied=int(ok_pos.sum()),
        pos_satisfied_rate=round(float(ok_pos.sum() / pn.sum()), 4),
        neg_groups=int((~pn).sum()), neg_already_satisfied=int(ok_neg.sum()),
        neg_satisfied_rate=round(float(ok_neg.sum() / (~pn).sum()), 4),
        grad_share_on_satisfied=round(float(gmag[ok_pos | ok_neg].sum() / gmag.sum()), 4),
        loss_share_on_satisfied=round(float(per[ok_pos | ok_neg].sum() / per.sum()), 4))


# ── C. 로짓 지오메트리 ───────────────────────────────────────────────────────
def geometry(Z, Y, pos_group):
    cats = {"true_label": Y,                            # 양성 항이 미는 대상(그룹당 최상위 1개)
            "sibling_in_pos_lno": pos_group & ~Y,       # 두 항 어디에도 닿지 않는 밴드
            "outside_pos_lno": ~pos_group}              # 음성 항이 누르는 대상
    base = Z[ANCHOR]
    out = {}
    for cname, m in cats.items():
        out[cname] = {"n": int(m.sum()), "anchor_mean_logit": round(float(base[m].mean()), 3)}
        for name in (ARM, SEED):
            out[cname][name] = round(float(Z[name][m].mean() - base[m].mean()), 3)
    out["logit_sd"] = {k: round(float(v.std()), 3) for k, v in Z.items()}
    return out


# ── D. 순위 손실의 위치 ──────────────────────────────────────────────────────
def rank_vectors(Zn, Y, ls):
    """문서별 0/1 — P@1 · `Lno` top-1 적중 · (그 문서에서) 그룹 내 top-1 `Mno` 적중."""
    n = len(Y)
    top1 = Zn.argmax(1)
    p1 = Y[np.arange(n), top1]
    gmax = np.stack([Zn[:, ls.lno_idx == g].max(1) for g in range(N_GROUPS)], 1)
    lno_top1 = gmax.argmax(1)
    lno_ok = ls.to_lno(Y)[np.arange(n), lno_top1].astype(bool)
    within = np.zeros(n, bool)
    for g in range(N_GROUPS):
        sel = lno_ok & (lno_top1 == g)
        c = np.where(ls.lno_idx == g)[0]
        if sel.sum():
            within[sel] = Y[np.where(sel)[0], c[Zn[sel][:, c].argmax(1)]]
    order = (-Zn).argsort(1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(NUM_LABELS)[None, :].repeat(n, 0), axis=1)
    return p1.astype(bool), lno_ok, within, float(rank[Y].mean() + 1)


def rank_localization(Z, Y, ls, n_boot=2000, seed=0):
    vec = {k: rank_vectors(v, Y, ls) for k, v in Z.items()}
    n = len(Y)
    out = {}
    for k, (p1, lno_ok, within, mr) in vec.items():
        out[k] = dict(p_at_1=round(float(p1.mean()), 4), lno_top1=round(float(lno_ok.mean()), 4),
                      within_lno_top1_cond=round(float(within[lno_ok].mean()), 4),
                      mean_rank_of_gold=round(mr, 3))
    rng = np.random.default_rng(seed)
    a = vec[ANCHOR]
    for k in (ARM, SEED):
        b, dp, dw = vec[k], np.empty(n_boot), np.empty(n_boot)
        for i in range(n_boot):
            j = rng.integers(0, n, n)
            dp[i] = b[0][j].mean() - a[0][j].mean()
            dw[i] = b[2][j][b[1][j]].mean() - a[2][j][a[1][j]].mean()
        out[k]["paired_bootstrap"] = {
            "n_boot": n_boot,
            "delta_p_at_1_pt": round(float(100 * dp.mean()), 2),
            "ci95_p_at_1_pt": [round(float(np.percentile(100 * dp, 2.5)), 2),
                               round(float(np.percentile(100 * dp, 97.5)), 2)],
            "delta_within_cond_pt": round(float(100 * dw.mean()), 2),
            "ci95_within_cond_pt": [round(float(np.percentile(100 * dw, 2.5)), 2),
                                    round(float(np.percentile(100 * dw, 97.5)), 2)]}
    return out


# ── E. lambda 축의 대리 실험 = 훈련 곡선 ─────────────────────────────────────
def val_curve(nb_name):
    """노트북에 남은 Trainer 표에서 (step, val micro, empty rate)를 뽑는다.

    두 런의 열 순서는 동일하다 — Step · Training Loss · Validation Loss · micro · macro ·
    sample · empty_rate · anchor_weighted (`14_01`은 뒤에 `Lno`·표적 열이 더 붙는다).
    """
    nb = json.loads((NB / f"{nb_name}.ipynb").read_text(encoding="utf-8"))
    best = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            h = o.get("data", {}).get("text/html")
            if not h:
                continue
            h = "".join(h) if isinstance(h, list) else h
            rows = [[re.sub("<.*?>", "", x).strip()
                     for x in re.findall(r"<t[dh].*?>(.*?)</t[dh]>", r, re.S)]
                    for r in re.findall(r"<tr>(.*?)</tr>", h, re.S)]
            rows = [r for r in rows if len(r) >= 8 and r[0].isdigit()]
            if len(rows) > len(best):
                best = rows
    return [(int(r[0]), float(r[3]), float(r[6])) for r in best]


def lambda_proxy():
    arm, anc = val_curve("14_01_HierLoss_MCLoss"), val_curve("11_01_CleanData_Recipe")
    assert len(arm) == len(anc) == 24, (len(arm), len(anc))
    assert [a[0] for a in arm] == [a[0] for a in anc], "두 런의 eval step이 다르다"
    d = np.array([100 * (b[1] - a[1]) for a, b in zip(anc, arm)])
    return dict(
        points=[{"step": a[0], "epoch": round(a[0] / 1576, 1),
                 "anchor_micro": a[1], "arm_micro": b[1], "delta_pt": round(float(x), 2),
                 "anchor_empty": a[2], "arm_empty": b[2]}
                for a, b, x in zip(anc, arm, d)],
        mean_delta_first_3ep_pt=round(float(d[:6].mean()), 2),
        mean_delta_after_pt=round(float(d[6:].mean()), 2),
        mean_delta_last_6_pt=round(float(d[-6:].mean()), 2),
        n_positive_points=int((d > 0).sum()))


# ── F. 갈래 1 산술 — 기반 손실을 BCE로 바꿨을 때의 천장 ──────────────────────
def micro_cells(Zn, Y, pos_group):
    P = Zn >= 0.0
    tp, fp, fn = int((P & Y).sum()), int((P & ~Y).sum()), int((~P & Y).sum())
    return dict(micro=round(2 * tp / (2 * tp + fp + fn), 4), tp=tp, fp=fp, fn=fn,
                fp_cross_lno=int((P & ~Y & ~pos_group).sum()),
                fp_within_lno=int((P & ~Y & pos_group).sum()))


def main():
    lm = json.load(open(OUT / "label_mappings.json",
                        encoding="utf-8"))
    ls = LabelSpace(lm["id2mno"], lm["mno2lno"], NUM_LABELS)
    lno = torch.as_tensor(np.asarray(ls.lno_idx), dtype=torch.long)
    group_loss = make_group_loss(lno)

    Y, keep, n_old = load_axis()
    n = len(Y)
    T = torch.as_tensor(Y.astype(np.float64))
    pos_group = ls.to_lno(Y)[:, ls.lno_idx]
    Z = {k: load_logits(t, keep, n_old, n) for k, t in TAGS.items()}

    # verify — lambda 산출에 쓰인 두 항의 크기가 이 구현으로 재현되는 범위인가
    assert abs(LAM - L_FOCAL_INIT / L_GROUP_INIT) < 1e-6, "lambda가 기록된 두 항의 비와 다르다"

    rows, lam_bce = budget(Z, T, lno, group_loss)
    tch = touched(Z[ANCHOR], T, lno, group_loss)
    sat = saturation(Z[ANCHOR], T, lno)
    geo = geometry(Z, Y, pos_group)
    rk = rank_localization(Z, Y, ls)
    curve = lambda_proxy()
    cells = {k: micro_cells(v, Y, pos_group) for k, v in Z.items()}

    init_focal = [v["grad_ratio_focal_base"] for k, v in rows.items() if k.startswith("init")]
    init_bce = [v["grad_ratio_bce_base"] for k, v in rows.items() if k.startswith("init")]

    print(f"정리 test {n:,} · lambda {LAM:.6f}(초기화 값 일치) · lambda_bce(같은 규칙) {lam_bce:.4f}")
    print("\nA. 기울기 예산 — group/base 비")
    print(f"  초기화(스케일 스윕): focal 기반 {min(init_focal):.2f}~{max(init_focal):.2f}x"
          f" · BCE 기반 {min(init_bce):.2f}~{max(init_bce):.2f}x")
    for k in TAGS:
        r = rows[k]
        print(f"  {k:>15} 종점: focal 기반 {r['grad_ratio_focal_base']:.2f}x"
              f"(몫 {r['grad_share_focal_base']:.1%}) · BCE 기반 {r['grad_ratio_bce_base']:.2f}x"
              f"(몫 {r['grad_share_bce_base']:.1%}) · 손실 값 비 {r['value_ratio_focal_base']:.2f}x")
    print(f"  앵커 종점의 focal 손실 {rows[ANCHOR]['focal']:.6f} 대 arm 종점 {rows[ARM]['focal']:.6f}"
          f" — arm이 앵커의 목적함수에서도 나쁘다")
    print("  그룹 항이 닿는 로짓의 평균 |grad| (앵커 종점)")
    for k, v in tch.items():
        print(f"    {k:>10} n={v['n']:>7,} focal {v['focal']:.3e} · lam*group {v['lam_group']:.3e}"
              f" → {v['ratio']:.1f}배")

    print("\nB. 그룹 목적의 포화도(앵커 종점)")
    print(f"  양성 그룹 {sat['pos_groups']:,} 중 이미 충족 {sat['pos_satisfied_rate']:.1%}"
          f" · 음성 그룹 {sat['neg_groups']:,} 중 {sat['neg_satisfied_rate']:.1%}")
    print(f"  기울기 총량 중 이미 충족된 그룹의 몫 {sat['grad_share_on_satisfied']:.1%}")

    print("\nC. 로짓 지오메트리(앵커 대비 평균 로짓 이동)")
    for k in ("true_label", "sibling_in_pos_lno", "outside_pos_lno"):
        v = geo[k]
        print(f"  {k:>19} n={v['n']:>9,} 앵커 {v['anchor_mean_logit']:+7.3f}"
              f" · {ARM} {v[ARM]:+6.3f} · {SEED} {v[SEED]:+6.3f}")
    print(f"  로짓 sd {geo['logit_sd']}")

    print("\nD. 순위 손실의 위치")
    for k in TAGS:
        v = rk[k]
        print(f"  {k:>15} P@1 {v['p_at_1']:.4f} · Lno top-1 {v['lno_top1']:.4f}"
              f" · 조건부 within {v['within_lno_top1_cond']:.4f} · 정답 평균 순위"
              f" {v['mean_rank_of_gold']:.3f}")
    for k in (ARM, SEED):
        b = rk[k]["paired_bootstrap"]
        print(f"  {k:>15} ΔP@1 {b['delta_p_at_1_pt']:+.2f}pt {b['ci95_p_at_1_pt']}"
              f" · Δwithin {b['delta_within_cond_pt']:+.2f}pt {b['ci95_within_cond_pt']}")

    print(f"\nE. lambda 축 대리 실험 — val 곡선 24지점 중 양수 {curve['n_positive_points']}개")
    print(f"  1~3 epoch(그룹 항 몫이 낮은 국면) 평균 {curve['mean_delta_first_3ep_pt']:+.2f}pt"
          f" · 이후 {curve['mean_delta_after_pt']:+.2f}pt · 최종 6지점"
          f" {curve['mean_delta_last_6_pt']:+.2f}pt")

    print("\nF. 갈래 1 산술 — 기반 손실을 BCE로")
    for k in TAGS:
        c = cells[k]
        print(f"  {k:>15} micro {c['micro']:.4f} · FP {c['fp']:,}"
              f"(cross {c['fp_cross_lno']:,}/within {c['fp_within_lno']:,}) · FN {c['fn']:,}")
    b, a = cells[BCE], cells[ANCHOR]
    free = 2 * b["tp"] / (2 * b["tp"] + b["fp"] - 63 + b["fn"])
    print(f"  BCE 기반선 {100*(b['micro']-a['micro']):+.2f}pt · 표적 효과(cross FP −63)를 공짜로"
          f" 얹어도 {free:.4f} = 앵커 대비 {100*(free-a['micro']):+.2f}pt")
    ceiling = 2 * a["tp"] / (2 * a["tp"] + a["fp"] - 63 + a["fn"])
    print(f"  focal 기반에서 같은 계산 = {ceiling:.4f}(앵커 대비 {100*(ceiling-a['micro']):+.2f}pt)"
          f" — 이 축이 표적에 남긴 효과의 상한")

    payload = {
        "note": "계층 손실(MCLoss) 실패 기제. lambda는 초기화 시점의 손실 '값'을 맞췄을 뿐 "
                "기울기를 맞추지 않았고, focal이 alpha*(1-p)^gamma/188로 자기소멸하는 사이 "
                "그룹 항이 종점 기울기의 다수를 가져간다. 표적 크기가 아니라 전달 기제의 실패다.",
        "n": n, "split": SPLIT, "tau": 0.5, "anchor": ANCHOR, "arm": ARM,
        "lambda": LAM, "lambda_bce_same_rule": round(lam_bce, 4),
        "script": "scripts/hierarchy_loss_grad_budget.py",
        "verify": "그룹 항 구현 == 14_01 5번 셀 · lambda == 기록된 두 항의 비 · "
                  "두 런의 eval step 24개 일치 · 로짓 행 축 == doc_ids_clean_test.json",
        "gradient_budget": rows,
        "gradient_on_touched_logits": tch,
        "group_objective_saturation": sat,
        "logit_geometry": geo,
        "rank_localization": rk,
        "lambda_proxy_training_curve": curve,
        "operating_cells": cells,
        "branch_arithmetic": {
            "bce_base_vs_anchor_pt": round(100 * (b["micro"] - a["micro"]), 2),
            "bce_base_plus_free_target_micro": round(free, 4),
            "bce_base_plus_free_target_vs_anchor_pt": round(100 * (free - a["micro"]), 2),
            "focal_base_plus_free_target_micro": round(ceiling, 4),
            "focal_base_plus_free_target_vs_anchor_pt": round(100 * (ceiling - a["micro"]), 2),
            "note": "'공짜'는 작동점 일치에서 실측된 표적 효과(cross FP −63)만 얹고 "
                    "부수 피해(within FP·FN)를 0으로 둔 도달 불가 가정이다. 그 가정에서도 "
                    "운영 판정선 +0.6pt는 물론 형식 하한 +0.4pt에 못 미친다.",
        },
    }
    (OUT / "hierarchy_loss_grad_budget.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT / 'hierarchy_loss_grad_budget.json'}")


if __name__ == "__main__":
    main()
