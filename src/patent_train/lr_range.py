"""LR range test — 발산 임계 진단.

lr을 여러 개 각각 풀런해 비교하는 대신, **한 번의 짧은 훈련 안에서 lr을 지수적으로 올려**
손실이 언제 무너지는지 본다(Smith, *Cyclical Learning Rates*). 답하는 질문은 "어느 lr이
12에폭 뒤 가장 높은 micro를 내는가"가 아니라 **"어디서 훈련이 깨지는가"** 다.

[ADR-0011](../../docs/adr/0011-resource-constrained-methodology.md)이 「전이 가능한 lr
판정법」으로 지목한 horizon-분리 프로브다. 고정-에폭 삼각 감쇠 프로브는 진행량이 peak lr에
비례해 **고lr을 구조적으로 우대**하므로 lr 서열을 판별하지 못하는데, "발산하는가"는 총 스텝
수와 무관한 성질이라 이 편향에 걸리지 않는다.

곡선은 세 구간으로 읽는다 — ① lr이 너무 작아 손실이 평탄한 구간, ② 가파르게 하강하는 학습
구간, ③ 최저점을 지나 치솟는 발산 구간. ②→③ 전환점이 발산 임계이며, 통상 운영 lr은 그보다
3~10배 아래에 둔다.

**판정은 절대값이 아니라 상대 위치로 한다.** 같은 자로 이미 안전이 실증된 조합(len512 ·
eff128 · lr 4.8e-4는 `11_01`이 12에폭 완주)을 함께 재고, 대상 조합에서 운영 lr이 전환점 대비
같은 위치에 있는지 본다. 경험칙을 이 프로젝트의 실측으로 대체한다.

    runner = TrainingRunner(cfg).setup()
    res = run_lr_range_test(runner)          # 모델을 발산까지 밀어 넣는다 — 이후 재사용 불가
    res.summary(operating_lr=4.8e-4)
    res.save()

⚠️ 프로브가 끝난 모델 가중치는 버린다. 여기서 이어서 훈련하지 않는다.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from transformers import TrainerCallback, TrainingArguments

from .losses import build_loss
from .trainer import LossTrainer

START_LR = 1e-6          # 관측 시작 — ① 평탄 구간이 보이도록 충분히 낮게
END_LR = 1e-2            # 관측 끝 — ③ 발산 구간까지 확실히 넘도록 높게
NUM_STEPS = 300          # 4자릿수를 훑는다(스텝당 약 x1.031)
SMOOTH_BETA = 0.98       # 원시 train loss는 배치마다 튄다 — EMA로 전환점을 드러낸다
DIVERGE_FACTOR = 4.0     # 평활 손실이 최저의 이 배수를 넘으면 중단(표준 LR finder 규칙)
MIN_STEPS_BEFORE_STOP = 15   # 초반 잡음으로 조기 중단되지 않도록


class ExpLrProbe(TrainerCallback):
    """스텝마다 lr을 지수적으로 올리고 손실을 기록한다.

    `lr_scheduler_type="constant"`로 두고 optimizer의 `param_groups`에 직접 쓴다 — HF는
    `on_step_begin` → `optimizer.step()` → `lr_scheduler.step()` 순서라, 여기서 쓴 값이 그
    스텝의 실효 lr이 되고 스케줄러가 되돌린 값은 다음 `on_step_begin`이 다시 덮는다.
    콜백이 동작하지 않으면 런 전체가 `START_LR`로 돌아 곡선이 평탄해진다 — 조용히 틀린 결과가
    아니라 눈에 띄는 실패로 나타나도록 base lr을 시작값에 맞춰 둔다.
    """

    def __init__(self, start_lr=START_LR, end_lr=END_LR, num_steps=NUM_STEPS,
                 beta=SMOOTH_BETA, diverge_factor=DIVERGE_FACTOR):
        self.start_lr, self.end_lr, self.num_steps = start_lr, end_lr, num_steps
        self.beta, self.diverge_factor = beta, diverge_factor
        self.mult = (end_lr / start_lr) ** (1.0 / max(1, num_steps - 1))
        self.lrs, self.losses, self.smoothed, self.applied = [], [], [], []
        self._cur = start_lr
        self._ema = None
        self._best = float("inf")
        self.stopped_at = None

    def on_step_begin(self, args, state, control, optimizer=None, **kw):
        optimizer = optimizer or kw.get("optimizer")
        if optimizer is None:                      # 콜백 계약이 바뀌면 조용히 넘어가지 않는다
            raise RuntimeError("on_step_begin에 optimizer가 오지 않았다: lr 주입 불가")
        self._cur = self.start_lr * self.mult ** state.global_step
        for g in optimizer.param_groups:
            g["lr"] = self._cur
        # 되읽어 실제로 반영됐는지 확인(첫 스텝에서 계약 위반을 끊는다)
        self.applied.append(float(optimizer.param_groups[0]["lr"]))

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or "loss" not in logs:
            return
        loss = float(logs["loss"])
        self._ema = loss if self._ema is None else self.beta * self._ema + (1 - self.beta) * loss
        n = len(self.losses) + 1
        sm = self._ema / (1 - self.beta ** n)       # bias 보정
        self.lrs.append(self._cur)
        self.losses.append(loss)
        self.smoothed.append(sm)
        self._best = min(self._best, sm)
        if n >= MIN_STEPS_BEFORE_STOP and sm > self._best * self.diverge_factor:
            self.stopped_at = self._cur
            control.should_training_stop = True


@dataclass
class LrRangeResult:
    """프로브 산출 — 곡선과 그 위에서 읽은 세 지점."""
    tag: str
    max_len: int
    eff_batch: int
    lrs: list
    losses: list
    smoothed: list
    lr_at_min_loss: float          # ②→③ 전환점 = 발산 임계
    lr_at_steepest: float          # ② 구간에서 log lr당 하강이 가장 가파른 지점
    lr_at_stop: float | None       # 평활 손실이 최저의 DIVERGE_FACTOR 배를 넘은 지점
    min_smoothed: float
    out_dir: str = field(default="", repr=False)

    def position_of(self, operating_lr: float) -> dict:
        """운영 lr이 전환점 대비 어디에 있는가 — 판정에 쓰는 상대 위치."""
        return {
            "operating_lr": operating_lr,
            "lr_at_min_loss": self.lr_at_min_loss,
            "headroom_x": round(self.lr_at_min_loss / operating_lr, 2),   # 전환점이 운영 lr의 몇 배
            "log10_margin": round(float(np.log10(self.lr_at_min_loss / operating_lr)), 3),
        }

    def summary(self, operating_lr: float | None = None) -> dict:
        print(f"[{self.tag}] len{self.max_len} · eff_batch {self.eff_batch} · {len(self.lrs)} step")
        print(f"  관측 구간 {self.lrs[0]:.2e} ~ {self.lrs[-1]:.2e}"
              f"{' (발산 중단)' if self.lr_at_stop else ''}")
        print(f"  ② 최급하강 lr {self.lr_at_steepest:.3e}")
        print(f"  ②→③ 전환점(발산 임계) lr {self.lr_at_min_loss:.3e}"
              f" · 평활 손실 최저 {self.min_smoothed:.5f}")
        if self.lr_at_stop:
            print(f"  손실이 최저의 {DIVERGE_FACTOR}배를 넘은 지점 {self.lr_at_stop:.3e}")
        pos = self.position_of(operating_lr) if operating_lr else None
        if pos:
            print(f"  운영 lr {operating_lr:.2e} → 전환점이 그 **{pos['headroom_x']}배**"
                  f" (log10 여유 {pos['log10_margin']:+.2f})")
        return {**self.to_dict(), "position": pos}

    def to_dict(self) -> dict:
        return {
            "tag": self.tag, "max_len": self.max_len, "eff_batch": self.eff_batch,
            "n_steps": len(self.lrs),
            "lr_at_steepest": self.lr_at_steepest,
            "lr_at_min_loss": self.lr_at_min_loss,
            "lr_at_stop": self.lr_at_stop,
            "min_smoothed": self.min_smoothed,
            "curve": {"lr": self.lrs, "loss": self.losses, "smoothed": self.smoothed},
        }

    def save(self, out_dir: str | None = None) -> str:
        out_dir = out_dir or self.out_dir
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, f"lr_range_{self.tag}.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[save] {fp}")
        return fp


def _read_curve(probe: ExpLrProbe) -> tuple:
    """평활 곡선에서 최급하강 지점과 최저점(=전환점)을 읽는다."""
    lr = np.asarray(probe.lrs)
    sm = np.asarray(probe.smoothed)
    i_min = int(sm.argmin())
    # log lr 축의 기울기 — ② 구간(최저점 이전)에서 가장 음수인 지점
    if i_min >= 2:
        slope = np.gradient(sm[: i_min + 1], np.log10(lr[: i_min + 1]))
        i_steep = int(slope.argmin())
    else:
        i_steep = i_min
    return float(lr[i_steep]), float(lr[i_min]), float(sm[i_min])


def run_lr_range_test(runner, *, start_lr=START_LR, end_lr=END_LR, num_steps=NUM_STEPS,
                      beta=SMOOTH_BETA, diverge_factor=DIVERGE_FACTOR) -> LrRangeResult:
    """`runner`의 모델·데이터로 LR range test를 돌리고 곡선을 반환한다.

    최종 런과 **모든 것을 동일하게** 두고 스케줄만 바꾼다(백본·헤드 초기화·손실·배치·데이터·
    샘플러·dtype). eval·save·early stop·wandb는 끄고 매 스텝 손실을 기록한다.

    ⚠️ 모델은 발산 구간까지 밀려 망가진다 — 프로브 뒤에 훈련·평가하지 않는다.
    """
    cfg = runner.cfg
    if runner.model is None:
        raise RuntimeError("모델이 없다 — load_model()을 먼저 호출할 것")

    probe = ExpLrProbe(start_lr, end_lr, num_steps, beta, diverge_factor)
    print(f"[lr-range] {start_lr:.1e} → {end_lr:.1e} · {num_steps} step"
          f" (스텝당 x{probe.mult:.4f}) · eff_batch {cfg.eff_batch} · len{cfg.max_len}")
    print(f"[lr-range] 문서 {num_steps * cfg.eff_batch:,}건"
          f" ≈ {num_steps * cfg.eff_batch / len(runner.data.dataset['train']):.3f} epoch")

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        learning_rate=start_lr,                      # 콜백 실패 시 평탄 곡선으로 드러나도록
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="constant",                # 감쇠 없음 — lr 축은 콜백이 지배한다
        warmup_ratio=0.0,                            # warmup은 lr 축을 흐린다
        per_device_train_batch_size=cfg.micro_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        train_sampling_strategy="group_by_length",   # 최종 런과 동일
        remove_unused_columns=False,
        max_steps=num_steps,
        bf16=runner.backbone.bf16,
        eval_strategy="no",
        save_strategy="no",
        logging_dir=cfg.logging_dir,
        logging_steps=1,                             # 스텝마다 손실을 받아야 곡선이 생긴다
        report_to="none",
    )
    trainer = LossTrainer(
        model=runner.model,
        args=args,
        train_dataset=runner.data.dataset["train"],
        data_collator=runner.data.collator,
        processing_class=runner.data.tokenizer,
        callbacks=[probe],
        loss_fn=build_loss(cfg.loss, **cfg.loss_params),
    )
    trainer.train()

    if not probe.losses:
        raise RuntimeError("손실이 하나도 기록되지 않았다 — logging_steps·콜백 계약 확인")
    # 콜백이 실제로 lr을 주입했는지 — 되읽은 값이 계획과 일치해야 한다
    assert np.allclose(probe.applied[: len(probe.lrs)], probe.lrs[: len(probe.applied)],
                       rtol=1e-6), "optimizer에 주입한 lr이 계획과 다르다"
    steep, at_min, min_sm = _read_curve(probe)
    return LrRangeResult(
        tag=cfg.tag, max_len=cfg.max_len, eff_batch=cfg.eff_batch,
        lrs=probe.lrs, losses=probe.losses, smoothed=probe.smoothed,
        lr_at_min_loss=at_min, lr_at_steepest=steep, lr_at_stop=probe.stopped_at,
        min_smoothed=min_sm, out_dir=str(Path(cfg.out_path).parent),
    )


def compare(control: LrRangeResult, target: LrRangeResult, operating_lr: float,
            tolerance_x: float = 1.5) -> dict:
    """검증된 조합(control)과 대상 조합(target)에서 운영 lr의 상대 위치를 대면시킨다.

    control은 이 lr로 풀런이 완주한 조합이어야 한다 — 그 상대 위치가 "안전하다고 실증된
    위치"의 기준이 된다. 판정은 절대 임계가 아니라 두 위치의 비로 한다.
    """
    c, t = control.position_of(operating_lr), target.position_of(operating_lr)
    ratio = t["headroom_x"] / c["headroom_x"]
    if ratio >= 1.0 / tolerance_x:
        verdict = "여유 있음: 운영 lr 그대로"
    elif t["headroom_x"] > 1.0:
        verdict = f"경계: control과 같은 위치가 되도록 lr을 {ratio:.2f}배로 낮출 것"
    else:
        verdict = "위험: 운영 lr이 전환점 이상. 이 레시피를 대상 조합에 쓰지 않는다"
    out = {
        "operating_lr": operating_lr, "tolerance_x": tolerance_x,
        "control": {"tag": control.tag, "max_len": control.max_len, **c},
        "target": {"tag": target.tag, "max_len": target.max_len, **t},
        "headroom_ratio_target_over_control": round(ratio, 3),
        "suggested_lr": round(operating_lr * min(1.0, ratio), 8),
        "verdict": verdict,
    }
    print(f"control len{control.max_len}: 전환점 {c['lr_at_min_loss']:.3e}"
          f" = 운영 lr의 {c['headroom_x']}배")
    print(f"target  len{target.max_len}: 전환점 {t['lr_at_min_loss']:.3e}"
          f" = 운영 lr의 {t['headroom_x']}배")
    print(f"상대 위치 비 {out['headroom_ratio_target_over_control']:.3f} → {verdict}")
    if out["suggested_lr"] < operating_lr:
        print(f"제안 lr {out['suggested_lr']:.3e}")
    return out
