"""훈련 파사드 — 단계별 메서드를 config 하나로 묶는다.

노트북은 이 클래스만 임포트하고, 단계를 셀 단위로 나눠 실행한다(중간 점검·부분 재실행이 쉽도록).

    runner = TrainingRunner(cfg)
    runner.load_data()        # 토크나이저 + 원본 데이터셋
    runner.prepare_data()     # max_len 절단(캐시)
    runner.load_model()       # 백본 스펙으로 분류기
    runner.build_trainer()    # TrainingArguments + 손실 주입 Trainer
    runner.train()            # 훈련만
    m = runner.evaluate("test")     # 평가만 — 메트릭 dict 반환(출력은 호출부에서)
    runner.save_metrics()     # 보관 중인 메트릭 → JSON
    runner.push_to_hub()      # 모델·토크나이저는 Hub로(로컬 저장은 save_model(), 통상 불필요)
    runner.predict_logits("test")   # 로짓 덤프(error_analysis 인계)

`setup()`은 앞의 세 로드 단계를 잇는 축약일 뿐이다. env·시드·백본 해석은 `__init__`에서 끝난다.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
from transformers import TrainingArguments, EarlyStoppingCallback

from .backbones import get_backbone
from .config import TrainConfig
from .data import PatentData
from .env import set_seed, setup_env
from .losses import build_loss
from .metrics import make_compute_metrics
from .model import build_model
from .trainer import LossTrainer


class TrainingRunner:
    """훈련 세션 — 단계별 메서드가 각각 하나의 책임만 갖는다."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        # HF_HOME은 cfg.hf_cache에서 온다 — 모델 다운로드(HF_HOME)와 데이터셋 캐시(cache_dir)가
        # 같은 경로를 보도록 SSOT를 하나로 둔다. 이미 bake된 값이 있으면 setdefault가 그것을 보존한다.
        env = {"HF_HOME": cfg.hf_cache}
        # 노트북명은 wandb code saving용 — wandb가 이 경로로 파일을 실제로 읽으므로 절대경로로 준다.
        # 지정 시에만 주입(미지정이면 wandb가 노트북명 탐지 실패를 경고하고 code saving을 끈다).
        if cfg.notebook_name:
            env["WANDB_NOTEBOOK_NAME"] = os.path.abspath(cfg.notebook_name)
        setup_env(env)
        set_seed(cfg.seed)
        self.backbone = get_backbone(cfg.backbone)
        self.data = PatentData(cfg, self.backbone)
        self.model = None
        self.trainer = None
        self.metrics = {}          # split -> 평가 결과

    @property
    def test_metrics(self):
        """test split 평가 결과(없으면 None)."""
        return self.metrics.get("test")

    # ── 구성 단계 ─────────────────────────────────────────────────────────

    def load_data(self) -> "TrainingRunner":
        """토크나이저와 원본 토큰화 데이터셋을 로드한다(prep 캐시가 있으면 원본은 생략)."""
        self.data.load_tokenizer().load()
        return self

    def prepare_data(self) -> "TrainingRunner":
        """원본에 `max_len` 절단을 적용한다(결과를 prep 캐시에 저장·재사용)."""
        self.data.prepare()
        return self

    def load_model(self) -> "TrainingRunner":
        """188-way 멀티라벨 분류기를 구성한다(cfg.checkpoint가 있으면 그 소스에서 복원)."""
        self.model = build_model(self.backbone, checkpoint=self.cfg.checkpoint)
        src = self.cfg.checkpoint or f"{self.backbone.model_name}@{self.backbone.rev[:8]}(신규 헤드)"
        print(f"[model] {src}")
        return self

    def setup(self) -> "TrainingRunner":
        """load_data → prepare_data → load_model 축약."""
        return self.load_data().prepare_data().load_model()

    def build_trainer(self) -> "TrainingRunner":
        """TrainingArguments를 짜고 config가 지정한 손실을 주입한 Trainer를 만든다.

        config의 시간 축은 에폭 단위이고, HF Trainer가 요구하는 step·eval 횟수 환산은 여기서 한다.
        `cfg.checkpoint`가 설정된 추론 경로는 훈련 부속(wandb·early stop·save·train_dataset)을 모두
        끄고 로짓 덤프에 필요한 최소 구성만 만든다.
        """
        if self.cfg.checkpoint is not None:
            return self._build_inference_trainer()
        cfg = self.cfg
        steps_per_epoch = math.ceil(len(self.data.dataset["train"]) / cfg.eff_batch)
        eval_steps = math.ceil(steps_per_epoch / cfg.evals_per_epoch)
        # EarlyStoppingCallback의 patience 단위는 **eval 횟수**다. 설정은 에폭으로 받아 여기서 환산한다.
        patience = cfg.early_stop_epochs * cfg.evals_per_epoch
        print(f"[schedule] {steps_per_epoch} step/epoch | eval·save {eval_steps} step마다"
              f"({cfg.evals_per_epoch}회/epoch) | early stop {cfg.early_stop_epochs} epoch(patience={patience} eval)")

        args = TrainingArguments(
            output_dir=cfg.output_dir,
            seed=cfg.seed,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            per_device_train_batch_size=cfg.micro_batch,
            per_device_eval_batch_size=cfg.eval_micro_batch,
            gradient_accumulation_steps=cfg.grad_accum,
            train_sampling_strategy="group_by_length",   # 유사 길이 배치로 padding 최소화
            remove_unused_columns=False,                 # custom collator가 키 선택 + length 컬럼 보존
            num_train_epochs=cfg.epochs,
            bf16=self.backbone.bf16,                     # 백본 제약(flash-attention-2는 반정밀도 필수)
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="no" if cfg.search else "steps",
            save_steps=eval_steps,
            save_total_limit=cfg.save_total_limit,
            logging_dir=cfg.logging_dir,
            logging_steps=50,
            metric_for_best_model="micro_f1",
            greater_is_better=True,
            load_best_model_at_end=not cfg.search,       # 풀런에서만 best 복원
            report_to="wandb",
            run_name=cfg.run_name,
        )
        self.trainer = LossTrainer(
            model=self.model,
            args=args,
            train_dataset=self.data.dataset["train"],
            eval_dataset=self.data.dataset["val"],
            data_collator=self.data.collator,
            processing_class=self.data.tokenizer,
            compute_metrics=make_compute_metrics(cfg.tau),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
            loss_fn=build_loss(cfg.loss, **cfg.loss_params),
        )
        return self

    def _build_inference_trainer(self) -> "TrainingRunner":
        """추론 전용 Trainer — 로짓 덤프에 필요한 최소 구성(훈련·wandb·early stop·save 없음).

        `predict_logits`가 덤프 동안 순차 샘플러를 강제하지만, 여기서도 `sequential`로 두어
        evaluate/predict를 직접 부르는 경우까지 행 순서를 보장한다.
        """
        cfg = self.cfg
        args = TrainingArguments(
            output_dir=cfg.output_dir,
            seed=cfg.seed,
            per_device_eval_batch_size=cfg.eval_micro_batch,
            train_sampling_strategy="sequential",        # eval·predict 로더의 행 순서 보장
            remove_unused_columns=False,                 # custom collator가 키 선택 + length 보존
            bf16=self.backbone.bf16,
            eval_strategy="no",
            save_strategy="no",
            logging_dir=cfg.logging_dir,
            report_to="none",                            # 추론엔 wandb 불필요
        )
        # eval_dataset은 predict가 데이터셋을 명시 인자로 받으므로 기본 참조용일 뿐이다.
        any_split = next(iter(self.data.dataset))
        self.trainer = LossTrainer(
            model=self.model,
            args=args,
            eval_dataset=self.data.dataset[any_split],
            data_collator=self.data.collator,
            processing_class=self.data.tokenizer,
            compute_metrics=make_compute_metrics(cfg.tau),
            loss_fn=build_loss(cfg.loss, **cfg.loss_params),
        )
        return self

    # ── 실행 단계 ─────────────────────────────────────────────────────────

    def train(self) -> "TrainingRunner":
        """훈련만 수행한다(추론 config에서는 거부)."""
        if self.cfg.checkpoint is not None:
            raise RuntimeError(
                "추론 config(cfg.checkpoint 설정)로 train() 호출 — 무의미한 자리값 lr로 파인튜닝된다. "
                "재훈련하려면 checkpoint 없이 레시피를 명시한 TrainConfig를 쓸 것"
            )
        self._require_trainer()
        self.trainer.train()
        return self

    def evaluate(self, split: str = "test") -> dict:
        """지정 split을 평가해 메트릭 딕셔너리를 반환한다(`self.metrics[split]`에도 보관).

        출력은 호출부(노트북)가 한다 — 여기서는 반환만.
        """
        self._require_trainer()
        metrics = self.trainer.evaluate(self.data.dataset[split], metric_key_prefix=split)
        self.metrics[split] = metrics
        return metrics

    def save_metrics(self) -> "TrainingRunner":
        """보관 중인 평가 메트릭(`self.metrics`, split 전체)을 JSON 한 파일로 저장한다."""
        cfg = self.cfg
        os.makedirs(cfg.out_path, exist_ok=True)
        fp = os.path.join(cfg.out_path, f"{cfg.tag}_metrics.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        print(f"[save] {fp}  splits={sorted(self.metrics)}")
        return self

    def save_model(self) -> "TrainingRunner":
        """모델·토크나이저를 로컬 `out_path`에 저장한다.

        통상 경로는 `push_to_hub()`다(팟을 지우면 로컬 사본은 사라진다) —
        볼륨 용량을 써서라도 로컬 사본이 필요한 경우에만 호출한다.
        """
        self._require_trainer()
        os.makedirs(self.cfg.out_path, exist_ok=True)
        self.trainer.save_model(self.cfg.out_path)
        self.data.tokenizer.save_pretrained(self.cfg.out_path)
        print(f"[save] {self.cfg.out_path}  best={self.trainer.state.best_metric}")
        return self

    def push_to_hub(self) -> "TrainingRunner":
        """모델·토크나이저를 `repo_final`로 push한다(팟을 지워도 남도록)."""
        self._require_trainer()
        self.trainer.model.push_to_hub(self.cfg.repo_final)
        self.data.tokenizer.push_to_hub(self.cfg.repo_final)
        print(f"[push] {self.cfg.repo_final}")
        return self

    def predict_logits(self, split: str, out_dir: str = None) -> np.ndarray:
        """지정 split의 로짓을 `logits_{tag}_{split}.npy`로 덤프한다(error_analysis 인계용).

        기본 저장 위치는 `out_path`가 아니라 그 **상위** 디렉터리다 — 하류 분석
        (`error_analysis.load_logits`)이 여러 런의 로짓을 한 디렉터리에서 tag로 구분해 읽기 때문이다.
        런별 디렉터리에 두려면 `out_dir`을 명시한다.

        덤프 동안만 샘플러를 순차로 되돌린다 — `train_sampling_strategy="group_by_length"`는
        **eval·predict 로더에도 적용**되어(`Trainer._get_eval_sampler`) 반환 행이 길이 그룹 순열로 나온다.
        평가 지표는 라벨을 같은 순서로 모으므로 영향이 없지만, 로짓 파일은 데이터셋 행 순서를 전제로
        정답·문서 id와 맞춰지므로 순열이 섞이면 하류 분석이 전부 어긋난다(순열은 복원 불가).
        """
        self._require_trainer()
        ds = self.data.dataset[split]
        args = self.trainer.args
        strategy, args.train_sampling_strategy = args.train_sampling_strategy, "sequential"
        try:
            out = self.trainer.predict(ds)
        finally:
            args.train_sampling_strategy = strategy
        logits = np.asarray(out.predictions)
        # 행 순서 보증 — 반환 라벨이 데이터셋 라벨과 행 단위로 같아야 한다.
        # 순열이 섞인 로짓은 하류에서 조용히 틀린 분석을 낳으므로 여기서 끊는다.
        assert np.array_equal(np.asarray(out.label_ids), np.asarray(ds["labels"], dtype=np.float32)), \
            f"{split} 로짓 행 순서가 데이터셋과 다르다 — eval 샘플러를 확인할 것"
        # predict가 함께 낸 지표를 보관 — 재덤프 시 체크포인트 정상 복원 대조에 쓴다(별도 evaluate 불필요).
        self.metrics[split] = out.metrics
        out_dir = out_dir or str(Path(self.cfg.out_path).parent)
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, f"logits_{self.cfg.tag}_{split}.npy")
        np.save(fp, logits)
        print(f"[dump] {fp}  shape={logits.shape}")
        return logits

    def _require_trainer(self) -> None:
        if self.trainer is None:
            raise RuntimeError("Trainer가 없다 — build_trainer()를 먼저 호출할 것")
