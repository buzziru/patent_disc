"""손실 주입 Trainer — 노트북별 `compute_loss` 변형을 하나로 통합.

HF `Trainer`의 `compute_loss`만 오버라이드해 주입된 `loss_fn`으로 손실을 계산한다.
손실 종류·하이퍼파라미터는 config에서 `build_loss`로 해석해 주입하므로, 손실을 바꿔도 이 클래스는 불변이다.

`model_accepts_loss_kwargs`는 `False`로 되돌린다 — transformers는 이 플래그가 참이고
`num_items_in_batch`가 잡히면 "모델이 토큰 수로 이미 정규화했다"고 보고 `training_step`에서
`loss / gradient_accumulation_steps`를 건너뛴다. 주입 손실은 배치 평균이라 그 정규화가 필요하고,
건너뛰면 `grad_accum`배만큼 그래디언트(=실효 lr)가 커진다.
"""

from transformers import Trainer


class LossTrainer(Trainer):
    """`loss_fn`(nn.Module: (logits, targets) -> scalar)을 주입받아 멀티라벨 손실을 계산한다."""

    def __init__(self, *args, loss_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.model_accepts_loss_kwargs = False   # grad_accum 정규화를 Trainer에 맡긴다(위 docstring)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 라벨만 빼고 그대로 넘긴다 — 백본이 요구하는 입력 키(token_type_ids 등)를 하드코딩하지 않는다.
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        loss = self.loss_fn(outputs.logits, labels.float())
        return (loss, outputs) if return_outputs else loss
