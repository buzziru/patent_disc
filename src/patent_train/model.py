"""모델 팩토리 — 백본 스펙으로 멀티라벨 헤드를 얹는다.

`AutoModelForSequenceClassification`에 분류 헤드(`num_labels` 출력)를 새로 얹는다.
`problem_type="multi_label_classification"`이지만 손실은 `LossTrainer`가 주입하므로 내장 BCE 경로는 우회된다.
모델 구성 인자(헤드·attn·num_labels)는 `Backbone`에서 온다 — config는 가중치 소스(`checkpoint`)만 정한다.
"""

import torch
from transformers import AutoModelForSequenceClassification

from .backbones import Backbone


def build_model(backbone: Backbone, checkpoint: str | None = None):
    """멀티라벨 분류기를 구성한다.

    `checkpoint`가 None이면 백본 사전학습분에서 헤드를 신규 초기화한다(훈련용, decoder.bias 드롭).
    주어지면 훈련된 분류기를 그 소스에서 복원한다(추론용) — Hub repo id든 로컬 디렉터리든
    `from_pretrained`가 동일하게 해석하므로 팟(캐시/로컬)·Colab(Hub 다운로드)을 한 코드로 덮는다.
    `revision`은 백본 사전학습분에만 붙인다 — 훈련된 repo에는 그 커밋 해시가 없다.
    """
    source = checkpoint or backbone.model_name
    kwargs = {} if checkpoint else {"revision": backbone.rev}   # 백본 가중치만 커밋 고정
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=source,
        num_labels=backbone.num_labels,
        problem_type="multi_label_classification",
        classifier_dropout=backbone.classifier_dropout,
        dtype=torch.float32,
        attn_implementation=backbone.attn_implementation,
        **kwargs,
    )
