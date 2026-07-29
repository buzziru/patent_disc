"""백본 스펙 — model_name·rev·데이터셋·헤드가 함께 움직이는 묶음.

인코더를 바꾸면 이 넷이 한꺼번에 바뀐다 — 각 백본은 자기 토크나이저로 만든 토큰화 데이터셋을 쓰고,
`rev`로 가중치·토크나이저 버전을 고정한다. 그래서 훈련 레시피(`TrainConfig`)와 분리해 레지스트리로 둔다.
새 인코더로 실험을 옮길 때 `BACKBONES`에 스펙 한 줄만 추가하면 config 코드는 불변이다(ADR-0009: 남은 레버는 모델).

`attn_implementation`·`classifier_dropout`은 모델 구성 인자라 백본에 함께 둔다(로컬/CPU 디버그는 "eager").
`bf16`도 같은 이유로 여기 있다 — flash-attention-2는 반정밀도 입력을 요구하므로 어텐션 구현과 한 몸이고,
레시피(lr·배치처럼 실험이 고르는 축)가 아니라 백본이 결정하는 제약이다.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Backbone:
    """인코더 백본 + 대응 토큰화 데이터셋 스펙."""
    key: str
    model_name: str
    rev: str
    dataset_id: str
    num_labels: int = 188
    attn_implementation: str = "flash_attention_2"
    classifier_dropout: float = 0.5
    bf16: bool = True               # flash-attention-2 필수 조건. "eager" 백본(CPU 디버그)만 False


# 레지스트리 — 백본 추가 시 여기에 한 줄. runner가 config의 backbone 키로 해석한다.
BACKBONES = {
    "axenc": Backbone(
        key="axenc",
        model_name="skt/A.X-Encoder-base",
        rev="9708f9c404ace91efd25c06fac2d73413616f4ef",
        dataset_id="ingyoun/patent-clean-text-modernbert-tokenized",
    ),
    "axenc_tapt": Backbone(
        key="axenc_tapt",
        model_name="ingyoun/A.X-patent-tapt-mlm",
        rev="62818c2595513a03f834c39a329c375153bc2661",
        dataset_id="ingyoun/patent-clean-text-modernbert-tokenized",
    ),
}


def get_backbone(key: str) -> Backbone:
    """레지스트리 키로 백본 스펙을 조회한다."""
    if key not in BACKBONES:
        raise KeyError(f"미등록 백본: {key!r} — 등록된 키: {sorted(BACKBONES)}")
    return BACKBONES[key]
