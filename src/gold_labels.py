"""정답 라벨 — 분석 스크립트가 데이터셋 없이 읽는 평가 축.

가공 데이터셋은 공개 배포하지 않으므로(`docs/data/data-pipeline.md`「재현 경로」),
분석에 필요한 비텍스트 컬럼만 `output/gold_labels_{split}.json`으로 저장소에 동봉한다.
담기는 것은 `document_id`·`label_ids`·`length_bin` 셋뿐이며 원문 텍스트는 포함하지 않는다.

`GoldSplit`은 `ds["label_ids"]`·`len(ds)` 형태의 접근을 그대로 받아, 호출부가
`load_dataset(...)`을 `load_gold(...)`로 바꾸기만 하면 되게 한다.
"""

import json
from pathlib import Path

COLUMNS = ("document_id", "label_ids", "length_bin")


class GoldSplit:
    """한 split의 정답 축. 데이터셋 객체와 같은 방식으로 컬럼을 읽는다."""

    def __init__(self, payload: dict):
        self._p = payload

    def __getitem__(self, col: str):
        if col not in COLUMNS:
            raise KeyError(
                f"{col!r}은 동봉 대상이 아니다 — 담긴 컬럼은 {COLUMNS}. "
                "텍스트 필드가 필요하면 데이터셋을 재생성한다(docs/data/data-pipeline.md)."
            )
        return self._p[col]

    def __len__(self) -> int:
        return self._p["n"]

    @property
    def column_names(self) -> list:
        return list(COLUMNS)


def load_gold(split: str, out_dir) -> GoldSplit:
    """`output/gold_labels_{split}.json`을 읽는다. split은 train/val/test."""
    path = Path(out_dir) / f"gold_labels_{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split"] == split, f"{path.name}의 split이 {payload['split']}다"
    assert len(payload["document_id"]) == payload["n"], f"{path.name}의 행 수가 n과 다르다"
    return GoldSplit(payload)
