"""데이터 — 사전 토크나이즈 데이터셋 로드·절단·동적 패딩.

`ingyoun/patent-clean-text-modernbert-tokenized`는 이미 다중핫 `labels`를 담은 토큰화 데이터셋이다.
`_prep`이 `max_len`으로 재절단(선두 <s> 유지, 꼬리 </s> 마감)하고 결과를 on-disk 캐시한다.
`MultiLabelCollator`가 배치별 동적 패딩 + float 라벨 텐서를 만든다.
splits는 `train`/`val`/`test`.

단계는 분리돼 있다 — `load_tokenizer()` → `load()`(원본) → `prepare()`(절단·캐시).
prep 캐시가 이미 있으면 원본이 필요 없으므로 `load()`는 다운로드를 건너뛴다.
"""

import os

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from .backbones import Backbone
from .config import TrainConfig


class MultiLabelCollator:
    """토크나이저 동적 패딩 + float 라벨. custom collator라 필요한 키만 골라 pad한다."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, feats):
        labels = torch.tensor([f["labels"] for f in feats], dtype=torch.float)
        keys = ("input_ids", "attention_mask")
        enc = [{k: f[k] for k in keys if k in f} for f in feats]
        batch = self.tok.pad(enc, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


class PatentData:
    """
    토크나이저·데이터셋·collator를 한 객체로 묶는다.
    토크나이저·데이터셋은 백본 스펙에서, `max_len`·캐시 경로는 config에서 온다.
    """

    def __init__(self, cfg: TrainConfig, backbone: Backbone):
        self.cfg = cfg
        self.bb = backbone
        self.tokenizer = None
        self.collator = None
        self.raw = None          # 절단 전 원본
        self.dataset = None      # 절단 후(훈련에 쓰는 것)

    # ── 캐시 ──────────────────────────────────────────────────────────────
    @property
    def prep_cache(self) -> str:
        """백본·길이별 절단 결과 캐시 경로(백본이 다르면 토크나이저가 달라 섞이면 안 된다)."""
        return os.path.join(self.cfg.prep_cache_root, f"{self.bb.key}_len{self.cfg.max_len}")

    def has_prep_cache(self) -> bool:
        return os.path.isdir(self.prep_cache)

    # ── 단계 ──────────────────────────────────────────────────────────────
    def load_tokenizer(self) -> "PatentData":
        """백본 스펙의 토크나이저를 `rev` 고정으로 로드하고 collator를 만든다."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.bb.model_name, revision=self.bb.rev)
        self.collator = MultiLabelCollator(self.tokenizer)
        return self

    @property
    def _infer(self) -> bool:
        """추론 경로(체크포인트 로드) 여부 — on-disk prep 캐시를 우회하는 조건."""
        return self.cfg.checkpoint is not None

    def _select_splits(self, ds):
        """`cfg.splits`가 주어지면 그 split만 남긴다(예: ("val","test") — train 로드 회피)."""
        if self.cfg.splits is None:
            return ds
        from datasets import DatasetDict
        return DatasetDict({sp: ds[sp] for sp in self.cfg.splits})

    def load(self) -> "PatentData":
        """원본(절단 전) 토큰화 데이터셋을 Hub에서 로드한다.

        prep 캐시가 이미 있으면(훈련 경로) 원본이 쓰이지 않으므로 다운로드를 건너뛴다.
        데이터셋이 갱신됐다면 prep 캐시를 지워야 새로 받는다. 추론 경로는 캐시를 우회하므로 항상 받는다.
        `cfg.splits`가 주어지면 그 split만 남긴다.
        """
        if not self._infer and self.has_prep_cache():
            print(f"[skip] prep 캐시 존재 — 원본 로드 생략: {self.prep_cache}")
            return self
        self.raw = self._select_splits(load_dataset(self.bb.dataset_id, cache_dir=self.cfg.hf_cache))
        return self

    def prepare(self) -> "PatentData":
        """`max_len` 절단을 적용한다.

        훈련 경로는 결과를 prep 캐시에 저장·재사용한다. 추론 경로(`cfg.checkpoint`)는 캐시를
        우회한다 — 휘발 VM에서 디스크 캐시 이득이 없고, split 부분집합만 prep한 캐시가 훗날 전체
        훈련 런에 재사용돼 조용히 train이 비는 것을 원천 차단한다.
        """
        if not self._infer and self.has_prep_cache():
            self.dataset = load_from_disk(self.prep_cache)
            return self
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.raw is None:
            self.load()
        self.dataset = self.raw.map(self._prep, batched=True)
        if not self._infer:
            self.dataset.save_to_disk(self.prep_cache)
        return self

    def _prep(self, batch):
        max_len = self.cfg.max_len
        eos_id = self.tokenizer.eos_token_id
        ids, masks = [], []
        for x, m in zip(batch["input_ids"], batch["attention_mask"]):
            if len(x) > max_len:
                x = x[: max_len - 1] + [eos_id]   # <s> 유지 + 꼬리를 </s>로 마감
                m = m[:max_len]
            ids.append(x)
            masks.append(m)
        return {"input_ids": ids, "attention_mask": masks, "length": [len(i) for i in ids]}
