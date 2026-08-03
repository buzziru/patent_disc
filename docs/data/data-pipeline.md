# 데이터 파이프라인 — zip → 정제 텍스트(HF Hub) → 소비 시 토큰화

원본 zip을 그대로 두고 **정제 텍스트를 한 번만 만들어 HF Hub에 올린 뒤, 실험마다 그것을 받아 토큰화**하는 2계층 설계다.

- **Layer 1**: 원본 zip을 1회 정제해 **토큰화하지 않은 텍스트 데이터셋**을 HF Hub(parquet)에 올린다. 모델·토크나이저와 무관하다.
- **Layer 2**: 실험마다 그것을 streaming으로 받아 **소비 시점에 토큰화**한다.

이렇게 하면 원본 zip을 디스크에 풀지 않아도 되고, Colab에 데이터를 업로드할 필요가 없으며, 두 플랫폼의 데이터 로딩 코드가 하나로 통일된다.

기호·용어는 [GLOSSARY.md](../GLOSSARY.md), 데이터 원본 구조는 [data.md](./data.md), 실행 인프라는 [colab-jobs.md](../infra/colab-jobs.md)·[lightning-jobs.md](../infra/lightning-jobs.md)를 참조한다.

## 왜 이 구조인가

- 데이터는 **특허 1건이 작은 JSON 1개**이고 zip당 수천 개가 들어 있다. 전량 `unzip`하면 파일이 수십만 개가 되어 파일시스템에 부담을 준다. 그래서 `zipfile`로 프로그램에서 스트리밍해 곧장 정제본을 만든다.
- Colab VM은 원격·휘발성이라 로컬 `data/`를 볼 수 없다. VM 안에서 HF Hub를 받아오면 매 세션 업로드가 사라진다.
- Lightning Job도 별도 GPU 머신의 컨테이너에서 돌아 로컬 `data/`를 볼 수 없다. 컨테이너 안에서 HF Hub를 받아오면 Colab과 로딩 코드가 그대로 같아진다.

**토큰화본이 아니라 텍스트를 올리는 이유**는 실험 변수를 열어 두기 위해서다. 미리 토큰화하면 세 축이 한꺼번에 고정된다 — ① 토크나이저(A.X-Encoder / KLUE-RoBERTa / KoBERT 3종 비교), ② 입력 필드 조합, ③ `MAX_LEN`. 텍스트를 SSOT로 두고 토큰화를 Layer 2로 미루면 세 축을 자유롭게 바꿀 수 있다.

---

## Layer 1 — 정제 텍스트 데이터셋 (1회, 모델·토크나이저 무관)

산출물은 `<user>/patent-clean-text`(HF Hub, split=train/validation/test, parquet)이며 토큰화하지 않은 상태다.

### 핵심 결정

- **라벨은 문서별 다중-핫이다.** 한 특허가 여러 `Mno`에 대응하므로 문서 단위로 `Mno` 집합을 모아 `label_ids: list[int]`(0~187)로 저장하고, 다중-핫 벡터는 Layer 2에서 만든다.
- **조인·집계 키는 `documentId`(= 파일명)다.** 라벨 zip은 파일명에 `Mno`가 들어 있어(`…_EG10_핵융합.zip` → `EG10`) 라벨 JSON을 열지 않고 zip 이름만으로 `Mno`를 얻는다.
- **분할은 `documentId` 단위로 한다.** 제공된 `Training`/`Validation` 폴더는 7,822개 문서가 겹쳐 누수가 있으므로 그대로 쓸 수 없다. 전체 고유 문서를 모아 재분할하며, 문서당 1행이므로 일반 셔플 분할이 곧 group-safe다.
- **텍스트 필드는 개별 컬럼으로 보존한다**(`invention_title`·`abstract`·`claims`·`ipc_main`). 연결과 필드 조합은 Layer 2에서 한다.

### 절차

**Pass 1 — 라벨 인덱스와 매핑** (Label zip의 `namelist()`만 읽고 JSON은 파싱하지 않는다)

```python
import re, zipfile
from pathlib import Path
from collections import defaultdict

DATA = Path(os.environ["DATA_ROOT"]) / "data"
MNO_RE = re.compile(r"[A-Z]{2}[0-9]{2}")          # EG10 형태

def build_label_index():
    doc2mno = defaultdict(set)                     # documentId -> {Mno}
    for split in ["Training", "Validation"]:       # 폴더 구분 무시하고 전부 모음
        for z in (DATA / split / "Label").glob("*.zip"):
            mno = MNO_RE.findall(z.stem)[0]         # zip명에서 Mno
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.endswith(".json"):
                        doc2mno[Path(n).stem].add(mno)
    return doc2mno

doc2mno = build_label_index()
MNO2ID = {m: i for i, m in enumerate(sorted({m for s in doc2mno.values() for m in s}))}  # 188개
ID2MNO = {i: m for m, i in MNO2ID.items()}
MNO2LNO = {m: m[:2] for m in MNO2ID}               # Lno = Mno 앞 2글자 (EG10 -> EG)
```

**Pass 2 — 텍스트 수집** (Orig zip 스트리밍, `documentId`를 최초 1회만 취해 중복을 제거한다)

```python
import json

TEXT_FIELDS = ["invention_title", "abstract", "claims", "ipc_main"]

def iter_clean_records(doc2mno):
    seen = set()
    for split in ["Training", "Validation"]:
        for z in (DATA / split / "Orig").glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if not n.endswith(".json"):
                        continue
                    doc = Path(n).stem
                    if doc in seen:                # 여러 zip 중복 수록 → 1회만
                        continue
                    seen.add(doc)
                    d = json.loads(zf.read(n))["dataset"]
                    mnos = sorted(doc2mno[doc])
                    yield {
                        "document_id": doc,
                        **{f: d.get(f) or "" for f in TEXT_FIELDS},   # 결측 → 빈 문자열
                        "mno": mnos,                                  # ["EG10", ...]
                        "label_ids": [MNO2ID[m] for m in mnos],       # [75, ...]
                        "lno": sorted({MNO2LNO[m] for m in mnos}),
                    }
```

**분할과 업로드** (`documentId` 단위)

```python
from datasets import Dataset

ds = Dataset.from_generator(lambda: iter_clean_records(doc2mno))
# 문서당 1행 → 일반 셔플 분할이 곧 group-safe. 예: train 0.9 / val 0.05 / test 0.05
tmp = ds.train_test_split(test_size=0.1, seed=42)
val_test = tmp["test"].train_test_split(test_size=0.5, seed=42)
splits = {"train": tmp["train"], "validation": val_test["train"], "test": val_test["test"]}

for name, part in splits.items():
    part.push_to_hub("<user>/patent-clean-text", split=name)   # 미토큰화 parquet
# MNO2ID / MNO2LNO 는 리포에 함께 저장(json) — 역매핑·Lno 매핑 재현용
```

두 가지를 이 단계에서 정한다.

- **빈 텍스트 문서**: 업체 baseline은 `abstract`와 `claims`가 **둘 다** 비면 제거했다(실측 1건). 같은 규칙을 적용할지 여기서 정해 필터링한다.
- **분할 비율·시드는 자체 비교선용이다.** baseline 프로토콜 재현은 별개 사안이며, 공식 `0.8249`는 절대 기준이 아니다([PROJECT.md](../../PROJECT.md) 평가 절).

---

## Layer 2 — 소비: 실험마다 토큰화 (Colab·Lightning 동일 코드)

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("skt/A.X-Encoder-base")   # 실험마다 모델별 토크나이저
ds = load_dataset("<user>/patent-clean-text", split="train", streaming=True)  # IterableDataset

FIELDS = ["invention_title", "abstract", "claims"]           # 입력 필드 조합 = 실험 변수
def build_input(ex):                                          # 필드 선택·연결 방식도 실험 변수
    return " ".join(ex[f] for f in FIELDS if ex[f])

NUM_LABELS = 188
def tokenize(ex):
    out = tok(build_input(ex), truncation=True, max_length=MAX_LEN)  # 가변 길이(패딩 X)
    y = [0.0] * NUM_LABELS
    for i in ex["label_ids"]:
        y[i] = 1.0                                            # 다중-핫 (float, BCE용)
    out["labels"] = y
    return out

ds = ds.map(tokenize, remove_columns=[c for c in ds.column_names])
```

- 모델 헤드는 `AutoModelForSequenceClassification.from_pretrained(..., num_labels=188, problem_type="multi_label_classification")`로 sigmoid + BCE 경로를 쓴다.
- 지표는 **다중 라벨 micro/macro-F1**(임계값 0.5)을 주로 보고, baseline과 맞대는 **top-1 예측 weighted-F1**과 `P@1/3/5`를 병기한다([PROJECT.md](../../PROJECT.md) 평가 절, 재현 절차는 [kobert-baseline.md](../experiments/kobert-baseline.md)). test split은 고정해 모든 실험에서 재사용한다.

## Streaming(IterableDataset) 주의

1. **`len()`이 없다** → HF `Trainer`에 `max_steps`를 명시해 스텝 기반 스케줄을 쓴다.
2. **셔플이 버퍼 기반이다** — `ds.shuffle(buffer_size=…, seed=…)`는 전역 셔플이 아니다. 버퍼를 넉넉히 잡고 epoch마다 seed를 재설정한다.
3. **동적 패딩**: `DataCollatorWithPadding`으로 배치 단위 패딩을 건다(저장은 가변 길이).
4. **다중-핫 라벨**: collator가 `labels`를 float 텐서로 유지하는지 확인한다(`BCEWithLogits` 요구사항).
5. **샤딩 저장**: parquet를 여러 shard로 나누면 스트리밍 처리량과 병렬 로딩에 유리하다.

## 원칙 요약

- **텍스트로 저장한다**(미리 토큰화하지 않는다) — 토크나이저·필드 조합·`MAX_LEN` 세 축을 소비 시점에 자유화한다.
- **가변 길이로 저장한다**(패딩 금지) — 패딩은 Layer 2의 collator에서 동적으로 건다.
- **라벨은 `label_ids` 리스트로 저장한다** — 다중-핫 벡터는 소비 시점에 만든다.
- **`MNO2ID`·`MNO2LNO`(188개)를 리포에 동봉한다** — 역매핑과 `Mno`→`Lno` 매핑을 재현하기 위해서다.
- **분할은 `documentId` 단위로 한다** — 제공 폴더의 누수 7,822건을 피한다.

## 플랫폼별 데이터 반입 요약

| | Colab | Lightning Job (Docker 이미지) |
| --- | --- | --- |
| 로컬 `data/` 접근 | 불가(원격·휘발 VM) | 불가(별도 GPU 머신의 컨테이너) |
| 권장 반입 | **HF Hub pull(streaming)** | **HF Hub pull(streaming)** 또는 `path_mappings`로 data-connection |
| 인증 | 노트북 `os.environ["HF_TOKEN"]` | SDK `env={"HF_TOKEN": …}` |
