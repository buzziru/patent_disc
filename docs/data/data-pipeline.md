# 데이터 파이프라인 — zip → 정제 텍스트(HF Hub) → 소비 시 토큰화

> **2계층 설계.** 
>
> Layer 1: raw zip을 1회 정제해 **미토큰화 텍스트 데이터셋**을 HF Hub(parquet)에 올린다(모델·토크나이저 무관). 
>
> Layer 2: 실험마다 그걸 streaming으로 받아 **소비 시점에 토큰화**한다.
>
> raw zip을 **디스크에 풀지 않음** · Colab에 데이터를 **업로드하지 않음** · 두 플랫폼의 **데이터 로딩 코드 일원화**.
> 데이터 원본 구조는 `[data.md](./data.md)`, 실행 인프라는 `[colab-jobs.md](../infra/colab-jobs.md)` · `[lightning-jobs.md](../infra/lightning-jobs.md)`.

## 왜 이 구조인가

- 데이터는 **특허 1건 = 작은 JSON 1개**(zip당 수천 개). 전량 `unzip`하면 파일 수십만 개 → inode·FS 부담. `zipfile`**로 프로그램 스트리밍**해 곧장 정제본을 만든다.
- Colab VM은 **원격·휘발성** → 로컬 `data/`를 못 본다. **VM 안에서 HF Hub를 pull**하면 매 세션 업로드가 사라진다.
- Lightning Job도 **별도 GPU 머신의 컨테이너**에서 돌아 로컬 `data/`를 못 본다. **컨테이너 안에서 HF Hub를 pull**하면 Colab과 로딩 코드가 그대로 같아진다.
- **토큰화본이 아니라 텍스트를 올리는 이유**: 토큰화하면 ①토크나이저(A.X-Encoder / KLUE-RoBERTa / KoBERT 3종 비교) ②입력 필드 조합(핵심 실험 변수) ③`MAX_LEN` 세 축이 한꺼번에 고정된다. 텍스트를 SSOT로 두고 **토큰화를 Layer 2(소비 시)로 미루면** 세 축을 자유롭게 실험한다.

---

## Layer 1 — 정제 텍스트 데이터셋 (1회, 모델·토크나이저 무관)

산출물: `<user>/patent-clean-text` (HF Hub, split=train/validation/test, parquet). **미토큰화.**

### 핵심 결정 (SSOT 반영)

- **레이블은 문서별 다중-핫(multi-label)**. 한 특허가 여러 `Mno`에 대응하므로, 문서 단위로 `Mno` 집합을 모은다. → `label_ids: list[int]`(0..187)로 저장, 다중-핫 벡터는 Layer 2에서 생성.
- **조인/집계 키는 `documentId`(=파일명, 고유 특허)**. 라벨 zip은 **파일명에 `Mno`가 인코딩**돼 있어(`…_EG10_핵융합.zip` → `EG10`) 라벨 JSON을 열 필요 없이 zip명만으로 `Mno`를 얻는다.
- **분할은 `documentId` 단위.** 제공된 `Training`/`Validation` 폴더는 **7,822개 문서가 겹쳐**(누수) 그대로 못 쓴다 → 전체 고유 문서를 모아 재분할한다(문서당 1행이므로 일반 셔플 분할이 곧 group-safe).
- 텍스트 필드는 **개별 컬럼**으로 보존(`invention_title`·`abstract`·`claims`·`ipc_main`). concat/필드조합은 Layer 2에서.

### 절차

**Pass 1 — 라벨 인덱스 + 매핑 (Label zip의 `namelist()`만; JSON 파싱 불필요)**

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

**Pass 2 — 텍스트 수집 (Orig zip 스트리밍, `documentId` 최초 1회만 = dedupe)**

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

**분할 + 업로드 (`documentId` 단위)**

```python
from datasets import Dataset

ds = Dataset.from_generator(lambda: iter_clean_records(doc2mno))
# 문서당 1행 → 일반 셔플 분할이 곧 group-safe. 예: train 0.9 / val 0.05 / test 0.05
tmp = ds.train_test_split(test_size=0.1, seed=42)
val_test = tmp["test"].train_test_split(test_size=0.5, seed=42)
splits = {"train": tmp["train"], "validation": val_test["train"], "test": val_test["test"]}

for name, part in splits.items():
    part.push_to_hub("<user>/patent-clean-text", split=name)   # 미토큰화 parquet
# MNO2ID / MNO2LNO 는 리포에 함께 저장(json) — 역매핑·Lno lookup 재현용
```

> ⚠️ **빈 텍스트 문서**: 업체 baseline은 `abstract`·`claims`가 **둘 다** 비면 제거했다(실측 1건). 같은 규칙을 적용할지 정해 여기서 필터링한다.
> ⚠️ **baseline 프로토콜 재현**은 별개 이슈 — 위 분할 비율/시드는 우리 자체 비교선용. 공식 0.8249는 절대 기준 아님(`[PROJECT.md](../../PROJECT.md)` 평가 절).

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

- 모델 헤드: `AutoModelForSequenceClassification.from_pretrained(..., num_labels=188, problem_type="multi_label_classification")` → sigmoid + BCE.
- 지표: **baseline 비교는 top-1 예측 weighted-F1**(KoBERT baseline과 동일 계산 — `../../PROJECT.md` 평가 절, 재현 절차 `../experiments/kobert-baseline.md`). 멀티레이블 프레이밍용으로 **micro/macro-F1(임계값 0.5, 검증셋 튜닝)·P@1/3/5** 병기. test split은 고정해 전 실험 재사용.

## Streaming(IterableDataset) 주의

1. `**len()` 없음** → HF `Trainer`에 `max_steps` 명시(steps 기반 스케줄).
2. **셔플이 버퍼 기반**: `ds.shuffle(buffer_size=…, seed=…)` — 전역 아님. buffer 넉넉히, epoch마다 seed 재설정.
3. **동적 패딩**: `DataCollatorWithPadding`으로 배치 단위 패딩(저장은 가변 길이).
4. **다중-핫 라벨**: collator가 `labels`를 float 텐서로 유지하도록 확인(BCEWithLogits).
5. **샤딩 저장**: parquet를 여러 shard로 → 스트리밍 처리량·병렬 로딩 유리.

## 원칙 요약

- **텍스트로 저장**(토큰화 미리 안 함) — 토크나이저·필드조합·`MAX_LEN` 3축을 소비 시 자유화.
- **가변 길이 저장**(패딩 금지) — 패딩은 Layer 2 collator에서 동적으로.
- **레이블은 `label_ids` 리스트로 저장**(가변) — 다중-핫 벡터는 소비 시 생성.
- `**MNO2ID`/`MNO2LNO`(188)**를 리포에 동봉(역매핑·`Mno`→`Lno` lookup 재현).
- **분할은 `documentId` 단위** — 제공 폴더 누수(7,822건) 회피.

## 플랫폼별 데이터 반입 요약


|               | Colab                        | Lightning Job (Docker 이미지)                                       |
| ------------- | ---------------------------- | ---------------------------------------------------------------- |
| 로컬 `data/` 접근 | ✗ (원격·휘발 VM)                 | ✗ (별도 GPU 머신의 컨테이너)                                              |
| 권장 반입         | **HF Hub pull(streaming)**   | **HF Hub pull(streaming)** (또는 `path_mappings`로 data-connection) |
| 인증            | 노트북 `os.environ["HF_TOKEN"]` | SDK `env={"HF_TOKEN": …}`                                        |


