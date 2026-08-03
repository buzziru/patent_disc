# KLUE-RoBERTa-large 대조군 — 계획·프로토콜

**512 토큰에 묶인 또 하나의 한국어 인코더**를 놓아 A.X-Encoder의 이득이 어디서 오는지 삼각측량하는 대조군이다. 장문 런의 경쟁자가 아니라 512 대조 런 옆에 놓이는 **두 번째 512 관측점**이다. 이 문서는 계획과 프로토콜이며, 실행은 후순위로 두었다.

기호·용어·런 코드는 [GLOSSARY.md](../GLOSSARY.md)를 참조한다.

기준선(고정 test 11,271): KoBERT **micro 0.8502 / top-1 weighted 0.8148**, exp2(A.X 512) **micro 0.8601 / 0.8203**, exp1(A.X 8192) **micro 0.8685 / 0.8256**. 상세는 [modernbert-comparison.md](./modernbert-comparison.md).

## 이 실험이 답하는 것과 답하지 못하는 것

512 창의 관측점이 KoBERT 하나뿐일 때, exp2−KoBERT의 +0.99pt는 "A.X가 좋아서"인지 "KoBERT가 낡아서"인지 구분되지 않는다. 512에 묶인 현대적 한국어 인코더를 하나 더 놓으면 이 축이 분리된다.

- **답한다**: 512 창에서 KoBERT의 열세가 모델 일반의 세대차인가, A.X 고유의 우위인가. KLUE가 exp2에 근접하면 전자(KoBERT가 낡음), KLUE가 KoBERT 쪽이면 후자(A.X 고유).
- **답하지 못한다**: **A.X vs KLUE는 크기가 통제되지 않는다** — A.X-Encoder-base **149M** vs KLUE-RoBERTa-large **337M**(2.26배). 이 비교의 delta에는 아키텍처·사전학습·**파라미터 수**가 함께 섞인다. 순수 아키텍처 비교로 읽어선 안 된다.
- **주의**: KLUE가 exp1(8192)에 못 미치는 것은 예정된 결과이지 발견이 아니다. KLUE 컨텍스트는 512로 고정돼 있어(아래 「512 창」) 장문 축에서는 애초에 경쟁하지 않는다.

## 공통 프로토콜

- **모델**: `klue/roberta-large`(revision `28d911204e9022eda172571ca8cc61eaffd942f7`로 pin). `model_type=roberta`, **337M 파라미터**, hidden 1,024 / 24 layers, vocab 32,000. 체크포인트 `architectures`는 `RobertaForMaskedLM`이라 `AutoModelForSequenceClassification` 로드 시 분류 head가 새로 초기화된다(정상 — 경고 무시).
- **데이터**: `ingyoun/patent-clean-text-roberta-tokenized`(train 201,895 / val 11,162 / test 11,271). 상류 `notebook/07_01_Prep_RoBERTa.ipynb`가 1회 토큰화해 Hub에 올렸다. **truncation 없이 저장**(train max 10,275 / test max 8,951)돼 있어 `max_length`는 소비 시점에서 건다 — A.X 데이터셋과 동일한 2계층 정책.
- **입력 필드**: `invention_title + ipc_main + abstract + claims`(공백 join, 빈 필드 skip) — KoBERT baseline·A.X와 동일 필드 집합.
- **타깃**: 문서별 188 다중-핫(`labels`, float), focal loss(alpha=0.25, gamma=2) — baseline·exp1·exp2와 정합.
- **고정 test 원칙**: 같은 test split·같은 `kobert_len` 길이 bin 위에서 평가(`../data/data.md` 「길이 슬라이스 bin」). 토큰화 데이터셋이 canonical `length_bin`(test 3,197 / 5,183 / 2,342 / 549)을 보존하므로 그 컬럼을 그대로 쓴다 — bin 축이 `kobert_len`에 고정돼 있어 토크나이저별로 bin을 따로 구할 일은 없다. RoBERTa 토큰 길이는 절단 규모를 보는 용도이고, 그 분포는 `../data/data.md` 「KLUE-RoBERTa 토크나이저」의 percentile로 관리한다.
- **평가**: 다중 라벨 micro/macro/sample-F1(τ=0.5) + 길이 bin + top-1 weighted + LRAP/R-Precision. `tag`는 `roberta-patent-len512`.
- ⚠️ **추론 `batch_size`는 batch 8 고정** — 동적 패딩이 배치 내 최장 문서에 맞추므로 배치를 바꾸면 로짓이 ~1e-4 흔들려 지표가 4자리에서 어긋난다(실측 근거 [`modernbert.md`](./modernbert.md) 공통 프로토콜).

## 512 창 — A.X와 성격이 다른 절단

`max_position_embeddings=514`이나 RoBERTa 관례상 앞 2칸이 `pad_token_id` offset이라 **실질 창은 512**다. A.X(16,384)와 달리 **창을 넘기면 절단이 선택이 아니라 필수**다.

| | A.X-Encoder | KLUE-RoBERTa |
| --- | --- | --- |
| 컨텍스트 | 16,384 | **512** |
| 무절단본을 그대로 투입 | 안전(코퍼스 max 10,523 < 16,384) | **train 63.5%가 창 초과** |
| 절단 누락 시 | 훈련 창과 어긋난 무의미한 평가 | **position embedding 인덱스 에러** |

⚠️ **절단 규칙**: 훈련·평가 모두 `notebook/05_01_ModernBERT_Len512.ipynb`의 `_prep`과 같은 자리에서 같은 방식으로 건다. 단순 슬라이싱(`x[:max_len]`)은 꼬리 `[SEP]`를 버리므로 `x[:max_len-1] + [sep_token_id]`로 마감한다.

```python
SEP_ID = tokenizer.sep_token_id            # 2

def _prep(batch):
    max_len = config["max_len"]            # 512
    ids, masks = [], []
    for x, m in zip(batch["input_ids"], batch["attention_mask"]):
        if len(x) > max_len:
            x = x[: max_len - 1] + [SEP_ID]   # [CLS] 유지 + 꼬리를 [SEP]로 마감
            m = m[:max_len]
        ids.append(x)
        masks.append(m)
    return {"input_ids": ids, "attention_mask": masks, "length": [len(i) for i in ids]}
```

- 시퀀스는 `[CLS]`(0) … 본문 … `[SEP]`(2)로 감싸이고, 앞의 `[CLS]`는 index 0이라 슬라이싱해도 보존된다.
- **A.X의 특수토큰 함정은 재발하지 않는다** — A.X는 `sep_token_id`(3)가 실제 마감 토큰 `eos`(1)와 달라 절단 복원에 쓰면 엉뚱한 토큰이 붙었으나, KLUE-RoBERTa는 `sep_token_id == eos_token_id == 2`로 일치한다. 여기서는 `sep_token_id`가 정답이다.
- **A.X보다 영향이 클 수 있다**: A.X exp1은 절단 문서가 <1%인 데다 `classifier_pooling="mean"`이라 토큰 1개 손실이 무해했다. KLUE는 **문서의 64%가 절단**되고 분류가 `[CLS]` 표현에 걸리므로, 마감 토큰 보존을 규칙으로 고정한다.

## dtype — A.X의 함정은 적용되지 않는다

`klue/roberta-large`는 **F32로 저장**돼 있다(safetensors dtype `F32`, 336,690,432 파라미터). A.X-Encoder를 붕괴시킨 "bf16 저장 체크포인트를 bf16 autocast로 훈련 → 업데이트 underflow" 경로가 성립하지 않는다.

그럼에도 **로드 시 `dtype=torch.float32`를 명시**한다 — 체크포인트 dtype에 의존하지 않고 fp32 마스터 가중치 + autocast(bf16)라는 표준 혼합정밀 레시피를 코드에 드러내기 위함이며, A.X 노트북과 경로를 통일한다. `bf16=True`(autocast)는 그대로 쓰고 **fp16은 쓰지 않는다**.

FA2(`attn_implementation="flash_attention_2"`)는 `RobertaForSequenceClassification`이 지원한다(`_supports_flash_attn=True`). 다만 512 고정 길이라 A.X 8192에서와 같은 결정적 이득은 없다 — 메모리가 빠듯할 때의 선택지로 둔다.

## 토크나이저 — 선측정 결과가 부과하는 두 가지

실측은 `../data/data.md` 「KLUE-RoBERTa 토크나이저」·`output/tokenizer_analysis.json`(`notebook/06_03`).

| | KoBERT | A.X | KLUE |
| --- | --- | --- | --- |
| vocab | 8,002 | 49,999 | 32,000 |
| 문자당 토큰 | 0.6045 | 0.5449 | **0.5369** |
| test >512 | 71.64% | 65.15% | 64.22% |
| 512 무손실 문서 | 28.36% | 34.85% | **35.78%** |
| `[UNK]` 비율 | 0.0000% | 0.0036% | **0.1495%** |
| UNK 보유 문서 | 0.00% | 1.06% | **21.93%** |

- **A.X-512 vs KLUE-512도 콘텐츠 양이 통제되지 않는다.** KLUE가 A.X보다 압축적이라(0.5369 vs 0.5449) 같은 512 창에 ~1% 더 많은 본문을 담는다. exp2 vs KoBERT를 오염시킨 것과 같은 종류의 교란 요인이 **크기 약 1/10로, 이번엔 대조군 쪽에 유리하게** 재발한다. 결론을 뒤집을 크기는 아니나, **KLUE가 exp2를 근소하게 이기면 이 성분을 먼저 배제**한 뒤 해석한다(절차는 [`modernbert-comparison.md`](./modernbert-comparison.md) 「커버리지 기제의 직접 실측」).
- **KLUE만 `[UNK]`가 실재한다** — 문서의 21.93%가 UNK를 최소 1개 포함(KoBERT는 SentencePiece라 0%, A.X는 1.06%). 훈련 전에 **무엇이 UNK로 떨어지는지 표본 확인**하고, 성능이 기대에 못 미치면 열화 후보로 우선 조사한다.

## 훈련 설정

`05_01`(exp2, A.X 512)을 기준으로 두고 **의도적으로 바꾸는 항목만** 아래에 둔다. 나머지(focal loss, `MultiLabelCollator`, `compute_metrics`, `group_by_length`, `metric_for_best_model="micro_f1"`, early stopping patience 6, `save_total_limit` 3)는 그대로 가져온다.

| 항목 | exp2 (A.X 512) | KLUE 대조군 | 사유 |
| --- | --- | --- | --- |
| `model_name` | `skt/A.X-Encoder-base` | `klue/roberta-large` | — |
| 파라미터 | 149M | 337M | 크기 교란 요인(위 「답하지 못하는 것」) |
| `dtype` 로드 | `float32`(bf16 함정 회피 필수) | `float32`(명시적 통일, 무해) | — |
| `micro_batch` | 8 | **미정 — `probe_batches`로 결정** | 2.26배 모델이라 재측정 필요 |
| `learning_rate` | 3e-5 | **미정 — 2e-5 우선** | 아래 참조 |
| `tag` / repo | `modernbert-patent-len512` | `roberta-patent-len512` / `ingyoun/roberta-patent-maxlen512` | — |

- **`micro_batch`는 반드시 `probe_batches`로 다시 잰다.** fp32 파라미터 기준 가중치 1.35GB + 그래디언트 1.35GB + AdamW state 2.7GB ≈ 5.4GB가 상주한 뒤 활성값이 얹힌다. `eff_batch=8`은 유지하고 `grad_accum`으로 맞춘다 — 유효 배치를 exp2와 같게 두어야 비교에서 배치 변수가 빠진다.
- **`learning_rate`는 3e-5를 그대로 쓰지 않는다.** RoBERTa-large는 큰 lr에서 발산·붕괴가 알려진 모델이고, KLUE 공식 baseline도 large에는 1e-5~3e-5 범위를 탐색한다. **2e-5로 시작**하고, 손실이 초반에 평탄해지거나 상수 분류기로 붕괴하면 1e-5로 낮춘다. warmup_ratio 0.1은 유지한다.
  - 이는 exp2와 훈련 레짐이 달라진다는 뜻이며, **KLUE−exp2 delta에 레짐 잔차가 섞인다**는 한계로 결과에 명시한다. lr을 억지로 맞춰 발산시키는 쪽이 더 나쁜 교환이다.

## 산출물

- 훈련 노트북 `notebook/07_02_RoBERTa_Len512.ipynb`(Colab), 평가 `notebook/07_03_RoBERTa_Len512_Metric.ipynb`.
- 체크포인트 `ingyoun/roberta-patent-maxlen512`, 지표 `output/total_metrics_roberta-patent-len512.json`, 로짓 `output/logits_roberta-patent-len512_{test,val}.npy`.
- 실측은 이 문서에 절을 추가하고, 3모델 비교는 [`modernbert-comparison.md`](./modernbert-comparison.md)가 4모델로 확장해 다룬다.

## 미정·다음 단계

- **`micro_batch`·`learning_rate` 확정** — 위 표. 훈련 전 `probe_batches` 1회 + lr 초반 손실 관찰.
- **UNK 표본 조사** — 문서 21.93%에 걸리는 UNK가 무엇인지(특허 기호·수식 추정) 확인.
- **τ 정책** — [`no-train-analysis.md`](./no-train-analysis.md) B가 3모델 공통 τ 정책을 세우면 KLUE도 같은 정책으로 평가한다. 한 모델에만 적용하면 비교가 불공정해진다.
