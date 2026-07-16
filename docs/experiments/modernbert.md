# A.X-Encoder(ModernBERT) 실험 — 계획·프로토콜 (허브)

> **목적**: 512-truncation KoBERT baseline을 **장문 인코더**(`skt/A.X-Encoder-base`, 한국어 ModernBERT, 16k 컨텍스트)로 개선하고, 같은 고정 test 위에서 정량 입증한다. 이 문서는 실험 **계획·공통 프로토콜·비교 축**을 고정한다. **실험별 실측은 [`modernbert-results.md`](./modernbert-results.md), 교차 비교·결론은 [`modernbert-comparison.md`](./modernbert-comparison.md).**
>
> 기준선(`kobert-baseline.md`, 고정 test 11,271): **앵커 top-1 weighted-F1 0.8148** / **멀티라벨 micro 0.8502 · macro 0.8470 · sample 0.8656** (τ=0.5). A.X-Encoder는 이 두 축 각각에 대해 개선을 재는 것이지, 두 수치를 서로 뺄셈하지 않는다.

## 현재 결론 (요약)

- **exp1(8192)이 KoBERT 재현선을 넘었다**: 멀티라벨 micro 0.8502 → **0.8685**(+1.83pt), 앵커 top-1 weighted 0.8148 → **0.8256**.
- **개선 분해**(exp2 512 control): 모델·토크나이저 효과 **+0.99pt** + 컨텍스트 길이 효과 **+0.84pt**(대략 절반씩). 길이 효과는 문서가 길수록 커져 최장 문서군(B3)에서 최대(+2.64pt).
- **멀티라벨 전 지표에서 exp1 > exp2 > KoBERT 단조**(F1 micro/macro/sample · LRAP · R-Precision · P@k). 상세는 [`modernbert-results.md`](./modernbert-results.md)·[`modernbert-comparison.md`](./modernbert-comparison.md).

## 공통 프로토콜 (모든 실험 고정)

- **모델**: `skt/A.X-Encoder-base` (한국어 ModernBERT, 컨텍스트 16,384, vocab 49,999, apache-2.0). 토크나이저 revision `9708f9c4`로 pin.
- **데이터**: `ingyoun/patent-clean-text-modernbert-tokenized` (train 201,895 / val 11,162 / test 11,271). 사전토큰화 완료본을 그대로 소비 — 입력 조합·토크나이즈는 상류(`notebook/04_01_Prep_ModernBERT.ipynb`)에서 1회 수행. **truncation 없이 최대 10,523토큰**으로 저장돼 있어 `max_length`는 소비 시점(훈련 config)에서 건다.
- **입력 필드**: `invention_title + ipc_main + abstract + claims` (공백 join, 빈 필드 skip) — KoBERT baseline과 동일 필드 집합.
- ⚠️ **dtype 함정**(실측): base `skt/A.X-Encoder-base`는 **bf16으로 저장**돼 있고(config `torch_dtype: "bfloat16"`), transformers **v5는** `dtype` **미지정 시 체크포인트 dtype 그대로(bf16) 파라미터를 로드**한다(v4는 fp32로 업캐스트해 이 문제가 안 보였음). bf16 파라미터를 `bf16=True`(autocast)로 훈련하면 옵티마이저가 bf16 파라미터를 직접 갱신 → 작은 갱신량(lr 3e-5×grad)이 8비트 가수부에서 **underflow로 소멸** → 가중치가 얼어붙어 **상수(다수클래스) 분류기로 붕괴**한다(디스크 가중치 손상이 아니라 업데이트 소멸이 원인). **해결: 로드 시** `dtype=torch.float32` **명시** → fp32 마스터 가중치 + autocast(bf16) 연산 = 표준 혼합정밀 레시피. FA2는 autocast가 bf16 q/k/v를 공급하므로 fast path·속도 이득 그대로 유지되고, 추가 비용은 파라미터 메모리 2배뿐(연산시간 손해 사실상 없음). **fp16은 금지**(ModernBERT 활성값이 fp16 범위를 넘겨 NaN) — 안전 dtype은 **fp32(훈련 파라미터)·bf16(추론)**이다. 추론(`04_03`)도 `dtype=torch.float32` 로드 + `torch.autocast(bfloat16)`로 훈련과 동일 경로. 로드 시 transformers가 `Flash Attention 2 only supports torch.float16 and torch.bfloat16 … current dtype is torch.float32` **경고**를 내지만 무해하다 — 파라미터 저장 dtype에 대한 알림일 뿐 실제 연산은 autocast가 bf16으로 수행한다.
- ⚠️ **토크나이저 특수토큰 함정**(실측): A.X-Encoder는 시퀀스를 `<s>`**(0) … 본문 …** `<\s>`**(1)** 로 감싼다. 마감 토큰은 **`eos_token_id`(=1)**이며, `tokenizer.sep_token_id`**는** `<sep>`**(=3)으로 실제 마감 토큰이 아니다** — 절단 복원에 이걸 쓰면 엉뚱한 토큰이 붙는다. 사전토큰화본을 `max_length`로 자를 때 단순 리스트 슬라이싱(`x[:max_len]`)은 꼬리의 `<\s>`를 버리므로(HF 표준 truncation은 보존) `x[:max_len-1] + [eos_token_id]`**로 마감**한다. 앞의 `<s>`는 index 0이라 슬라이싱해도 보존된다. 영향 자체는 작다(8,192 초과 문서가 1% 미만 + `classifier_pooling: "mean"`이라 토큰 1개 손실이 평균에 미치는 영향 미미).
- **타깃**: 문서별 188 멀티핫(`labels`), sigmoid + BCE 계열 손실(baseline 정합을 위해 focal 옵션 포함 검토).
- **고정 test 원칙**: KoBERT와 **같은 test split·같은** `kobert_len` **길이 bin** 위에서 평가(`../data/data.md` 「길이 슬라이스 bin」). 비교 축을 흔들지 않기 위해 bin은 A.X 토큰이 아니라 KoBERT 토큰으로 고정한다.
- **평가**: `notebook/03_02_Metric.ipynb`(멀티라벨 micro/macro/sample-F1 + 길이 bin + 앵커 top-1 + LRAP/R-Precision). `tag`를 `axencoder_len{max_len}` 등으로 실험마다 유일하게 잡아 로짓 캐시 오염을 막는다.
- ⚠️ **평가 입력도 훈련과 같은 `max_len`으로 절단**(실측): 사전토큰화 데이터셋은 truncation 없이 전체 토큰(최대 10,523)을 담으므로, 평가 시 test를 **훈련과 동일한 `max_len`으로 절단**(`x[:max_len-1]+[eos]`)한 뒤 추론한다. 이 절단을 빠뜨리면 512 모델에 전체 문서가 들어가 훈련 창과 어긋난 무의미한 평가가 된다(exp2에서 특히 치명적 — test의 ~72%가 512 초과). KoBERT는 512로 사전토큰화된 데이터를 써 절단이 불필요했으나 ModernBERT는 무절단본이라, 평가 노트북(`04_03`·`05_02`)이 `_truncate`로 훈련 `_prep`과 동일 절단을 재현한다.
- **인프라**: Colab L4 기본(장문은 메모리를 많이 써 24GB 안전, `../infra/colab-jobs.md`). 필요 시 Lightning Job.

## 실험 목록

| # | 입력 | `max_length` | 검증 가설 | 필수 | 예상 비용 |
| --- | --- | --- | --- | --- | --- |
| **1** | full length | 8,192 (사실상 full — >8,192 극소수만 절단, 아래 「장문 처리 전략」) | **장문 무손실 → 512 truncation으로 버린 정보(주로** `claims`**)를 회복** | ✅ 필수 | ~10h |
| **2** | 512 truncation | 512 | **control**: A.X를 baseline과 같은 512 창으로 묶어, "장문 효과"에서 "모델·토크나이저 자체 효과"를 분리 | ✅ 필수 | 저렴(짧은 시퀀스) |
| **3** | full + **형식** | exp1과 동일 | 항목명으로 문서를 구조화(예: `[명칭] … [IPC] … [초록] … [청구항] …`)해 넣으면 추가 이득이 있는가 | 조건부 | ~10h |

> exp1·exp2는 완료(실측 [`modernbert-results.md`](./modernbert-results.md), 비교 [`modernbert-comparison.md`](./modernbert-comparison.md)). exp3은 아래 「실험 3 시행 판단」 참조.

## 장문 처리 전략 (exp1·exp3, full length)

full length는 극소수 장문(p99≈3,621, max 10,523)이 배치에 섞일 때 padding·메모리가 튀어 OOM·throughput 저하를 부른다. 아래 4개를 조합해 처리한다.

1. **FlashAttention-2 (FA2)** — `attn_implementation="flash_attention_2"`. 장문 attention의 메모리·속도 병목을 완화해 긴 시퀀스를 실현 가능하게 하는 전제.
2. `group_by_length=True` — 유사 길이끼리 배치로 묶어 padding 낭비 최소화. 길이 편차가 큰(중앙값 628 vs 10k대 꼬리) 이 분포에 특히 효과적.
3. `max_length=8,192` — 코퍼스 max(10,523)보다 낮게 두어 최악 배치 메모리를 bound. p99≈3,621의 한참 위라 **>8,192인 극소수(<1%)만 절단**, 사실상 full 커버리지를 유지하면서 10k대 outlier의 OOM 위험만 잘라낸다.
4. **유효 배치 등화** — 장문은 스텝당 시퀀스 수(micro-batch)를 작게 잡을 수밖에 없으므로 **gradient accumulation으로 exp2(**`max_len`**=512)와 동일한 유효 배치 크기**를 맞춘다. exp1↔exp2 비교에서 길이 외 변수(유효 배치)를 고정해 **컨텍스트 길이의 순수 기여**를 깨끗이 분리하기 위함.

**sequence packing은 제외한다.** MLM 사전학습에선 표준 기법이나, 분류 태스크에선 문서 경계 attention 마스킹·라벨 정렬 구현 비용이 커 이득 대비 부담이 크다. 위 1~4로 throughput을 확보한다.

## 핵심 비교 축

실현된 비교 결과는 [`modernbert-comparison.md`](./modernbert-comparison.md)에 있고, 여기서는 각 축의 **설계 의도**를 고정한다.

- **exp1 vs KoBERT**: 최종 headline — 장문 인코더가 baseline을 이기는가(멀티라벨 micro / 앵커 top-1 각각).
- **exp1 vs exp2**: **가장 중요한 ablation.** 같은 모델·같은 토크나이저에서 컨텍스트 길이만 다르므로, 개선분 중 **컨텍스트 길이의 순수 기여**를 분리한다. exp2가 KoBERT를 이미 이기면 개선의 상당 부분은 길이가 아니라 모델/토크나이저 우위라는 뜻(가설 반증 신호).
- **exp1 vs exp3**: **형식 구조화의 기여.** 같은 길이에서 입력 포맷만 다르므로 exp3의 값은 **오로지 exp1과의 delta로만** 해석된다 — exp3은 단독으로 해석되는 실험이 아니다.
- **길이 bin Δ(A.X − KoBERT)**: B0(≤512)에서 ≈0, B1→B3로 단조 증가하면 장문 가설 지지. 전 구간 균일 개선이면 길이 효과가 아니라 모델 자체 성능차(`../data/data.md` 「검증 로직」).

## 실험 3(형식) 시행 판단

**결론: 1·2를 먼저 완주하고, 3은 조건부·후순위로 둔다. 자르지는 않되 파일럿으로 신호를 먼저 본다.**

근거:

- **3은 코어 가설과 직교한다.** 프로젝트의 핵심 주장(장문으로 512 한계 개선)은 **1·2만으로 완결**된다. 3은 "입력 포맷"이라는 **다른 변수**를 재는 보너스 ablation이지 필수 경로가 아니다.
- **3은 단독 해석이 불가능하다.** full+형식의 성능은 그 자체로 의미가 없고 **exp1(full, 무형식)과의 delta**로만 "형식이 도움됐다"를 말할 수 있다. 즉 exp1이 이미 돌아가 있어야 3이 해석되므로, "단독 시행"은 성립하지 않는다 — 항상 exp1과 **쌍(pair)**으로만 값이 나온다.
- **10h를 null 결과에 쓸 위험.** 형식 구조화 효과는 클 수도, 무시할 수도 있다. 전량(full length, 전 epoch)으로 바로 10h를 태우기 전에 **싸게 신호를 탐지**하는 게 합리적:
  - 파일럿 A: **exp2와 짝지어** max_len=512 + 형식으로 먼저 돌린다(짧은 시퀀스라 저렴). 512 창에서 형식이 유의미하면 full에서도 기대할 수 있다.
  - 파일럿 B: train 부분표본 또는 epoch 축소로 full+형식을 조기 관찰 → 신호가 있으면 전량 승격, 없으면 1줄 negative로 종료.
- **비대칭 페이오프.** 형식이 도움되면 exp3이 최강 모델이 되고 차별점이 하나 늘어난다(low risk, moderate cost). 도움 안 돼도 "형식 구조화는 무효"라는 **깨끗한 negative**는 그 자체로 기록 가치가 있다. 다만 이 페이오프가 10h를 정당화하는지는 **1·2 결과를 본 뒤** 판단한다 — exp1이 장문 이득을 확증하지 못하면(즉 강한 full-length 베이스가 없으면) 거기에 형식을 얹는 것 자체가 무의미하다.

우선순위: **exp2(저렴) → exp1(핵심) → (exp1이 장문 이득 확증 시) exp3 파일럿 → 전량**. 예산 압박으로 하나를 접어야 하면 접는 대상은 3이다.

## 미정·다음 단계

- **손실 함수**: BCE vs focal(baseline과 정합). baseline이 focal(alpha=0.25, gamma=2)이므로 우선 정합, 이후 BCE 대조.
- **형식 스키마 확정**(exp3 진행 시): 항목명 마커 토큰 형태(`[청구항]` 등 리터럴 vs special token 추가) 결정. special token 추가 시 임베딩 확장 필요.
- **임계 튜닝**: τ=0.5가 최적이 아니라는 신호(empty rate·랭킹>임계결정) — val 임계 튜닝을 성능 레버로 남긴다([`modernbert-comparison.md`](./modernbert-comparison.md) 「멀티라벨 지표 비교」).
- **라벨 개수별 분해**: 단일 vs 다라벨(≥2) 성능 분리는 아직 미측정 — 평가 노트북에 label-cardinality bin 추가 여지.
- `max_length`**·장문 처리는 확정**(위 「장문 처리 전략」: FA2 + `group_by_length` + `max_length=8,192` + gradient accumulation 유효 배치 등화, packing 제외).
