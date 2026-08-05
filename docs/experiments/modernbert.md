# A.X-Encoder(ModernBERT) 실험 — 계획·프로토콜 (허브)

512 토큰에서 잘리는 KoBERT baseline을 **장문 인코더**(`skt/A.X-Encoder-base`, 한국어 ModernBERT, 16k 컨텍스트)로 개선하고 같은 고정 test 위에서 정량 입증하는 실험군이다. 이 문서는 그 **계획·공통 프로토콜·비교 축**을 담는 허브이며, 특히 **훈련을 조용히 망가뜨리는 함정 네 가지**(dtype·특수토큰·평가 절단·추론 배치)를 고정한다.

- 실험별 실측 → [modernbert-results.md](./modernbert-results.md)
- 교차 비교와 결론 → [modernbert-comparison.md](./modernbert-comparison.md)
- 기호·용어·런 코드 → [GLOSSARY.md](../GLOSSARY.md)

비교 기준선은 [kobert-baseline.md](./kobert-baseline.md)의 재현치다(고정 test 11,271): **top-1 weighted-F1 0.8148** / **다중 라벨 micro 0.8502 · macro 0.8470 · sample 0.8656**(τ=0.5). 두 축 각각에서 개선을 재는 것이지 두 수치를 서로 빼지 않는다.

## 현재 결론 (요약)

- **exp1(8192)이 KoBERT 재현선을 넘었다**: 다중 라벨 micro 0.8502 → **0.8685**(+1.83pt), top-1 weighted 0.8148 → **0.8256**.
- **개선 분해**(exp2 512 control): **컨텍스트 길이에 +0.84pt**(exp1−exp2 — 같은 모델·같은 토크나이저라 통제된 귀속), **모델 성분 +0.99pt**(exp2−KoBERT — 아키텍처·사전학습·토크나이저 + 512 창에서의 커버리지 우위가 섞인 값이며 더 쪼개지 않는다). 창 확장 효과는 문서가 길수록 커져 최장 문서군(B3)에서 최대(+2.64pt).
- **다중 라벨 전 지표에서 exp1 > exp2 > KoBERT 단조**(F1 micro/macro/sample · LRAP · R-Precision · P@k). 상세는 [`modernbert-results.md`](./modernbert-results.md)·[`modernbert-comparison.md`](./modernbert-comparison.md).

## 공통 프로토콜 (모든 실험 고정)

- **모델**: `skt/A.X-Encoder-base` (한국어 ModernBERT, 컨텍스트 16,384, vocab 49,999, apache-2.0). 토크나이저 revision `9708f9c4`로 pin.
- **데이터**: `ingyoun/patent-clean-text-modernbert-tokenized` (train 201,895 / val 11,162 / test 11,271). 사전토큰화 완료본을 그대로 소비 — 입력 조합·토크나이즈는 상류(`notebook/04_01_Prep_ModernBERT.ipynb`)에서 1회 수행. **truncation 없이 최대 10,523토큰**으로 저장돼 있어 `max_length`는 소비 시점(훈련 config)에서 건다.
- **입력 필드**: `invention_title + ipc_main + abstract + claims` (공백 join, 빈 필드 skip) — KoBERT baseline과 동일 필드 집합.
- **타깃**: 문서별 188 다중-핫(`labels`), sigmoid + BCE 계열 손실(baseline 정합을 위해 focal 옵션 포함 검토).
- **고정 test 원칙**: KoBERT와 **같은 test split, 같은 `kobert_len` 길이 bin** 위에서 평가한다(`../data/data.md` 「길이 bin」). 비교 축을 흔들지 않기 위해 bin은 A.X 토큰이 아니라 KoBERT 토큰으로 고정한다.
- **평가**: `notebook/03_02_Metric.ipynb`(다중 라벨 micro/macro/sample-F1 + 길이 bin + top-1 weighted + LRAP/R-Precision). `tag`를 `axencoder_len{max_len}` 등으로 실험마다 유일하게 잡아 로짓 캐시 오염을 막는다.
- **인프라**: 이 축의 초기 실험은 Colab L4에서 돌았다(장문은 메모리를 많이 써 24GB가 안전하다, `../infra/colab-jobs.md`). 훈련 코드를 패키지로 굳힌 뒤의 장시간 런은 RunPod 팟이 맡는다(`../infra/runpod-jobs.md`).

## 훈련·평가를 망가뜨리는 함정 네 가지

네 가지 모두 **조용히 실패한다** — 예외가 나지 않고 지표만 틀린다. 새 런을 걸기 전에 확인한다.

### ⚠️ 1. dtype — 모델이 상수 분류기로 붕괴한다

**증상**: 훈련이 정상으로 보이는데 모델이 다수 클래스만 출력하는 상수 분류기가 된다.

**원인**: `skt/A.X-Encoder-base`는 **bf16으로 저장**돼 있고(config `torch_dtype: "bfloat16"`), transformers **v5는 `dtype`을 지정하지 않으면 체크포인트 dtype 그대로 bf16으로 로드**한다(v4는 fp32로 올려 받아 이 문제가 보이지 않았다). bf16 파라미터를 `bf16=True`(autocast)로 훈련하면 옵티마이저가 bf16 파라미터를 직접 갱신하는데, 작은 갱신량(lr 3e-5 × 기울기)이 8비트 가수부에서 **언더플로로 소멸**한다. 가중치가 얼어붙는 것이며, 디스크의 가중치가 손상된 것이 아니라 업데이트가 사라지는 것이다.

**조치**: 로드 시 `dtype=torch.float32`를 **명시**한다 → fp32 마스터 가중치 + autocast(bf16) 연산이라는 표준 혼합정밀 레시피가 된다.

- FA2는 autocast가 bf16 q/k/v를 공급하므로 fast path와 속도 이득이 그대로 유지된다. 추가 비용은 파라미터 메모리 2배뿐이고 연산 시간 손해는 사실상 없다.
- **fp16은 쓰지 않는다** — ModernBERT의 활성값이 fp16 범위를 넘겨 NaN이 난다. 안전한 조합은 훈련 파라미터 fp32, 추론 bf16이다.
- 추론(`04_03`)도 `dtype=torch.float32` 로드 + `torch.autocast(bfloat16)`로 훈련과 같은 경로를 쓴다.
- 로드 시 transformers가 `Flash Attention 2 only supports torch.float16 and torch.bfloat16 … current dtype is torch.float32` 경고를 내지만 무해하다. 파라미터 저장 dtype에 대한 알림일 뿐이고 실제 연산은 autocast가 bf16으로 수행한다.

### ⚠️ 2. 특수토큰 — 절단하면 마감 토큰이 사라진다

A.X-Encoder는 시퀀스를 `<s>`(0) … 본문 … `</s>`(1)로 감싼다. 마감 토큰은 **`eos_token_id`(=1)**이며, `tokenizer.sep_token_id`는 `<sep>`(=3)으로 **실제 마감 토큰이 아니다.** 절단 복원에 `sep_token_id`를 쓰면 엉뚱한 토큰이 붙는다.

사전토큰화본을 `max_length`로 자를 때 단순 리스트 슬라이싱(`x[:max_len]`)은 꼬리의 `</s>`를 버린다(HF 표준 truncation은 보존한다). 따라서 `x[:max_len-1] + [eos_token_id]`로 마감한다. 앞의 `<s>`는 index 0이라 슬라이싱해도 보존된다.

영향 자체는 작다 — 8,192를 넘는 문서가 1% 미만이고 `classifier_pooling`이 `"mean"`이라 토큰 하나의 손실이 평균에 미치는 영향이 미미하다.

### ⚠️ 3. 평가 절단 — 빠뜨리면 평가가 무의미해진다

사전토큰화 데이터셋은 절단 없이 전체 토큰(최대 10,523)을 담는다. 그러므로 평가할 때도 test를 **훈련과 동일한 `max_len`으로 절단**(`x[:max_len-1]+[eos]`)한 뒤 추론해야 한다.

이 절단을 빠뜨리면 512 모델에 문서 전체가 들어가 훈련 창과 어긋난 무의미한 평가가 된다. 512 대조 런에서 특히 치명적인데, test의 약 72%가 512를 넘기 때문이다. KoBERT는 512로 사전토큰화된 데이터를 써서 절단이 불필요했으나 ModernBERT는 무절단본이라, 평가 노트북(`04_03`·`05_02`)이 `_truncate`로 훈련의 `_prep`과 동일한 절단을 재현한다.

### ⚠️ 4. 추론 배치 크기 — 지표의 네 번째 자리가 바뀐다

평가와 로짓 덤프는 **batch 8로 고정**한다(`03_02`·`04_03`·`05_02`·`06_00` 전부 동일).

`EvalCollator`의 동적 패딩(`padding=True`)이 배치 내 최장 문서에 맞추므로, 배치 크기를 바꾸면 패딩량과 행렬 모양이 바뀐다. 그러면 bf16 autocast의 누산 순서와 cuBLAS 커널 선택이 달라져 **로짓이 약 1e-4 흔들린다**. 그 자체는 무해하지만, τ=0.5 경계와 top-1 argmax에서 문서 몇 건이 뒤집혀 지표가 네 번째 자리에서 어긋난다.

로짓 재덤프(`06_00`)에서 batch를 64로 올린 두 모델만 SSOT와 불일치했고(mb512 top-1 weighted-F1 0.8203→0.8199, micro 0.8601→0.8600), batch 8을 유지한 `modernbert-patent-len8192`만 다섯 지표 전부 네 자리까지 일치했다. 이 대조는 **체크포인트·절단·행 순서·dtype 경로가 정확하다는 증거**이기도 하다. 아울러 토크나이저를 체크포인트가 아니라 base(`skt/A.X-Encoder-base` revision `9708f9c4`)에서 로드해도 결과가 같음을 확인했다.

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

- **exp1 vs KoBERT**: 최종 대표 비교 — 장문 인코더가 baseline을 이기는가(다중 라벨 micro / top-1 weighted 각각).
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

- **손실 함수**: 카디널리티 회수를 겨냥한 손실 교체(ZLPR·ASL·DL2). 계획·후보 평가·프로토콜은 [`loss-function.md`](./loss-function.md).
- **형식 스키마 확정**(exp3 진행 시): 항목명 마커 토큰 형태(`[청구항]` 등 리터럴 vs special token 추가) 결정. special token 추가 시 임베딩 확장 필요.
- **임계 튜닝**: τ=0.5가 최적이 아니라는 신호(empty rate·랭킹>임계결정) — val 임계 튜닝을 성능 레버로 남긴다. 실행 계획은 [`no-train-analysis.md`](./no-train-analysis.md) B.
- **라벨 개수별 분해**: 단일 vs 다라벨(≥2) 성능 분리는 아직 미측정. 실행 계획은 [`no-train-analysis.md`](./no-train-analysis.md) A(오류 분해와 함께 수행).
- `max_length`**·장문 처리는 확정**(위 「장문 처리 전략」: FA2 + `group_by_length` + `max_length=8,192` + gradient accumulation 유효 배치 등화, packing 제외).

## 기각된 후보

- **토크나이저 도메인 튜닝(vocab 확장·특허 코퍼스 재학습) — 기각.** 근거는 세 가지다.
  1. **exp1의 coverage가 이미 1.0000**(test 전량, `notebook/06_03` 실측). 토크나이저를 압축적으로 만들어 토큰 수를 줄여도 8,192 창에 **추가로 담을 본문이 없다** — 커버리지 채널이 소진된 상태라 이득의 경로가 없다. 512 모델에는 의미가 있으나 512는 control이지 결과물이 아니다.
  2. **A.X 과분절은 실측되지 않았다.** 과분절은 KoBERT(vocab 8,002)의 열화 신호로 관측된 것이고, A.X는 KLUE-RoBERTa와 동급이다(어절당 2.503 vs 2.471조각, 3+조각 37.81% vs 36.07%). 개선 여지의 근거가 없다.
  3. **어절당 조각 수는 과분절과 정상 형태소 분절을 구분하지 못한다.** 한국어 어절은 어간+조사 구조라 `반도체의` → `[반도체][##의]`(2조각)는 올바른 분절이다. 2.503이 과분절인지 형태소 바닥에 가까운지 이 지표로는 판정할 수 없어, 튜닝의 목표치조차 세울 수 없다.
- 비용·부작용도 불리하다: vocab 확장은 새 토큰 임베딩이 랜덤 초기화라 특허 코퍼스 continued MLM pretraining 없이는 정착하지 않고(exp3 ~10h를 크게 상회), 모델이 바뀌어 exp1·exp2·exp3의 비교선이 오염된다.
