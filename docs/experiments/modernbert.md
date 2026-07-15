# A.X-Encoder(ModernBERT) 실험 — 계획과 프로토콜

> **목적**: 512-truncation KoBERT baseline을 **장문 인코더**(`skt/A.X-Encoder-base`, 한국어 ModernBERT, 16k 컨텍스트)로 개선하고, 같은 고정 test 위에서 정량 입증한다. 이 문서는 실험 **계획·공통 프로토콜·비교 축**을 고정하고, 실행 결과는 실측 후 갱신한다.
>
> 기준선(`../experiments/kobert-baseline.md`, 고정 test 11,271): **앵커 top-1 weighted-F1 0.8148** / **멀티라벨 micro 0.8502 · macro 0.8470 · sample 0.8656** (τ=0.5). A.X-Encoder는 이 두 축 각각에 대해 개선을 재는 것이지, 두 수치를 서로 뺄셈하지 않는다.

## 공통 프로토콜 (모든 실험 고정)

- **모델**: `skt/A.X-Encoder-base` (한국어 ModernBERT, 컨텍스트 16,384, vocab 49,999, apache-2.0). 토크나이저 revision `9708f9c4`로 pin.
- **데이터**: `ingyoun/patent-clean-text-modernbert-tokenized` (train 201,895 / val 11,162 / test 11,271). 사전토큰화 완료본을 그대로 소비 — 입력 조합·토크나이즈는 상류(`notebook/04_01_Prep_ModernBERT.ipynb`)에서 1회 수행. **truncation 없이 최대 10,523토큰**으로 저장돼 있어 `max_length`는 소비 시점(훈련 config)에서 건다.
- **입력 필드**: `invention_title + ipc_main + abstract + claims` (공백 join, 빈 필드 skip) — KoBERT baseline과 동일 필드 집합.
- ⚠️ **토크나이저 특수토큰 함정**(실측): A.X-Encoder는 시퀀스를 **`<s>`(0) … 본문 … `<\s>`(1)** 로 감싼다. 마감 토큰은 **`eos_token_id`(=1)**이며, **`tokenizer.sep_token_id`는 `<sep>`(=3)으로 실제 마감 토큰이 아니다** — 절단 복원에 이걸 쓰면 엉뚱한 토큰이 붙는다. 사전토큰화본을 `max_length`로 자를 때 단순 리스트 슬라이싱(`x[:max_len]`)은 꼬리의 `<\s>`를 버리므로(HF 표준 truncation은 보존) **`x[:max_len-1] + [eos_token_id]`로 마감**한다. 앞의 `<s>`는 index 0이라 슬라이싱해도 보존된다. 영향 자체는 작다(8,192 초과 문서가 1% 미만 + `classifier_pooling: "mean"`이라 토큰 1개 손실이 평균에 미치는 영향 미미).
- **타깃**: 문서별 188 멀티핫(`labels`), sigmoid + BCE 계열 손실(baseline 정합을 위해 focal 옵션 포함 검토).
- **고정 test 원칙**: KoBERT와 **같은 test split·같은 `kobert_len` 길이 bin** 위에서 평가(`../data/data.md` 「길이 슬라이스 bin」). 비교 축을 흔들지 않기 위해 bin은 A.X 토큰이 아니라 KoBERT 토큰으로 고정한다.
- **평가**: `notebook/03_02_Metric.ipynb`(멀티라벨 micro/macro/sample-F1 + 길이 bin + 앵커 top-1 + LRAP/R-Precision). `tag`를 `axencoder_len{max_len}` 등으로 실험마다 유일하게 잡아 로짓 캐시 오염을 막는다.
- **인프라**: Colab L4 기본(장문은 메모리를 많이 써 24GB 안전, `../infra/colab-jobs.md`). 필요 시 Lightning Job.

## 실험 목록

| # | 입력 | `max_length` | 검증 가설 | 필수 | 예상 비용 |
| --- | --- | --- | --- | --- | --- |
| **1** | full length | 8,192 (사실상 full — >8,192 극소수만 절단, 아래 「장문 처리 전략」) | **장문 무손실 → 512 truncation으로 버린 정보(주로 `claims`)를 회복** | ✅ 필수 | ~10h |
| **2** | 512 truncation | 512 | **control**: A.X를 baseline과 같은 512 창으로 묶어, "장문 효과"에서 "모델·토크나이저 자체 효과"를 분리 | ✅ 필수 | 저렴(짧은 시퀀스) |
| **3** | full + **형식** | exp1과 동일 | 항목명으로 문서를 구조화(예: `[명칭] … [IPC] … [초록] … [청구항] …`)해 넣으면 추가 이득이 있는가 | 조건부 | ~10h |

## 장문 처리 전략 (exp1·exp3, full length)

full length는 극소수 장문(p99≈3,621, max 10,523)이 배치에 섞일 때 padding·메모리가 튀어 OOM·throughput 저하를 부른다. 아래 4개를 조합해 처리한다.

1. **FlashAttention-2 (FA2)** — `attn_implementation="flash_attention_2"`. 장문 attention의 메모리·속도 병목을 완화해 긴 시퀀스를 실현 가능하게 하는 전제.
2. **`group_by_length=True`** — 유사 길이끼리 배치로 묶어 padding 낭비 최소화. 길이 편차가 큰(중앙값 628 vs 10k대 꼬리) 이 분포에 특히 효과적.
3. **`max_length=8,192`** — 코퍼스 max(10,523)보다 낮게 두어 최악 배치 메모리를 bound. p99≈3,621의 한참 위라 **>8,192인 극소수(<1%)만 절단**, 사실상 full 커버리지를 유지하면서 10k대 outlier의 OOM 위험만 잘라낸다.
4. **유효 배치 등화** — 장문은 스텝당 시퀀스 수(micro-batch)를 작게 잡을 수밖에 없으므로 **gradient accumulation으로 exp2(`max_len`=512)와 동일한 유효 배치 크기**를 맞춘다. exp1↔exp2 비교에서 길이 외 변수(유효 배치)를 고정해 **컨텍스트 길이의 순수 기여**를 깨끗이 분리하기 위함.

**sequence packing은 제외한다.** MLM 사전학습에선 표준 기법이나, 분류 태스크에선 문서 경계 attention 마스킹·라벨 정렬 구현 비용이 커 이득 대비 부담이 크다. 위 1~4로 throughput을 확보한다.

## 핵심 비교 축

- **exp1 vs KoBERT**: 최종 headline — 장문 인코더가 baseline을 이기는가(멀티라벨 micro / 앵커 top-1 각각).
- **exp1 vs exp2**: **가장 중요한 ablation.** 같은 모델·같은 토크나이저에서 컨텍스트 길이만 다르므로, 개선분 중 **컨텍스트 길이의 순수 기여**를 분리한다. exp2가 KoBERT를 이미 이기면 개선의 상당 부분은 길이가 아니라 모델/토크나이저 우위라는 뜻(가설 반증 신호).
- **exp1 vs exp3**: **형식 구조화의 기여.** 같은 길이에서 입력 포맷만 다르므로 exp3의 값은 **오로지 exp1과의 delta로만** 해석된다 — exp3은 단독으로 해석되는 실험이 아니다.
- **길이 bin Δ(A.X − KoBERT)**: B0(≤512)에서 ≈0, B1→B3로 단조 증가하면 장문 가설 지지. 전 구간 균일 개선이면 길이 효과가 아니라 모델 자체 성능차(`../data/data.md` 「검증 로직」).

## exp1 실측 결과 (full length 8192)

> 실행: `notebook/04_02_ModernBERT_MaxLen.ipynb`, 실행 결과 `notebook_output/04_02_ModernBERT_MaxLen_output.ipynb`, 테스트 지표 `output/modernbert-patent-len8192_test_metrics.json`.

**구성**(KoBERT 재현과 레시피 정합 — 길이·모델·토크나이저 외 변수 고정): `max_len=8,192`(>8,192 극소수 `x[:max_len-1]+[eos]`로 마감), 손실 `FocalLoss(alpha=0.25, gamma=2)`, lr 3e-5, 유효 배치 8(micro-batch×grad_accum로 512 런과 등화), `attn_implementation="flash_attention_2"`, `group_by_length`, 12에폭(global_step 302,844). **비용은 두 층위로 구분해 읽는다.** *내재 연산시간*(`train_runtime`)은 33,291s(≈9.25h) — KoBERT 재현 27,037s(≈7.5h) 대비 +23%로, 8,192 full-length 커버리지를 얻고도 증가가 완만하다(median 628토큰·꼬리만 장문 + FA2 + `group_by_length`, 코퍼스 FLOPs ~2배). *운영 wall-clock*은 이와 다르다: 두 런 모두 Colab 세션 한계로 중단·재개를 반복했고, 장문인 exp1은 재시작·체크포인팅 부담이 커 벽시계가 **≈29h(KoBERT ≈10h)**로 훨씬 컸다. 즉 한계 연산비는 낮아도 **반복 실행의 운영 비용은 작지 않았다** — 비용을 인용할 때 둘을 섞지 않는다.

### 고정 test(11,271) 비교

| 축 | 지표 | KoBERT (기준선) | ModernBERT exp1 | Δ | 상대 오차감소 |
| --- | --- | --- | --- | --- | --- |
| 멀티라벨 (τ=0.5) | micro-F1 | 0.8502 | **0.8684** | +0.0182 | 12.1% |
| | macro-F1 | 0.8470 | **0.8648** | +0.0178 | 11.6% |
| | sample-F1 | 0.8656 | **0.8825** | +0.0169 | 12.6% |
| 앵커 | top-1 weighted-F1 | 0.8148 | **0.8257** | +0.0109 | — |
| 참고 | empty rate | 1.16% | 1.34% | +0.18pt | — |

두 축은 계산이 달라 서로 뺄셈하지 않고 각 축에서 비교한다. exp1이 **두 축 모두** baseline을 이겼다(headline: 멀티라벨 micro +1.8pt, 앵커 +1.1pt). (참고: 앵커 0.8257이 공식 0.8249를 넘지만 **서로 다른 test set**이라 직접 비교 대상 아님.)

### 해석

- **개선은 잡음이 아니라 견고하다.** 4개 지표(micro/macro/sample/anchor)가 일관되게 상승했고 상대 오차감소가 세 멀티라벨 지표에서 ~12%로 나란하다. 특히 KoBERT 재현에서 원본(0.8038) 아래로 내려갔던 **macro가 반전**(0.7870 → 0.8648)해, 꼬리 클래스 손해가 해소됐다.
- **이득이 truncation 상한에 맞닿는다.** KoBERT의 **B0(≤512, 잘림 없음) micro는 0.8685**였는데(`../experiments/kobert-baseline.md` bin 표) exp1 **전체 micro는 0.8684**로 사실상 동일하다. 즉 장문 인코더가 전체 test를 "잘림 없는 짧은 문서" 수준으로 끌어올린 그림이다. 이 데이터셋은 KoBERT가 bin 전 구간을 0.869→0.821로 완만히만 하락(최악 B3는 549건)해 **길이가 회복 가능한 이론적 상한 자체가 크지 않다** — +1.8pt은 그 상한에 근접한 값이며, "장문이면 더 컸어야"라는 직관은 이 분포의 truncation 헤드룸을 과대평가한 것이다.
- **비용은 "저비용"으로 단정하지 않는다.** 한계 연산비는 완만하나(FLOPs ~2배, `train_runtime` +23%), 중단·재개 반복으로 exp1 운영 wall-clock은 ≈29h(KoBERT ≈10h)로 컸다. 연산 효율(장문의 낮은 한계비용)과 iteration 비용(긴 벽시계·재시작 부담)은 층위가 달라 나눠 기록한다.
- **개선의 원인(길이 vs 모델)은 아직 단정 불가 — aggregate만으로는 양쪽이 동률로 정합한다.** 전체 micro 0.8684가 KoBERT B0 상한(0.8685)에 닿은 것은 두 시나리오와 **똑같이** 부합한다: (a) 장문이 B1~B3의 truncation 손실을 회복해 상한으로 수렴, (b) 더 강한 모델이 B0 포함 전 구간을 균일하게 +1.8pt 끌어올림. aggregate 한 점으로는 (a)와 (b)를 가를 수 없다. 특히 A.X-Encoder는 vocab 49,999로 KoBERT SentencePiece(≈8,002)보다 한국어 분절이 크게 개선돼 있어 **길이와 무관한 모델·토크나이저 기여만으로도 상당 폭이 설명될 수 있다** — 따라서 "시퀀스 길이가 큰 원인"이라는 결론은 현재 근거로는 이르다. **exp2(512 control)**와 **bin별 Δ(A.X − KoBERT)** 표가 이를 가른다: 길이 효과면 Δ가 B0≈0 → B1~B3 증가, 모델 효과면 Δ가 전 구간 균일. → exp2 우선 실행.
- **empty rate 소폭 상승(1.16→1.34%).** τ=0.5가 최적이 아닐 여지 — val 임계 튜닝을 성능 레버로 남긴다.
- **과적합.** train focal loss가 1.3e-5까지 내려가(KoBERT 재현 1.96e-4보다 더 낮음) 사실상 암기 상태 — 정규화·조기중단 여지가 있으나 test 지표가 이미 개선된 상태라 후순위.

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
- **`max_length`·장문 처리는 확정**(위 「장문 처리 전략」: FA2 + `group_by_length` + `max_length=8,192` + gradient accumulation 유효 배치 등화, packing 제외).
