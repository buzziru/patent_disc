# 도메인 사전학습(TAPT) — 자체 특허 코퍼스 MLM

일반 도메인 백본(`skt/A.X-Encoder-base`)을 자체 특허 코퍼스로 MLM 계속학습(TAPT)한 뒤 분류 파인튜닝해, **백본 표현 품질이 남은 레버인지**를 재는 축이다. 축의 성격·폐기된 대안(KorPatBERT 계열 이용 불가, 토크나이저 갈래, 임베딩-only 계속학습)은 `../../NEXT_SESSION.md` 「도메인 사전학습 축」에 있다.

이 문서는 **MLM 단계의 실측과 아티팩트**, 그리고 분류 파인튜닝의 **판정 규약**을 담는다.

| 단계 | 노트북 | 상태 |
| --- | --- | --- |
| MLM 계속학습 | `notebook_output/13_01_TAPT_MLM.ipynb` | 완료 — 5 epoch 완주 |
| 분류 파인튜닝 | `notebook_output/13_02_TAPT_Train.ipynb` | 완료 — **기준 미달**(정리 test micro 0.8572, 앵커 −0.15pt) |

## MLM 프로토콜

### 코퍼스와 누수 차단

`ingyoun/patent-clean-text-modernbert-tokenized`(정리 데이터)의 **train split만 학습에 넣는다**. eval은 val split이고, **test는 MLM에 전혀 들어가지 않는다**. val 역시 loss 관측용일 뿐이다 — `load_best_model_at_end`를 쓰지 않아 val이 가중치 선택에도 관여하지 않고, 최종본은 마지막 에폭 그대로다.

### 청킹 — 절단 대신 비겹침 분할

저장된 `input_ids`는 truncation이 없어 최대 10,523 토큰이다. `max_len=2048`로 단순 절단하면 >2048 문서(3.67%)의 꼬리가 버려져 **총 토큰의 5.87%(약 9.4M)** 가 학습에서 사라진다. 대신 각 문서를 `stride = max_len − 2 = 2046`으로 비겹침 분할하고 청크마다 `<s>…</s>`를 다시 씌워(사전학습 포맷 일치) 전량을 학습에 넣는다. 짧은 문서(≤2048)는 원본과 동일하다. `labels`(다중핫)·`document_id` 등 MLM에 불필요한 컬럼은 제거한다.

| split | 문서 | 청크 | 증가 | 총 토큰 |
| --- | ---: | ---: | ---: | ---: |
| train | 201,616 | 211,159 | +4.7% | 160.4M |
| val | 11,132 | 11,658 | +4.7% | 8.85M |

청킹은 문서를 나눌 뿐 토큰을 버리지 않으므로 epoch당 토큰은 [data.md](../data/data.md)의 train 총량과 같다(청크 경계에 특수토큰 2개가 추가되는 분만 차이). **5 epoch = 약 802M 토큰**으로, 계획 예산(0.5~1B)의 중간이다.

### 레시피

ModernBERT 사전학습 레시피를 따르되 작은 코퍼스에 맞춰 lr·epoch를 보수적으로 잡았다.

| 항목 | 값 | 비고 |
| --- | --- | --- |
| `mlm_probability` | 0.30 | ModernBERT 사전학습 관례(BERT의 0.15가 아니다) |
| lr | 5e-5 | linear · `warmup_ratio` 0.06 |
| `weight_decay` | 1e-5 | |
| `adam_beta2` / `adam_epsilon` | 0.98 / 1e-6 | ModernBERT 사전학습 레시피 |
| `eff_batch` | 128 | `micro_batch` 4 × `grad_accum` 32 |
| epoch / step | 5 / 8,250 | 1,650 step/epoch |
| 정밀도·어텐션 | bf16 · `flash_attention_2` | len8192 계열과 구현 일치 |
| 샘플링 | `group_by_length` | 유사 길이 배치로 패딩 최소화 |
| seed | 42 | 단일 시드 방법론([ADR-0011](../adr/0011-resource-constrained-methodology.md)) |

### 하드웨어와 소요

Colab L4(24GB). `probe_batches`(len2048, 모든 시퀀스가 최대 길이·패딩 0인 최악 조건) 실측 상한:

| 경로 | 통과 | OOM |
| --- | --- | --- |
| train | micro 4 (peak 12.9GB) | micro 8 |
| eval | micro 4 (peak 21.6GB) | micro 8 |

MLM eval은 50,000 vocab 로짓이 메모리를 지배해 train보다 peak가 크다 — 분류(188 로짓)의 `eval_micro_batch=512` 감각을 그대로 옮기면 첫 에폭 끝에서 죽는다.

런은 epoch 2 뒤 한 번 중단됐고 `checkpoint-3300`에서 재개했다. 재개 호출 구간(step 3,300→8,250, 3 epoch + eval 3회)의 실측 소요는 30,700s ≈ 8.5h이므로 **epoch당 ≈2.8h, 5 epoch 총 ≈14h**다(앞 2 epoch 구간은 별도 계측이 남지 않았다). 재개가 스케줄을 보존한 것은 `max_steps=8250` 유지와 최종 lr 6.4e-9(선형 소멸 종점)로 확인된다.

## MLM 실측

| epoch | step | train loss | val loss |
| ---: | ---: | ---: | ---: |
| 1 | 1,650 | — | 0.4272 |
| 2 | 3,300 | 0.4062 | 0.4042 |
| 3 | 4,950 | — | 0.3917 |
| 4 | 6,600 | — | 0.3793 |
| 5 | 8,250 | 0.3713 | 0.3742 |

(step 50 시점 train loss 0.9389 → 전 구간 하강.)

- **과적합 신호가 없다.** val loss가 5 epoch 내내 단조 하강하고 train과의 격차도 벌어지지 않는다(ep5에서 train 0.3713 대 val 0.3742). "작은 코퍼스라 MLM 과적합" 우려는 이 예산에서는 실현되지 않았다 — 역으로 **5 epoch가 수렴점이 아니라 예산 종료**이며, 분류 이득이 확인되면 epoch 연장이 남은 여지다.
- MLM loss 자체는 판정 지표가 아니다. 판정은 아래 분류 파인튜닝 결과로만 한다.

## 아티팩트

| 항목 | 값 |
| --- | --- |
| repo | `ingyoun/A.X-patent-tapt-mlm` |
| rev | `62818c2595513a03f834c39a329c375153bc2661` |
| 백본 레지스트리 키 | `axenc_tapt` (`src/patent_train/backbones.py`) |

`hub_strategy="all_checkpoints"`로 push돼 repo에 5개 에폭 체크포인트(optimizer state 포함)가 함께 남아 있다 — 총 9.57GB 중 루트 0.60GB, 체크포인트 8.97GB.

### 아티팩트 검증 (재현 절차)

Trainer가 마지막에 "End of training" 커밋을 남기지 않아 커밋 목록만으로는 완주 여부가 보이지 않는다. 다음 세 가지로 확인한다.

1. **루트 가중치 = 최종 에폭**: 루트 `model.safetensors`의 sha256이 `checkpoint-8250/model.safetensors`와 일치(`d676a7ae…`). `checkpoint-8250/trainer_state.json`이 `global_step 8250 / epoch 5.0`을 확인한다.
2. **토크나이저 동일성**: `skt/A.X-Encoder-base@9708f9c`와 vocab(50,000)·`bos`/`eos`/`pad`/`mask` id·인코딩 결과가 완전히 일치한다. TAPT는 vocab을 바꾸지 않으므로 **기존 토큰화 데이터셋(`dataset_id`)을 그대로 쓴다.**
3. **헤드 초기화 대칭**: `skt/A.X-Encoder-base`의 `architectures`도 `ModernBertForMaskedLM`이다. 따라서 `AutoModelForSequenceClassification` 로드 시 앵커·TAPT 양쪽 모두 MLM의 `head.*`(ModernBertPredictionHead)가 전이되고 `classifier`만 신규 초기화된다 — **TAPT 쪽만 헤드가 사전학습돼 유리해지는 교락은 없다.** 로드 시 `decoder.*` 드롭 경고는 정상이다.

## 분류 파인튜닝 (`13_02`)

앵커 `11_01`에서 **백본 키 하나만** 바꾼 단일 변수 대조다. 손실·길이·배치·lr·스케줄·early stop이 모두 동일하다.

- **설정**: `backbone="axenc_tapt"` · focal(α=0.25, γ=2) · len512 · `eff_batch` 128 · lr 4.8e-4 · linear · `warmup_ratio` 0.1 · 12 epoch · `early_stop_epochs=2` · eval·save 에폭당 2회.
- **산출**: `ingyoun/A.X-patent-len512-tapt`(Hub) · `modernbert-patent-len512-tapt_metrics.json` · val/test 로짓 덤프.

### 판정 기준

- **비교선은 `11_01`(정리 test micro 0.8588)이다.** `exp1`(정리 test 0.8683)은 len8192라 백본 효과에 입력 길이가 섞인다 — TAPT가 512에서 이득을 내면 그때 장문 설정으로 확장해 `exp1`과 대면한다.
- **유의 폭은 +0.4pt 이상**이다. 정리 test(11,244)에서 paired 델타의 표본 잡음 sd가 0.18~0.21pt이므로 그 미만은 잡음으로 설명된다([eval-noise.md](eval-noise.md)). 판정에는 로짓 덤프 기반 paired bootstrap(`scripts/eval_noise_bootstrap.py`)을 쓴다.
- 부수 관측으로 **길이 bin별 slope 불변**을 재확인한다. 이 축의 동기는 slope 개선이 아니라 전 구간 level 상승이므로([longdoc-degradation.md](longdoc-degradation.md)), slope이 펴지지 않는 것은 실패 신호가 아니다.
- 유의 폭을 못 넘으면 축을 접는다. full DAPT(3B 커리큘럼) 확장은 TAPT 신호가 확인된 뒤에만 검토한다.

### 실측 — 기준 미달

12 epoch 완주(18,912 step, 8h 14m). 정리 test(11,244) 지표는 두 런 모두 최종 에폭 가중치에서 잰 값이다.

| 지표 | `13_02`(TAPT) | `11_01`(앵커) | Δ (pt) |
| --- | ---: | ---: | ---: |
| micro-F1 | 0.8572 | 0.8588 | −0.15 |
| macro-F1 | 0.8545 | 0.8565 | −0.20 |
| sample-F1 | 0.8724 | 0.8738 | −0.14 |
| anchor weighted-F1 | 0.8198 | 0.8215 | −0.16 |
| empty rate | 0.0119 | 0.0117 | +0.02 |

**판정: 유의 폭 +0.4pt에 미달하고 부호가 음수 — 축을 접는다.** 네 지표가 동시에 −0.14~−0.20pt로 부호가 일치하지만 폭이 표본 잡음 sd(0.18~0.21pt) 규모라 "TAPT가 해롭다"로도 읽지 않는다. 이 예산의 TAPT는 백본 표현 품질을 분류 성능으로 **옮기지 못했다**. MLM loss가 5 epoch 내내 단조 하강한 것과 무관하게 하류 이득이 없으므로, MLM loss는 이 축의 대리 지표가 아님이 실측으로 확인된다.

### 훈련 곡선 — 초기 우위는 없다

val micro-F1을 eval 지점(788 step, 에폭당 2회)마다 짝지은 델타(TAPT − 앵커) 24개의 분포:

| 구간 | eval 수 | 평균 Δ (pt) |
| --- | ---: | ---: |
| 전체 | 24 | −0.03 |
| ~3 epoch | 6 | +0.04 |
| ~6 epoch(전반) | 12 | +0.05 |
| 6 epoch~(후반) | 12 | −0.11 |
| 마지막 3 epoch | 6 | −0.17 |

24개 델타의 sd는 **0.43pt**이고 양수는 10개다. 초기 우위로 보이는 것은 ep1.0(+0.69)·ep1.5(+0.67) 두 점인데 바로 다음 ep2.0이 −1.29로 되돌린다 — 3 epoch까지의 평균은 +0.04pt로 사실상 0이다. 두 런은 서로 다른 최적화 궤적이라 eval별 델타 산포가 **고정 모델의 표본 잡음(sd ~0.2pt)보다 크다**. 한두 eval 지점의 부호를 신호로 읽지 않는다.

형상 자체는 [training-curves.md](training-curves.md)의 4런 공통 형상과 같다 — val loss는 중반에 최저(0.000416~0.000426)를 찍고 끝에서 두 배로 오르나 micro-F1은 마지막 eval이 최고점이다. 두 런 모두 early stop이 발동하지 않고 12 epoch를 완주했고 best = 마지막 체크포인트다.

### 로짓 덤프 결함 — 폐기

`13_02`가 낸 `logits_modernbert-patent-len512-tapt_{val,test}.npy`는 **데이터셋 행 순서와 어긋난 순열**이었다. 순열은 복원 불가이고 축이 종결돼 재덤프 가치가 없으므로 **두 파일은 삭제했다.** 진단 근거는 다음과 같다.

- 정리 test 정답과 대면하면 micro **0.0059** · top-1 **0.0062**(우연 수준)인데, 문서당 예측 수는 1.22개로 정상이다 — **모델은 정상이고 행 축만 어긋났다.**
- 앵커 로짓과의 행별 코사인이 self **0.051** vs 최적 매칭 **0.800**이다. 정렬이 맞는 두 런(exp2 vs `11_01`)에서는 self 0.630 ≈ 최적 0.653으로, 참 짝이 곧 최적 짝이다.
- 클래스축 순열 가설은 배제된다 — 헝가리안 최적 배정 뒤에도 micro 0.042에 그친다.

런이 낸 지표(micro 0.8572)는 Trainer가 라벨을 같은 순서로 모아 계산하므로 **유효하다.** 어긋난 것은 덤프 파일뿐이며, 판정은 그 지표 위에 선다.

대가로 프로토콜이 요구한 **paired bootstrap CI**와 부수 관측인 **길이 bin별 slope**는 미집행으로 남는다. 점추정 −0.15pt가 +0.4pt로 바뀔 수 없어 판정 자체는 흔들리지 않는다. 로짓이 필요한 작업(오류 분석·KD teacher 편입)을 훗날 되살리려면 `11_03`과 같은 추론 전용 경로로 다시 덤프해야 한다 — `TrainConfig.for_inference(checkpoint="ingyoun/A.X-patent-len512-tapt")` → `predict_logits`. 모델 자체는 Hub에 남아 있다.

### 왜 효과가 없었나

**TAPT 코퍼스가 파인튜닝 코퍼스와 같은 문서 집합인데, 노출량은 파인튜닝 쪽이 더 크다.** 같은 train split 201,616 문서에 대해

| 단계 | 설정 | 총 토큰 |
| --- | --- | ---: |
| TAPT MLM | 5 epoch · len2048 청킹(절단 없음) | 802M |
| 분류 파인튜닝 | 12 epoch · len512 절단(문서당 평균 461 토큰) | **1,116M** |

파인튜닝이 같은 텍스트를 1.4배 더 보고, 게다가 라벨 감독이 붙어 있다. MLM이 짜낼 수 있는 분포 정보를 파인튜닝이 이미 더 많이·더 강한 신호로 보고 있으므로 **TAPT가 새로 넣는 정보가 사실상 0**이다. TAPT의 원 논문(Gururangan et al. 2020)에서 이득이 큰 쪽은 라벨 데이터가 수천~수만 건인 저자원 과제다. 「한계」 절이 이미 "도메인 적응이 아니라 과제 코퍼스 적응"이라 적어 둔 그 성질이 곧 실패 기제다.

**표현이 병목이 아님은 이미 실측돼 있었다.** 최장 문서에서도 정답이 top-5에 98.1%·top-3에 96.4% 잔존하고([longdoc-degradation.md](longdoc-degradation.md)), 고정 풀링 프로브가 표현 헤드룸을 <~0.5pt로 경계지었다([ADR-0012](../adr/0012-representation-pooling-closure.md)). 남은 결손은 확신 오답 1,500~2,000개가 손실의 73~82%를 지고([training-curves.md](training-curves.md)) 그 다수가 cross-`Lno` 두 번째 라벨·형제 혼동이다([hierarchy-loss.md](hierarchy-loss.md)). 백본 표현 품질을 겨냥하는 레버가 결정·라벨 구조에 있는 병목을 옮길 수 없다 — **ADR-0012가 풀링 쪽에서 만난 벽을 사전학습 쪽에서 다시 만난 것**이다.

**수렴 가속조차 없었다.** TAPT에서 가장 흔히 보고되는 이득은 최종 성능이 아니라 초기 수렴인데, 첫 eval(ep0.5)이 0.6385 대 0.6380으로 동일하다. 백본이 과제 텍스트에 미리 적응돼 있었다면 여기서 벌어져야 한다.

**토크나이저는 손대지 않았다.** vocab 50,000을 유지했으므로(그래야 기존 토큰화 데이터셋과 단일 변수 대조가 성립한다) 분절 수준의 도메인 불일치는 애초에 이 축의 사정권이 아니다. 다만 A.X 토크나이저의 한국어 특허 압축률은 이미 실측 우위라([no-train-analysis.md](no-train-analysis.md) C) 큰 레버도 아니었다.

### MLM 체크포인트를 바꿨다면

`13_01`이 `hub_strategy="all_checkpoints"`로 5개 에폭 체크포인트를 남겨 두었으나, **에폭을 바꿔 재시도할 근거가 없다.** TAPT 용량-반응 곡선의 양 끝이 이미 측정돼 있기 때문이다 — TAPT 0 epoch(=앵커 `11_01`, 백본 원본)가 0.8588, TAPT 5 epoch가 0.8572로 **양 끝의 간격이 0.15pt이고 표본 잡음 sd(0.18~0.21pt) 안**이다. 중간 체크포인트는 이 두 끝 사이의 용량이므로, 결과가 달라지려면 곡선이 양 끝보다 0.4pt 이상 위로 볼록해야 한다.

- MLM val loss가 5 epoch 내내 **단조 하강**해(0.4272→0.3742) "중간이 최적"을 시사하는 신호가 MLM 쪽에 없다.
- 중간이 더 나을 유일한 기제인 **파국적 망각**이라면 ep5가 앵커보다 뚜렷이 나빠야 하는데 −0.15pt(잡음 내)다 — 망각도 거의 없었다. 즉 TAPT 용량을 늘리든 줄이든 아무 일도 일어나지 않는 구간에 있다.
- 무엇보다 위의 코퍼스 동일성은 **용량과 무관한 이유**다. 없는 정보는 노출 시간을 조절해도 생기지 않는다.

비용도 맞지 않는다. 체크포인트 1개 검증 = 12 epoch 파인튜닝 8h 14m이고, 4개 스윕이면 ~33h GPU다. 단일 시드 방법론([ADR-0011](../adr/0011-resource-constrained-methodology.md))에서 훈련 잡음이 미측정이라 잡음 폭 안에 다 들어올 4점 스윕은 **잡음을 재는 데 33시간을 쓰는 일**이 된다.

**이 축을 되열 조건은 epoch가 아니라 코퍼스다** — train split 밖의 외부 특허 문헌으로 수십억 토큰을 넣는 DAPT라야 파인튜닝이 보지 못하는 정보가 생긴다. 실패한 것은 "계속학습이라는 원리"가 아니라 "라벨 코퍼스와 같은 코퍼스로 계속학습한 설정"이다.

## 함정

- **MLM 학습 길이(2048)와 분류 파인튜닝 길이(512)가 다르다.** 의도된 설계다 — MLM은 코퍼스 전량을 노출시키려 2048 청킹을 쓰고, 분류는 앵커와의 단일 변수 대조를 위해 512로 고정한다. 두 길이를 맞추는 것은 대조를 깨는 변경이다.
- **`prep_cache` 키는 `{backbone}_len{max_len}`이라 `axenc_tapt_len512`가 새로 생긴다.** 토크나이저가 동일해 내용은 `axenc_len512`와 같지만 캐시는 공유되지 않는다 — 팟 볼륨을 재사용해도 원본 재다운로드·재map이 1회 발생한다(구 데이터로 학습될 위험은 없다).
- **MLM eval 배치를 분류 감각으로 잡지 않는다.** 위 probe 표 참조 — vocab 로짓 크기가 다르다.
- **`hub_strategy="all_checkpoints"`는 optimizer state까지 올린다.** 5 epoch 런이 9.6GB를 차지했다. 재개용으로는 유용하나, 축이 끝나면 체크포인트 디렉터리 정리를 고려한다.
- **훈련 경로 Trainer로 `predict_logits`를 부르면 행 순열 위험이 남는다.** `train_sampling_strategy="group_by_length"`는 eval·predict 로더에도 적용되며, `predict_logits`가 덤프 동안 `sequential`로 되돌리는 방어와 행 순서 assert가 `13_02`에서 순열을 막지 못했다(`11_01`에서 한 번 발생해 `11_03`으로 재덤프한 것과 같은 실패 모드다). **로짓은 훈련이 끝난 뒤 추론 전용 경로(`TrainConfig.for_inference` → `_build_inference_trainer`)에서 별도 덤프한다.** 덤프 직후 SSOT micro와 대조하는 검증(`11_03` 마지막 셀)을 같은 노트북 안에서 돌려 순열을 그 자리에서 잡는다.

## 한계

- **단일 시드 1회 런**이다([ADR-0011](../adr/0011-resource-constrained-methodology.md)). 훈련 잡음은 미측정이므로 판정 여유가 유의 폭에 겨우 걸치면 확정하지 않는다([eval-noise.md](eval-noise.md)).
- TAPT 코퍼스가 분류 train split과 **같은 문서 집합**이다. 라벨은 쓰지 않았고 test는 제외했으므로 평가 누수는 없으나, 도메인 적응이라기보다 **과제 코퍼스 적응**(TAPT의 정의 그대로)이며 외부 특허 코퍼스로 넓힌 DAPT와는 다른 것을 재고 있다.
- 5 epoch는 수렴이 아니라 예산 종료다 — 이 결과로 "TAPT 포화"를 주장할 수 없다.
