# 다음 세션 인계

> 읽는 순서: `PROJECT.md`(SSOT) → 이 문서. 수치 SSOT는 `output/*.json`, 손실 축은 `docs/experiments/loss-function.md`.

## 데이터 클리닝 — 완료·Hub 미반영 (전 지표에 영향)

`documentId` 재분할이 못 거른 **입력 동일 케이스**를 제거했다. `documentId`는 특허 문서를 식별하지 정규화된 모델 입력을 식별하지 않아, 서로 다른 `documentId`인데 모델 입력(`title+ipc_main+abstract+claims`)이 바이트 단위로 같은 문서가 있었다. 검출·제거는 `notebook/10_03_Label_Conflict_Clean.ipynb`, 사실·근거는 `docs/data/data.md` 「주의」, 제거 목록 SSOT는 `output/label_conflict_docs.json`.

- **제거 336문서**: 라벨 충돌 89(42그룹, 전원) · train 내 정확-중복 199(사본 1개 유지) · eval 정확-중복 누수 45(val 20+test 25, train 사본 유지·eval 제거) · val+test 정확-중복 3(test 유지·val 제거). 제거 후 train 201,616 / val 11,132 / test 11,244(총 223,992), 잔여 충돌 0.
- **지금까지의 모든 지표는 정리 이전 split에서 측정됐으나 영향은 무시할 수준(실측).** 구 test(11,271)에서 27건(정확-중복 eval 누수 25 + 충돌 2)을 뺀 정리 test(11,244)로 저장 로짓을 재계산한 결과 전 모델 micro **−0.02~0.03pt**(exp1 0.8685→**0.8683**, exp2 0.8601→0.8599, KoBERT 0.8502→0.8500), 방향은 하락(암기로 맞힌 누수 행 제거). **모든 서열·격차 불변**(exp1>exp2>KoBERT, exp1−exp2 +0.84pt). 재훈련 불필요. SSOT `output/headline_cleaned_test.json`.
- **미반영 작업**: ① `10_03`을 `DRY_RUN=False`로 실행해 `ingyoun/patent-clean-text`(정리 base)·`...-modernbert-tokenized`(필터본) push. ② RoBERTa·KoBERT 토큰화본도 같은 336 `document_id`로 필터링(재토큰화 불필요 — 순서·`document_id` 정합 확인 후 동일 적용). ③ 필요 시 정리된 test로 headline 재측정.

## 지금 상태

레시피(배치·LR) 확정이 끝나고 손실 함수 축이 진행 중이다. 지금은 손실 A/B의 두 번째 후보 **ASL(`09_02`)이 RunPod에서 훈련 중**이다.

- **레시피 확정**(`08_01` lr1e-4 / `08_02` lr5e-4 탐색): **len512 · eff_batch 128 · lr 4.8e-4 · linear · 12 epoch · warmup_ratio 0.1 · group_by_length.** long-document 축 비교용으로 고정했던 `eff_batch=8`을 128로 올리고 LR을 선형에 가깝게 4.8e-4로 맞췄다. 머신은 L4 → A40.
- **손실 축.** 판정 축은 멀티라벨 micro-F1, 표적은 **다라벨 문서 카디널리티 회수**(오라클-k 상한 대비 회수율). 계획·후보 평가·프로토콜·실측은 `docs/experiments/loss-function.md`(SSOT).
  - **ZLPR(`09_01`) — 음성, 미채택.** test micro **0.8493**로 비교 기준선 exp2 focal(0.8601)을 1.08pt 밑돔. 적응 임계는 작동(empty_rate 0.96%)했으나 회수한 예측이 진양성보다 거짓양성 쪽이었다. 카디널리티 1.2·단일 라벨 86%에서 랭킹 손실 이득이 나지 않는다는 단서의 실현.
  - **ASL(`09_02`) — RunPod 훈련 중.** γ+=0 / γ−=4 / margin=0.05. ⚠️ 아래 「즉시 확인」.
  - **DL2(`09_03`) — scaffold, 미착수.**

현재 최고 성능(full run): **exp1 A.X@8192 micro 0.8685**. 512 계열 최고는 exp2 focal 0.8601. 새 레시피(eff128/lr4.8e-4)에서의 **focal full run은 없다** — ZLPR A/B는 `08_02` focal 2-epoch 탐색 궤적으로 대조했다(1.5 epoch에 val micro 0.835로 exp2 수렴 궤도).

| 런 | max_len | 손실 | micro-F1 | 비고 |
| --- | --- | --- | --- | --- |
| KoBERT 재현 | 512 | focal | 0.8502 | 비교 기준점 |
| exp2 (A.X) | 512 | focal | 0.8601 | 손실 A/B 기준선 (구 레시피 eff8/lr3e-5) |
| exp1 (A.X) | 8192 | focal | **0.8685** | 최고 full run |
| ZLPR (A.X) | 512 | ZLPR | 0.8493 | 신 레시피 첫 full run, 미채택 |
| ASL (A.X) | 512 | ASL | — | 훈련 중 |

## ⚠️ 즉시 확인 — ASL LR 불일치

`09_02` 로컬 노트북 config는 `learning_rate: 3e-5`인데, **확정 레시피이자 ZLPR A/B가 쓴 값은 4.8e-4**다(16× 차이). 손실 A/B는 레시피를 고정하고 손실만 바꿔야 성립한다 — ASL을 3e-5로 돌리면 "손실 효과"와 "16× 낮은 LR"이 섞이고, eff_batch 128에서 undertrain될 공산이 커 ASL이 실제보다 나쁘게 나온다. **RunPod에서 도는 값이 3e-5인지 확인하고, 그렇다면 4.8e-4로 고쳐 재제출한다.**

## 손실 A/B — 남은 일

> 프로토콜·판정 기준은 `docs/experiments/loss-function.md` 「프로토콜」. 아래는 착수 요약.

- **ASL 완료 시**: test micro/macro/sample + **k≥2 슬라이스 micro 회수율**(오라클-k 상한 exp1 +1.63pt 대비) + 진단(k≥2 평균 예측 라벨 수(기준 정답 2.355)·과소예측 문서 비율(기준 44.3%)·empty rate·val 최적 τ)을 내어 `loss-function.md` 「실측」에 반입. exp2(0.8601)·ZLPR(0.8493) 대비 판정. m이 점수 분포를 밀 수 있으니 **val 최적 τ가 0.5에서 벗어나는지 진단**(튜닝이 아니라 정합 확인).
- **DL2**: 착수 전 **변형을 못박는다** — per-class(macro) dice는 배치 내 클래스별 양성 희소로 불안정하고, pooled dice는 안정적이나 F1-대리 매력이 흐려진다(매력과 위험이 같은 축). 예산이 남을 때만.

## 함정 (놓치기 쉬움)

- **ASL margin m이 점수 분포를 이동**시켜 val 최적 τ를 0.5에서 밀어낼 수 있다. τ 튜닝을 배제한 프로젝트라 τ=0.5 정합이 이득 실현의 전제 — val에서 진단한다.
- **현행 대조 손실은 BCE가 아니라 `FocalLoss(alpha=0.25, gamma=2)`다.** `alpha`가 손실 전체에 곱해지는 상수라 클래스 균형 역할을 하지 않아 비대칭 처리가 사실상 γ 하나뿐이다(문서당 음성:양성 ≈156:1). ASL·ZLPR이 겨냥하는 지점이다.
- **`eval_steps`·`save_steps`는 이제 `steps_per_epoch` 기반 자동 계산**(에폭당 2회)이라 배치 하드코딩 문제는 해소됐다. 배치를 다시 바꾸면 자동으로 따라온다.
- **exp1은 12에폭을 전부 소진**하고 early stopping이 한 번도 발동하지 않았다(수렴이 아니라 예산 종료). linear 스케줄이 12에폭에 맞춰 LR을 0으로 감쇠시킨다 — 훈련 길이·스케줄 형태는 손실 축 종료 후 검토 대상.
- **`classifier_pooling`은 이미 `mean`**(A.X-Encoder-base 기본값). CLS 병목 우려는 이 모델에 해당하지 않는다.

## 확정된 사실 · 정정

- **주 비교 지표는 멀티라벨 micro-F1.** top-1 weighted-F1과 P@1/3/5는 벤더 baseline의 레거시이며 병기용(anchor).
- **임계값은 레버가 아니다.** global τ 오라클 micro 헤드룸 +0.0000 ~ +0.0008. 문서 상대 임계값(`p ≥ α·p_max`)은 sample-F1을 +0.66 ~ 0.85pt 올리나 micro는 −0.23 ~ −0.52pt로 떨어뜨린다. 카디널리티 헤드룸은 **손실이 점수 분포를 재배치해야** 회수된다.
- **앙상블은 채택하지 않는다.** 단일 모델 운영 요구. 3모델 로짓평균 micro 0.8685 → 0.8752(+0.67pt)이나 추론 비용 3배.
- 나머지 닫힌 갈래는 `PROJECT.md` 「닫힌 갈래」 표를 따른다 — 계층 확장·임계값 정책·캘리브레이션·입력 필드 실험·long-document 추가 검증. **새 근거 없이 다시 제안하지 않는다.**

## 열린 질문

- **주 모델을 8192로 갈지 512로 갈지** — 운영 단일 모델 관점의 추론 비용 차이. 손실 축 종료 후 결정한다.
- **훈련 길이·LR 스케줄 형태**(linear 유지 여부) — exp1이 스케줄을 소진한 사실과 함께 검토한다.
- **KLUE-RoBERTa-base 대조군**(선택) — 성능이 아니라 주장 방어(크기 confound 제거)가 목적. 예산이 남을 때만.

## 작업 규약

- 노트북 작성은 **지시가 있을 때만** 한다. 손실 노트북은 `09_02`(ASL)·`09_03`(DL2)까지 있다.
- 검증은 산출물에서 재현되게 한다 — 판정 기준은 서술 통계가 아니라 **결정 질문에 직접 답하는 양**(오라클·회수율·학습 곡선)으로 세우고, `### verify` 셀에 assert와 SSOT 대조를 남긴다.
