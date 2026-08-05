# A.X-Encoder(ModernBERT) 실험 — 실험별 실측

A.X-Encoder 실험의 **런별 실측치**를 모은 문서다. 모델 간 델타와 그 분해는 비교 문서가 담고, 여기서는 각 런이 단독으로 낸 값만 기록한다. 수치 SSOT는 `output/total_metrics_*.json`이다.

- 계획·프로토콜 → [modernbert.md](./modernbert.md)
- 교차 비교·결론 → [modernbert-comparison.md](./modernbert-comparison.md)
- 기호·용어·런 코드 → [GLOSSARY.md](../GLOSSARY.md)

## exp1 실측 결과 (full length 8192)

> 훈련: `notebook/04_02_ModernBERT_MaxLen.ipynb`(실행 결과 `notebook_output/04_02_ModernBERT_MaxLen_output.ipynb`, 훈련 중 test 지표 `output/modernbert-patent-len8192_test_metrics.json`).
> 지표 SSOT: 03_02 다중 라벨 프로토콜을 그대로 적용한 **전용 평가** `notebook/04_03_ModernBERT_Len8192_Metric.ipynb`(실행 결과 `notebook_output/04_03_ModernBERT_Len8192_Metric_output.ipynb`, 전체 지표 `output/total_metrics_modernbert-patent-len8192.json`). 아래 수치는 04_03 전용 평가 기준(04_02 훈련 중 지표와 4자리까지 일치).

**구성**(KoBERT 재현과 레시피 정합 — 길이·모델·토크나이저 외 변수 고정): `max_len=8,192`(>8,192 극소수 `x[:max_len-1]+[eos]`로 마감), 손실 `FocalLoss(alpha=0.25, gamma=2)`, lr 3e-5, 유효 배치 8(micro-batch×grad_accum로 512 런과 등화), `attn_implementation="flash_attention_2"`, `group_by_length`, 12에폭(global_step 302,844 = 25,237 steps/epoch × 12 — **early stopping이 발동하지 않고 스케줄을 전부 소진**했다. linear 스케줄이 12에폭에 맞춰 LR을 0으로 감쇠시키므로, 런은 수렴해서가 아니라 예산이 끝나서 종료됐다). **훈련 시간은 exp1 ≈29h, exp2(512) ≈14h, KoBERT 재현 ≈10h.**

### 고정 test(11,271) 비교

| 축 | 지표 | KoBERT (기준선) | ModernBERT exp1 | Δ | 상대 오차감소 |
| --- | --- | --- | --- | --- | --- |
| 다중 라벨 (τ=0.5) | micro-F1 | 0.8502 | **0.8684** | +0.0182 | 12.1% |
| | macro-F1 | 0.8470 | **0.8648** | +0.0178 | 11.6% |
| | sample-F1 | 0.8656 | **0.8825** | +0.0169 | 12.6% |
| baseline 정합 | top-1 weighted-F1 | 0.8148 | **0.8256** | +0.0108 | — |
| 참고 | empty rate | 1.16% | 1.35% | +0.19pt | — |

두 축은 계산이 달라 서로 뺄셈하지 않고 각 축에서 비교한다. exp1이 **두 축 모두** baseline을 이겼다(대표 수치: 다중 라벨 micro +1.8pt, 기준 런 +1.1pt). (참고: 기준 런 0.8256이 공식 0.8249를 넘지만 **서로 다른 test set**이라 직접 비교 대상 아님.)

### exp1 길이 bin·랭킹 (전용 평가)

길이 bin은 KoBERT `kobert_len` 고정 축(`../data/data.md` 「길이 슬라이스 bin」).

| bin | n | micro | macro |
| --- | --- | --- | --- |
| B0 (≤512) | 3,197 | 0.8765 | 0.8662 |
| B1 (512–1024) | 5,183 | 0.8734 | 0.8696 |
| B2 (1024–2048) | 2,342 | 0.8516 | 0.8398 |
| B3 (>2048) | 549 | 0.8490 | 0.7516 |

- **랭킹**: LRAP 0.9371 / R-Precision 0.8970.
- **argmax 보정 다중 라벨**(빈 예측 문서에 argmax 1개 강제): micro 0.8675 / macro 0.8641 / sample 0.8871 — keep 대비 sample만 소폭 상승(+0.0045), 빈 예측 1.35%가 지표에 큰 왜곡을 주지 않음을 확인.
- **top-1 p@k**: p@1 0.9053 / p@3 0.9705 / p@5 0.9829.

exp1 자체 bin은 B0(micro 0.8765) → B3(0.8490)로 완만히 하락한다. 이 표는 **bin별 Δ(A.X − KoBERT)** 산출(길이 vs 모델 판별)의 A.X 쪽 입력이며, KoBERT·exp2 대조는 [`modernbert-comparison.md`](./modernbert-comparison.md) 「3-모델 bin 비교」.

### 해석

- **개선은 잡음이 아니라 견고하다.** 4개 지표(micro·macro·sample·top-1 weighted)가 일관되게 상승했고 상대 오차감소가 세 다중 라벨 지표에서 ~12%로 나란하다. 특히 KoBERT 재현에서 원본(0.8038) 아래로 내려갔던 **macro가 반전**(0.7870 → 0.8648)해, 꼬리 클래스 손해가 해소됐다.
- **이득이 truncation 상한에 맞닿는다.** KoBERT의 **B0(≤512, 잘림 없음) micro는 0.8685**였는데(`kobert-baseline.md` bin 표) exp1 **전체 micro는 0.8684**로 사실상 동일하다. 즉 장문 인코더가 전체 test를 "잘림 없는 짧은 문서" 수준으로 끌어올린 그림이다. 이 데이터셋은 KoBERT가 bin 전 구간을 0.869→0.821로 완만히만 하락(최악 B3는 549건)해 **길이가 회복 가능한 이론적 상한 자체가 크지 않다** — +1.8pt은 그 상한에 근접한 값이며, "장문이면 더 컸어야"라는 직관은 이 분포의 truncation 헤드룸을 과대평가한 것이다.
- **비용은 "저비용"으로 단정하지 않는다.** 장문인 exp1 훈련은 ≈29h로 KoBERT 재현 ≈10h의 약 3배가 들었다.
- **컨텍스트 길이에 귀속되는 몫은 +0.84pt다(exp2 control).** 측정 분해는 전체 micro +1.83pt = exp2−KoBERT **+0.99pt**(둘 다 512, 모델 성분) + exp1−exp2 **+0.84pt**(둘 다 A.X, 창 확장 성분). 후자는 같은 모델·같은 토크나이저 비교라 길이에 통제 귀속되고, 전자는 아키텍처·사전학습·토크나이저에 더해 **토크나이저 압축이 512 창에서 만든 커버리지 우위**(~10% 더 많은 본문)가 섞인 값이라 순수 아키텍처 이득으로 읽지 않는다. 상세는 [`modernbert-comparison.md`](./modernbert-comparison.md) 「길이 vs 모델 분해」.
- **empty rate 소폭 상승(1.16→1.35%).** τ=0.5가 최적이 아닐 여지 — val 임계 튜닝을 성능 레버로 남긴다.
- **과적합.** train focal loss가 1.3e-5까지 내려가(KoBERT 재현 1.96e-4보다 더 낮음) 사실상 암기 상태 — 정규화·조기중단 여지가 있으나 test 지표가 이미 개선된 상태라 후순위.

## exp2 실측 결과 (512 control)

> 훈련: `notebook/05_01_ModernBERT_Len512.ipynb`. 지표 SSOT: `notebook/05_02_ModernBERT_Len512_Metric.ipynb`(실행 결과 `notebook_output/05_02_ModernBERT_Len512_Metric_output.ipynb`, 전체 지표 `output/total_metrics_modernbert-patent-len512.json`). 평가는 훈련과 동일하게 test를 `max_len=512`로 절단 후 추론.

**구성**: exp1과 길이(`max_len=512`)만 다르고 나머지 정합 — `FocalLoss(0.25, 2)`, lr 3e-5, 유효 배치 8, `attn_implementation="flash_attention_2"`, `group_by_length`. exp1↔exp2 비교에서 컨텍스트 길이만 남기기 위한 control. **훈련 시간 ≈14h.**

### 고정 test(11,271) 비교

| 축 | 지표 | KoBERT (기준선) | exp2 (512) | Δ | 상대 오차감소 |
| --- | --- | --- | --- | --- | --- |
| 다중 라벨 (τ=0.5) | micro-F1 | 0.8502 | **0.8601** | +0.0099 | 6.6% |
| | macro-F1 | 0.8470 | **0.8572** | +0.0102 | 6.7% |
| | sample-F1 | 0.8656 | **0.8720** | +0.0064 | 4.8% |
| baseline 정합 | top-1 weighted-F1 | 0.8148 | **0.8203** | +0.0055 | — |
| 참고 | empty rate | 1.16% | 1.79% | +0.63pt | — |

exp2는 **512 창에서도 네 지표 모두 KoBERT를 넘는다** — 개선의 일부가 길이가 아니라 **모델·토크나이저 자체**에서 온다는 직접 증거(가설 반증 신호였으나, exp1이 exp2를 다시 이기므로 길이 효과도 함께 실재).

### exp2 길이 bin·랭킹

| bin | n | micro | macro |
| --- | --- | --- | --- |
| B0 (≤512) | 3,197 | 0.8719 | 0.8606 |
| B1 (512–1024) | 5,183 | 0.8661 | 0.8631 |
| B2 (1024–2048) | 2,342 | 0.8398 | 0.8284 |
| B3 (>2048) | 549 | 0.8226 | 0.7286 |

- **랭킹**: LRAP 0.9318 / R-Precision 0.8895.
- **top-1 p@k**: p@1 0.8999 / p@3 0.9661 / p@5 0.9791.

## clean-data 512 실측 (08_01 레시피)

> 훈련: `notebook/11_01_CleanData_Recipe.ipynb`(실행 결과 `notebook_output/11_01_CleanData_Recipe.ipynb`). 지표 SSOT: `output/modernbert-patent-len512-b128_metrics.json`(훈련 중 test 평가 — exp1/exp2의 전용 평가와 달리 length bin·ranking 분해는 미산출). 데이터는 [ADR-0010](../adr/0010-data-cleaning.md) 클리닝본(입력-동일 라벨 충돌·중복 제거, test 11,244).

**구성**: exp2와 손실(`FocalLoss(0.25, 2)`)·길이(`max_len=512`)는 같고 **레시피와 데이터가 다르다** — `08_01/08_02`로 확정한 eff_batch 128 · lr 4.8e-4(linear scaling) · 12 epoch, clean 데이터. 손실 축(`09_xx`)이 갖지 못했던 **레시피 정합 focal 기준선**이다.

**test**: micro **0.8588** · macro 0.8565 · sample 0.8738 · empty_rate 1.17% · top-1 weighted 0.8215.

### 해석

- **focal micro는 레시피에 불변이다.** eff8/lr3e-5(exp2 0.8601)과 eff128/lr4.8e-4(0.8588)이 잡음 내에서 같다 — 배치·lr 16배 상향은 micro를 움직이지 않는다. 단일 run이라 ±0.1~0.2pt대 seed 잡음과 −0.13pt를 구분하지 않는다.
- **배치 상향의 이득은 캘리브레이션 축에 나타난다.** empty rate가 exp2 1.79%→1.17%로 내렸다. `09_xx` eff128 런이 손실 무관하게 focal eff8보다 낮은 empty rate를 보여([loss-function.md](loss-function.md) 실측) 이 이동을 손실이 아니라 **배치 성분**으로 귀속한다. micro 상한은 512 focal에서 이미 포화에 가까워 잡음 감소가 천장을 밀지 못한다.
- **손실 축 종결을 보강한다.** 동일 레시피(eff128/lr4.8e-4)에서 focal이 BCE·ZLPR·ASL을 모두 앞선다([loss-function.md](loss-function.md) 실측 표) — 손실 후보들의 열세가 레시피 교란 요인이 아님을 레시피 정합 focal로 확정한다([ADR-0009](../adr/0009-loss-axis-closure.md)).
- **클리닝 성분은 이 런에서 분리되지 않는다.** 이 런과 exp2를 같은 정리 test에서 대조한 per-class paired diff-in-diff(충돌 연루 74클래스 대 비연루 114클래스)가 유의하지 않다(`11_02`, `output/error_analysis_cleandata_vs_exp2.json`) — 판정·수치는 [ADR-0010](../adr/0010-data-cleaning.md) 「결과·영향」.

## 배포 런 실측 (`16_01`, max_len 4096)

> 훈련: `notebook_output/16_01_Model_4096.ipynb`. 지표 SSOT: `output/modernbert-patent-len4096-op_metrics.json`. 데이터는 [ADR-0010](../adr/0010-data-cleaning.md) 클리닝본(test 11,244), 모델 `ingyoun/A.X-patent-len4096-op`.

**구성**: `11_01`에서 `max_len`만 4096으로 바꾼 단일 변수 런이다 — `FocalLoss(0.25, 2)` · eff_batch 128 · lr 4.8e-4 · 12 epoch · seed 42 동일, `micro_batch=16`(grad_accum 8). 18,912 step 완주.

**test**: micro **0.8660** · macro 0.8638 · sample 0.8835 · empty_rate 0.83% · top-1 weighted 0.8251 · P@1 0.9051.

- 기준 런 `11_01` 대비 **+0.72pt**(paired bootstrap CI95 [+0.33, +1.10]), exp1(정리 재계산 0.8683)과는 **구분되지 않는다**(−0.23pt, CI95 [−0.60, +0.13]).
- 이득이 길이 bin에서 단조 증가하고(≤512 +0.29 → 1024–2048 +1.58pt) exp1 대비 격차는 길이와 무관하게 흩어져, 4096의 길이 부채 추정 ≲0.03pt가 유지된다.
- 헤드라인·오류 구조·한계 기술의 전체 실측은 [final-run.md](final-run.md)가 소유한다.

## 전 런 대조 — test micro-F1

프로젝트가 낸 모든 full run을 한 표에 둔다. 축별 판정과 기제는 각 축 문서에 있으며 여기서는 서열만 본다.

| 런 | max_len | 손실 | 레시피 | micro-F1 | 비고 |
| --- | --- | --- | --- | --- | --- |
| KoBERT 재현 | 512 | focal | eff8/lr3e-5 | 0.8502 | 비교 기준점([kobert-baseline.md](kobert-baseline.md)) |
| exp2 (A.X) | 512 | focal | eff8/lr3e-5 | 0.8601 | 512 계열 최고·손실 A/B 기준선 |
| **exp1 (A.X)** | **8192** | focal | eff8/lr3e-5 | **0.8685** | **최고 full run** |
| ZLPR (A.X) | 512 | ZLPR | eff128/lr4.8e-4 | 0.8493 | 미채택([ADR-0009](../adr/0009-loss-axis-closure.md)) |
| ASL (A.X) | 512 | ASL | eff128/lr4.8e-4 | 0.8362 | 미채택 |
| BCE (A.X) | 512 | BCE | eff128/lr4.8e-4 | 0.8538 | 진단(γ의 순수 값 −0.62pt, 시드 취약) |
| `11_01` (A.X) | 512 | focal | eff128/lr4.8e-4 | 0.8588 | 정리 데이터·신 레시피 첫 focal 풀런 — **현행 기준 런** |
| `13_02` (A.X TAPT) | 512 | focal | eff128/lr4.8e-4 | 0.8572 | 도메인 축 종결(기준 런 −0.15pt, 잡음 내) |
| `11_04` (A.X seed153) | 512 | focal | eff128/lr4.8e-4 | 0.8570 | 시드 축 측정용 기준 런 재현(Δ −0.176pt) |
| `14_01` (A.X MCLoss) | 512 | focal+MCL λ0.0444 | eff128/lr4.8e-4 | 0.8467 | 계층 손실 1런(기준 런 −1.20pt, **잡음 밖**) |
| **`16_01` (A.X)** | **4096** | focal | eff128/lr4.8e-4 | **0.8660** | **배포 모델** — 기준 런 +0.72pt(잡음 밖) |

- **exp1~BCE는 구 test(11,271) 기준이고 `11_01`·`13_02`·`11_04`·`14_01`·`16_01`만 정리 test(11,244)다.** 정리 test 재계산값은 exp1 0.8683 · exp2 0.8599 · KoBERT 0.8500이며 서열·격차는 불변이다(`output/headline_cleaned_test.json`). **서로 다른 test에서 잰 micro를 나란히 놓지 않는다** — `16_01`과 exp1의 0.8683 대조는 이 재계산값으로 한다.
- 배포 런(`16_01`, max_len 4096)의 결정·근거·오류 구조는 [final-run.md](final-run.md)에 있다.
