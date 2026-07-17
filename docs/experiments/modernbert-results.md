# A.X-Encoder(ModernBERT) 실험 — 실험별 실측

> 계획·프로토콜은 [`modernbert.md`](./modernbert.md), 교차 비교·결론은 [`modernbert-comparison.md`](./modernbert-comparison.md).
> 이 문서는 **실험별 단일 모델 실측**을 소유한다(수치 SSOT: `output/total_metrics_*.json`). 모델 간 delta·분해는 비교 문서에서 다룬다.

## exp1 실측 결과 (full length 8192)

> 훈련: `notebook/04_02_ModernBERT_MaxLen.ipynb`(실행 결과 `notebook_output/04_02_ModernBERT_MaxLen_output.ipynb`, 훈련 중 test 지표 `output/modernbert-patent-len8192_test_metrics.json`).
> 지표 SSOT: 03_02 멀티라벨 프로토콜을 그대로 적용한 **전용 평가** `notebook/04_03_ModernBERT_Len8192_Metric.ipynb`(실행 결과 `notebook_output/04_03_ModernBERT_Len8192_Metric_output.ipynb`, 전체 지표 `output/total_metrics_modernbert-patent-len8192.json`). 아래 수치는 04_03 전용 평가 기준(04_02 훈련 중 지표와 4자리까지 일치).

**구성**(KoBERT 재현과 레시피 정합 — 길이·모델·토크나이저 외 변수 고정): `max_len=8,192`(>8,192 극소수 `x[:max_len-1]+[eos]`로 마감), 손실 `FocalLoss(alpha=0.25, gamma=2)`, lr 3e-5, 유효 배치 8(micro-batch×grad_accum로 512 런과 등화), `attn_implementation="flash_attention_2"`, `group_by_length`, 12에폭(global_step 302,844). **훈련 시간은 exp1 ≈29h, KoBERT 재현 ≈10h.**

### 고정 test(11,271) 비교

| 축 | 지표 | KoBERT (기준선) | ModernBERT exp1 | Δ | 상대 오차감소 |
| --- | --- | --- | --- | --- | --- |
| 멀티라벨 (τ=0.5) | micro-F1 | 0.8502 | **0.8684** | +0.0182 | 12.1% |
| | macro-F1 | 0.8470 | **0.8648** | +0.0178 | 11.6% |
| | sample-F1 | 0.8656 | **0.8825** | +0.0169 | 12.6% |
| 앵커 | top-1 weighted-F1 | 0.8148 | **0.8256** | +0.0108 | — |
| 참고 | empty rate | 1.16% | 1.35% | +0.19pt | — |

두 축은 계산이 달라 서로 뺄셈하지 않고 각 축에서 비교한다. exp1이 **두 축 모두** baseline을 이겼다(headline: 멀티라벨 micro +1.8pt, 앵커 +1.1pt). (참고: 앵커 0.8256이 공식 0.8249를 넘지만 **서로 다른 test set**이라 직접 비교 대상 아님.)

### exp1 길이 bin·랭킹 (전용 평가)

길이 bin은 KoBERT `kobert_len` 고정 축(`../data/data.md` 「길이 슬라이스 bin」).

| bin | n | micro | macro |
| --- | --- | --- | --- |
| B0 (≤512) | 3,197 | 0.8765 | 0.8662 |
| B1 (512–1024) | 5,183 | 0.8734 | 0.8696 |
| B2 (1024–2048) | 2,342 | 0.8516 | 0.8398 |
| B3 (>2048) | 549 | 0.8490 | 0.7516 |

- **랭킹**: LRAP 0.9371 / R-Precision 0.8970.
- **argmax 보정 멀티라벨**(빈 예측 문서에 argmax 1개 강제): micro 0.8675 / macro 0.8641 / sample 0.8871 — keep 대비 sample만 소폭 상승(+0.0045), 빈 예측 1.35%가 지표에 큰 왜곡을 주지 않음을 확인.
- **앵커 p@k**: p@1 0.9053 / p@3 0.9705 / p@5 0.9829.

exp1 자체 bin은 B0(micro 0.8765) → B3(0.8490)로 완만히 하락한다. 이 표는 **bin별 Δ(A.X − KoBERT)** 산출(길이 vs 모델 판별)의 A.X 쪽 입력이며, KoBERT·exp2 대조는 [`modernbert-comparison.md`](./modernbert-comparison.md) 「3-모델 bin 비교」.

### 해석

- **개선은 잡음이 아니라 견고하다.** 4개 지표(micro/macro/sample/anchor)가 일관되게 상승했고 상대 오차감소가 세 멀티라벨 지표에서 ~12%로 나란하다. 특히 KoBERT 재현에서 원본(0.8038) 아래로 내려갔던 **macro가 반전**(0.7870 → 0.8648)해, 꼬리 클래스 손해가 해소됐다.
- **이득이 truncation 상한에 맞닿는다.** KoBERT의 **B0(≤512, 잘림 없음) micro는 0.8685**였는데(`kobert-baseline.md` bin 표) exp1 **전체 micro는 0.8684**로 사실상 동일하다. 즉 장문 인코더가 전체 test를 "잘림 없는 짧은 문서" 수준으로 끌어올린 그림이다. 이 데이터셋은 KoBERT가 bin 전 구간을 0.869→0.821로 완만히만 하락(최악 B3는 549건)해 **길이가 회복 가능한 이론적 상한 자체가 크지 않다** — +1.8pt은 그 상한에 근접한 값이며, "장문이면 더 컸어야"라는 직관은 이 분포의 truncation 헤드룸을 과대평가한 것이다.
- **비용은 "저비용"으로 단정하지 않는다.** 장문인 exp1 훈련은 ≈29h로 KoBERT 재현 ≈10h의 약 3배가 들었다.
- **컨텍스트 길이에 귀속되는 몫은 +0.84pt다(exp2 control).** 측정 분해는 전체 micro +1.83pt = exp2−KoBERT **+0.99pt**(둘 다 512, 모델 성분) + exp1−exp2 **+0.84pt**(둘 다 A.X, 창 확장 성분). 후자는 같은 모델·같은 토크나이저 비교라 길이에 통제 귀속되고, 전자는 아키텍처·사전학습·토크나이저에 더해 **토크나이저 압축이 512 창에서 만든 커버리지 우위**(~10% 더 많은 본문)가 섞인 값이라 순수 아키텍처 이득으로 읽지 않는다. 상세는 [`modernbert-comparison.md`](./modernbert-comparison.md) 「길이 vs 모델 분해」.
- **empty rate 소폭 상승(1.16→1.35%).** τ=0.5가 최적이 아닐 여지 — val 임계 튜닝을 성능 레버로 남긴다.
- **과적합.** train focal loss가 1.3e-5까지 내려가(KoBERT 재현 1.96e-4보다 더 낮음) 사실상 암기 상태 — 정규화·조기중단 여지가 있으나 test 지표가 이미 개선된 상태라 후순위.

## exp2 실측 결과 (512 control)

> 훈련: `notebook/05_01_ModernBERT_Len512.ipynb`. 지표 SSOT: `notebook/05_02_ModernBERT_Len512_Metric.ipynb`(실행 결과 `notebook_output/05_02_ModernBERT_Len512_Metric_output.ipynb`, 전체 지표 `output/total_metrics_modernbert-patent-len512.json`). 평가는 훈련과 동일하게 test를 `max_len=512`로 절단 후 추론.

**구성**: exp1과 길이(`max_len=512`)만 다르고 나머지 정합 — `FocalLoss(0.25, 2)`, lr 3e-5, 유효 배치 8, `attn_implementation="flash_attention_2"`, `group_by_length`. exp1↔exp2 비교에서 컨텍스트 길이만 남기기 위한 control.

### 고정 test(11,271) 비교

| 축 | 지표 | KoBERT (기준선) | exp2 (512) | Δ | 상대 오차감소 |
| --- | --- | --- | --- | --- | --- |
| 멀티라벨 (τ=0.5) | micro-F1 | 0.8502 | **0.8601** | +0.0099 | 6.6% |
| | macro-F1 | 0.8470 | **0.8572** | +0.0102 | 6.7% |
| | sample-F1 | 0.8656 | **0.8720** | +0.0064 | 4.8% |
| 앵커 | top-1 weighted-F1 | 0.8148 | **0.8203** | +0.0055 | — |
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
- **앵커 p@k**: p@1 0.8999 / p@3 0.9661 / p@5 0.9791.
