# 지식 증류(KD) 실험 — 계획·프로토콜 (이종 앙상블 → 단일 student)

> **목적**: 이종 teacher 앙상블의 측정된 헤드룸을 **단일 배포 모델**로 증류해 회수한다. [ADR-0005](../adr/0005-no-ensemble.md)가 접은 앙상블 이득(단일 모델 제약 때문에 미회수)을 **추론 시점 단일 모델**로 가져오는 경로다 — 앙상블은 훈련 시점 teacher로만 쓰고 배포는 student 하나다(닫힌 갈래 「앙상블」의 재제안이 아니다). 손실 축([ADR-0009](../adr/0009-loss-axis-closure.md))이 회수하지 못한 **k≥2 카디널리티**를 겨냥하는 대안 경로이기도 하다.
>
> 판정 축은 **멀티라벨 micro-F1**(`PROJECT.md` 평가 절), 표적 슬라이스는 **k≥2 micro**다. 이득 여부는 신뢰가 아니라 A/B로 가른다.
>
> 선행 게이트 완료 — 아래 「헤드룸 게이트」. **판정: GREEN(증류할 상한 실재).**

## 배경 — 왜 KD인가

- **응답 기반(response-based) KD는 토크나이저에 무관하다.** teacher의 soft target은 **공유 188-Mno 출력 공간**의 확률이지 토큰 표현이 아니다. 따라서 서로 다른 토크나이저·아키텍처(KoBERT wordpiece / A.X BPE)의 이종 teacher를 한 student로 증류할 수 있다. hidden-state 증류였다면 토크나이저 불일치로 막히나, 출력 공간이 같아 이종성이 오히려 다양성 자산이 된다.
- **단일 모델 제약을 어기지 않는다.** 앙상블은 soft target을 오프라인으로 생성하는 데만 쓴다. student는 추론에서 모델 하나다 — [ADR-0005](../adr/0005-no-ensemble.md)의 "3배 추론 비용" 문제가 없다.
- **손실이 못 넘은 벽의 대안.** [ADR-0009](../adr/0009-loss-axis-closure.md)는 전역 손실 재배치가 k=1 과대예측·k≥2 과소예측의 FP:FN 부호 뒤집힘을 동시에 풀 수 없다고 종결했다. 게이트에서 앙상블은 **k≥2 micro를 +1.42pt** 회수한다 — 손실이 아니라 teacher 다양성이 이 슬라이스를 움직인다.

## 헤드룸 게이트 — 완료

훈련 없이 이미 덤프된 teacher 로짓만으로, 앙상블(= KD 상한)이 최고 단일 teacher를 넘는지 확인한다. gap이 없으면 student가 exp1을 못 넘으므로 착수하지 않는다.

- **SSOT**: `output/kd_gate_ensemble.json`. **평가 셋**: 정리 test(11,244, [ADR-0010](../adr/0010-data-cleaning.md)) — 현 비교선(exp2 정리 0.8599)과 같은 축. 구 로짓(구 split 11,271)을 정리 문서로 슬라이스·정렬하고 정리 라벨로 재계산한다.
- **정렬 가드**: 단일 teacher 정리 test micro 재계산이 정리 SSOT와 일치(exp1 0.8683 · KoBERT 0.8500, `output/headline_cleaned_test.json`).
- **앙상블**: teacher별 `sigmoid` 후 확률 공간에서 가중 평균한다 — 이종 손실(ASL의 shift된 로짓 등)이 로짓 스케일을 다르게 만들어 로짓 평균은 스케일에 민감하기 때문이다. 가중치는 **val에서 선택 후 test 적용**(누수 차단), 임계 τ=0.5 고정.

### 게이트 실측 (정리 test 11,244)

| 모델 | micro | k=1 | k≥2 | empty | B0 | B1 | B2 | B3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single exp1 (8192, focal) | 0.8683 | 0.8915 | 0.8048 | 1.35% | 0.8762 | 0.8733 | 0.8514 | 0.8490 |
| single ASL (512) | 0.8359 | 0.8476 | 0.8044 | 0.43% | 0.8527 | 0.8410 | 0.8123 | 0.7944 |
| single KoBERT (512) | 0.8500 | 0.8715 | 0.7912 | 1.17% | 0.8682 | 0.8529 | 0.8258 | 0.8212 |
| **앙상블 best** (prob 0.5/0.2/0.3) | **0.8756** | 0.8964 | 0.8190 | 1.37% | 0.8842 | 0.8805 | 0.8572 | 0.8584 |
| 앙상블 등가중 | 0.8746 | 0.8929 | 0.8252 | 1.23% | 0.8843 | 0.8810 | 0.8521 | 0.8546 |
| 앙상블 로짓평균 | 0.8748 | 0.8927 | 0.8267 | 1.12% | 0.8842 | 0.8806 | 0.8533 | 0.8580 |

- **헤드룸: 전역 micro +0.73pt · k≥2 micro +1.42pt**(앙상블 best − 최고 단일). best 가중치는 val micro 최적(exp1 0.5 / ASL 0.2 / KoBERT 0.3).
- **이득은 다양성에서 온다.** ASL·KoBERT가 exp1보다 한참 낮은데도 앙상블이 +0.73pt를 얻는다 — 강한 teacher가 아니라 오류 탈상관이 동력이다. anchor oracle-any(셋 중 하나라도 top-1 적중) 0.9520으로 최고 단일 0.9050 대비 +4.70pt, 쌍별 top-1 일치율 0.860~0.877.
- **표적 슬라이스를 회수한다.** k≥2 micro가 앙상블에서 0.8190~0.8267로 최고 단일(exp1 0.8048)을 +1.4~1.9pt 넘는다. ASL의 recall 성분과 exp1의 랭킹이 상쇄가 아니라 보완한다.
- **이득이 전 length-bin에 분포한다(+0.58~+0.94pt).** B3(>2048)뿐 아니라 **B0–B2(2048 창이 완전히 보는 66.8%+)에도** 이득이 있어 2048 student가 대부분을 전이받을 수 있다 — student 길이 결정의 직접 근거.
- **soft target 배합이 operating point 레버다.** best(exp1 상향)는 전역 micro 최고, 등가중·로짓평균은 k≥2 최고 — 증류 타깃을 어떻게 짜느냐로 student가 k=1↔k≥2 트레이드 위 어디에 앉을지 조절된다.

## teacher — 확정 (게이트로 근거화)

세 teacher는 각기 다른 축으로 앙상블에 기여한다. 로짓은 이미 덤프돼 있다(정리 문서로 슬라이스해 사용).

| teacher | tag | 역할 | 기여 근거 |
| --- | --- | --- | --- |
| **exp1** | `modernbert-patent-len8192` | 최고 성능·랭킹 골격, 길이 다양성(8192) | 정리 micro 0.8683, 전 지표 최고 |
| **ASL** | `modernbert-patent-len512-asl` | k≥2 recall 성분(과소예측 억제) | k≥2 micro 0.8044·FN 최저, operating point 다양성 |
| **KoBERT** | `kobert-patent-baseline_len512` | 아키텍처 다양성(유일 이종 BERT) | 쌍별 top-1 일치율 최저(0.860~0.871), 탈상관 주도 |

## soft target — 구성

- **앙상블 확률**: `q_c = Σ_k w_k · sigmoid(z_{k,c})`, 가중치 `w = (exp1 0.5, ASL 0.2, KoBERT 0.3)`(게이트 val 선택). teacher별로 자기 시야(exp1 8192, ASL·KoBERT 512)를 담은 확률이라, student(2048)가 8192 teacher의 신호를 부분 상속하는 **context distillation**이 성립한다.
- **주 타깃 = val-micro 최적 가중 앙상블**(전역 micro 최대화, 헤드라인 정합). 가중치는 val에서 고정해 train·test에 적용만 한다.
- **대안 타깃 = 로짓평균**(k≥2 최고). 주 타깃이 k≥2에서 부진하면 표적 회수용으로 전환한다(배합 축 실측으로 가른다).
- ⚠️ **확률 공간에서 앙상블한다**(로짓 아님). 이종 손실의 로짓 스케일 차이 때문이며, 게이트가 이 방식으로 검증됐다.

## KD 손실 — 멀티라벨(sigmoid) 형식

softmax KD가 아니라 **188 독립 sigmoid의 라벨별 이진 KD**다.

- **혼합 손실**: `L = (1−λ)·L_hard + λ·L_distill`.
  - `L_hard` = focal(α=0.25, γ=2), 정답 멀티핫 `y` — 프로젝트 확정 손실.
  - `L_distill` = student sigmoid `p_c`와 soft target `q_c`의 라벨별 BCE(요소 평균, `patent_train.losses` reduction 규약 정합).
- **λ(혼합)**: 주 런 **λ=0.5**. 예산이 열리면 val에서 λ∈{0.3, 0.5, 0.7, 1.0} 스윕(1.0 = 순수 증류, 하드 라벨 제거 — 위험하나 정보량 확인용).
- **온도 T**: 주 런 **T=1**(원 앙상블 확률을 그대로 타깃). 확률 공간 앙상블이라 온도는 부차 knob으로 두고, 주 런이 부진할 때만 ablation.
- **구현 경계**: focal 대체 지점(`FocalTrainer.compute_loss`)만 혼합 손실로 교체하고 나머지 경로는 공통 프로토콜 고정 — 손실 축 실험과 동일한 통제.

## student — 설정

- **모델**: `skt/A.X-Encoder-base`에서 초기화(exp1·exp2와 동일 백본).
- **max_len = 2048.** 게이트에서 앙상블 이득이 B0–B2에 걸쳐 있어 2048 창으로 대부분 전이 가능하고, ≥2048 문서는 4.9%(B3)뿐이다. 8192 대비 추론 비용을 크게 낮추면서 이득을 회수하는 지점이다. **narrative-safe 대안 = student 8192**(exp1 위 순수 가산, 압축 서사 없음).
- **레시피**: 확정 레시피(`08_01`) eff_batch 128 · lr 4.8e-4(linear scaling) · 12 epoch · linear · warmup_ratio 0.1 · `early_stop_epochs=2`. 정리 데이터(train 201,616 / val 11,132 / test 11,244).
- **공통 프로토콜 고정**([modernbert.md](modernbert.md) 「공통 프로토콜」): dtype fp32 마스터 + bf16 autocast, FA2, `group_by_length`, `classifier_pooling="mean"`, eos 마감 절단, 평가도 훈련과 동일 `max_len` 절단, 추론 batch 8.
- ⚠️ **레시피는 512에서 튜닝돼 2048에 적용된다.** lr의 배치 스케일링은 길이와 독립이라 그대로 쓰되, 2048 최적 이탈 가능성은 미검증분으로 둔다.

## 선행 작업 — teacher soft target을 train에 덤프 (주 비용선)

KD 훈련은 **모든 훈련 문서(정리 train 201,616)**에 대한 teacher 확률이 필요하다 — val/test 덤프만으로는 부족하다.

1. **teacher 로짓을 정리 train에 덤프**: exp1@8192 · ASL@512 · KoBERT@512를 정리 train에 대해 추론. **exp1@8192 × 201,616 문서가 이 실험의 주 비용**(GPU 수 시간)이며 ASL·KoBERT@512는 저렴하다. 추론 진입점은 `src/patent_train`(`TrainConfig.for_inference` + `build_model(checkpoint=)`), 순차 샘플러로 행 순서를 `document_id`에 고정(`group_by_length` 순열 함정 회피 — `NEXT_SESSION.md` 함정, `runner.predict_logits`가 assert).
2. **soft target 조립**: `q = Σ w_k·sigmoid(z_k)`를 (201,616, 188)로 만들고 정리 train `document_id` 순서로 저장(fp16). 이 배열이 라벨과 함께 KD 훈련 입력이 된다.
3. **정렬 verify**: 각 teacher train 로짓 행 순서 == train `document_id`, 각 teacher train micro가 val/test 수준과 정합.

## 판정 프로토콜

- **1런 = 1설정.** 확정 레시피 위에서 손실만 혼합 KD로 교체. 나머지 변수는 공통 프로토콜 고정.
- **필수 2런**:
  - **student 2048 focal(무-KD) — 통제군.** 2048이 8192·512 대비 길이만으로 주는 값을 분리한다. KD 이득 귀속의 기준선이다.
  - **student 2048 KD(주 타깃, λ=0.5) — 처리군.**
- **판정 기준**:
  1. **주 지표**: student(2048 KD) 정리 test micro vs exp1(8192) 0.8683. 넘거나 동률인가.
  2. **회수율**: `(student − exp1) / (앙상블 − exp1)` — +0.73pt 상한 대비 회수 비율.
  3. **표적 슬라이스**: k≥2 micro vs exp1 0.8048 — 손실이 못 옮긴 카디널리티를 상속했는가.
  4. **KD 귀속**: student(2048 KD) vs 통제군(2048 focal). KD 순효과. 통제군 vs exp1(8192)은 길이 성분(2048 vs 8192).
  5. **효율**: 2048 추론 비용 vs 8192 — student가 exp1을 동률 이상으로 따라잡으면 압축 이득(같은 성능·낮은 비용).
- **진단 병기**(손실 프로토콜과 동일 배터리): k=1/k≥2 평균 예측 라벨 수·과소예측률(기준 exp1 k≥2 정답 2.35)·empty rate·val 최적 τ·length-bin micro(B0–B2 전이 확인).
- **결정 규칙**:
  - **채택(2048 KD 배포)**: student(2048 KD) ≥ exp1(8192) micro + 추론 비용 우위.
  - **부분**: student > 통제군이나 < exp1(8192) → KD는 이득이나 8192 미달. 2048-KD vs 8192-single을 비용/성능으로 결정.
  - **음성**: student ≈ 통제군 → KD 무효, 축 종결. exp1(8192) 또는 2048-focal을 비용으로 선택.

## verify

- soft target 가중치 == 게이트 선택치 · `q`가 teacher 로짓에서 재현.
- 혼합 손실이 `compute_loss`에서만 갈리고 나머지 경로는 `11_01`과 동일.
- student 평가 τ=0.5 재계산이 훈련 중 지표와 4자리 일치 · 로짓 행 순서 == `document_id`.
- 정리 test/val(11,244/11,132) 일관 사용 — 구 split(11,271/11,162)과 혼용 금지.

## 스코프 밖 / 미채택

- **feature 기반(hidden-state) 증류 — 미채택.** 이종 토크나이저로 토큰 정렬이 불가하다. 응답 기반(로짓/확률) 증류만 쓴다.
- **앙상블 배포 — 닫힘([ADR-0005](../adr/0005-no-ensemble.md)).** KD는 앙상블을 훈련 시점 teacher로만 쓴다.
- **teacher 재훈련 — 없음.** 기존 exp1·ASL·KoBERT 체크포인트를 그대로 쓴다(로짓 재사용).
- **λ/T/가중치 스윕 — 선택.** 예산이 열릴 때 val에서만. 주 판정은 필수 2런으로 낸다.

## 실측

(주 런 완료 후 반입 — SSOT `output/*.json`, 판정은 위 「결정 규칙」)
