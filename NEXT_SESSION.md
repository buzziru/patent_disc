# 다음 세션 인계

> 읽는 순서: `PROJECT.md`(SSOT) → 이 문서. 수치 SSOT는 `output/*.json`, 손실 축은 `docs/experiments/loss-function.md`, 결정 경위는 `docs/adr/`.

## 지금 상태 — 도메인 축 종결, 다음 착수는 계층 손실

`13_02`(TAPT 백본 분류 파인튜닝)가 완주해 도메인 사전학습 축이 **기준 미달로 종결**됐다([ADR-0013](docs/adr/0013-domain-pretraining-closure.md), 실측 SSOT [domain-pretraining.md](docs/experiments/domain-pretraining.md)). 활성 축은 **계층 손실**([hierarchy-loss.md](docs/experiments/hierarchy-loss.md), 표적 질량 실측 완료 → 훈련 1런)이고 KD는 그 뒤다([knowledge-distillation.md](docs/experiments/knowledge-distillation.md), 게이트 GREEN이나 GPU 단계가 여럿).

- **`13_02` 결과**: 정리 test micro **0.8572**(macro 0.8545 · sample 0.8724 · anchor weighted 0.8198)로 앵커 `11_01`(0.8588) 대비 **−0.15pt**. 판정선 +0.4pt 미달이고 네 지표 부호가 모두 음수이나 폭은 표본 잡음(sd 0.18~0.21pt) 안이다 — "해롭다"가 아니라 **효과 없음**이다.
- **곡선에도 이득 없음**: eval 24지점 짝지은 델타 평균 −0.03pt · sd 0.43pt, 3 epoch까지 평균 +0.04pt. 초기 우위로 보이던 ep1.0(+0.69)·ep1.5(+0.67)는 ep2.0(−1.29)이 되돌린다. TAPT의 통상 이득인 **초기 수렴 가속조차 없다**(첫 eval 0.6385 대 0.6380).
- **기제 = 코퍼스 동일성**: 같은 train split 201,616 문서에서 TAPT 5 epoch가 802M 토큰인 반면 분류 파인튜닝 12 epoch는 **1,116M 토큰**(len512 절단, 문서당 평균 461)이다. 파인튜닝이 같은 텍스트를 1.4배 더, 라벨 감독과 함께 본다 — MLM이 넣을 정보가 없다.
- **⚠️ `13_02` 로짓은 행 순열로 깨져 삭제했다.** 훈련 경로 Trainer로 `predict_logits`를 불러 「함정」 1번이 재발했다(`11_01`에 이은 두 번째). 런이 낸 지표는 유효하나 paired bootstrap CI·길이 bin slope는 미집행으로 남는다. 모델은 Hub(`ingyoun/A.X-patent-len512-tapt`)에 있으니 필요하면 `11_03`식 추론 전용 경로로 재덤프한다.

## 완료된 축

| 축 | 결론 | SSOT |
| --- | --- | --- |
| 손실 함수 | **종결** — ZLPR 0.8493 · ASL 0.8362 · BCE 0.8538이 모두 focal exp2 0.8601 미달. 기제는 FP:FN 부호 뒤집힘(k=1 과대예측 vs k≥2 과소예측). DL2 미착수 확정 | [ADR-0009](docs/adr/0009-loss-axis-closure.md) · [loss-function.md](docs/experiments/loss-function.md) |
| 카디널리티 디코딩 | **음성** — raw 확률 기대-F1 plug-in이 k≥2를 +0.90pt 회수하나 k=1 과대예측과 분리 불가로 전역 −0.22pt. 오라클-k +1.60pt는 도달 불가 상한 | [cardinality-decoding.md](docs/experiments/cardinality-decoding.md)(`10_01`) |
| 장문 열화 | **디프리오리티** — 최장 문서도 정답 top-5 ~98% 잔존이라 표현 붕괴가 아니다. label-aware attention 풀링 헤드룸 <~0.5pt | [longdoc-degradation.md](docs/experiments/longdoc-degradation.md)(`10_02`) |
| 계층 확장(`Lno` 게이트) | **하드 top-1 게이트 불채택** — 단일 `Lno` 게이트의 micro recall 상한 0.8590이 exp1 현재 recall 0.8697 아래(4모델 동일). 기존 2단계 추정(`Lno` 정확도 × 전체 오라클)은 1단계 실패를 이중 계상한 결함 추정량이고, 조건부로 교정하면 flat과 **항등**이라 판정력이 0이었다. 라벨 형상도 하드 게이팅을 배제한다(두 단계 모두 다중레이블이어야 하고, 그러면 표현력이 flat과 같다). **게이트 없는 형태·훈련된 조건부 2단계는 미측정**(2단계 상한 +3.48pt = sibling 질량, 조건부 현재 0.9630)이나, 주 지표 결손의 다수는 형제가 아니라 cross-`Lno` 두 번째 라벨이다(k≥2 FN 1,064 중 cross 589·형제 475) | [ADR-0002](docs/adr/0002-flat-multilabel-hierarchy.md) · [modernbert-comparison.md](docs/experiments/modernbert-comparison.md)「오류 구조」 · [data.md](docs/data/data.md)「계층 형상」(`scripts/hierarchy_conditional.py`·`scripts/multilabel_shape.py`) |
| 도메인 사전학습 | **종결** — 자체 코퍼스 TAPT가 정리 test micro 0.8572로 앵커 0.8588 대비 −0.15pt(판정선 +0.4pt 미달). 기제는 코퍼스 동일성(TAPT 802M 대 파인튜닝 1,116M 토큰, 같은 문서). MLM 체크포인트 교체·기성 `KorPatElectra` 모두 불채택 | [ADR-0013](docs/adr/0013-domain-pretraining-closure.md) · [domain-pretraining.md](docs/experiments/domain-pretraining.md)(`13_01`·`13_02`) |
| 데이터 클리닝 | **완료·Hub 반영 완료** — 입력 동일 336문서 제거. train 201,616 / val 11,132 / test 11,244, 잔여 충돌 0 | [ADR-0010](docs/adr/0010-data-cleaning.md)(`10_03`) |
| 코드 구조 | **src 전환 완료** — `src/patent_train`(config·backbones·data·model·losses·metrics·trainer·runner·probe). 노트북은 `TrainConfig` 하나로 실행 | [ADR-0011](docs/adr/0011-resource-constrained-methodology.md) · [runpod-jobs.md](docs/infra/runpod-jobs.md) |

## 성능 (test micro-F1)

| 런 | max_len | 손실 | 레시피 | micro-F1 | 비고 |
| --- | --- | --- | --- | --- | --- |
| KoBERT 재현 | 512 | focal | eff8/lr3e-5 | 0.8502 | 비교 기준점 |
| exp2 (A.X) | 512 | focal | eff8/lr3e-5 | 0.8601 | 512 계열 최고·손실 A/B 기준선 |
| **exp1 (A.X)** | **8192** | focal | eff8/lr3e-5 | **0.8685** | **최고 full run** |
| ZLPR (A.X) | 512 | ZLPR | eff128/lr4.8e-4 | 0.8493 | 미채택 |
| ASL (A.X) | 512 | ASL | eff128/lr4.8e-4 | 0.8362 | 미채택 |
| BCE (A.X) | 512 | BCE | eff128/lr4.8e-4 | 0.8538 | 진단(γ의 순수 값 −0.62pt) |
| `11_01` (A.X) | 512 | focal | eff128/lr4.8e-4 | 0.8588 | 정리 데이터·신 레시피 첫 focal 풀런 — **현행 앵커** |
| `13_02` (A.X TAPT) | 512 | focal | eff128/lr4.8e-4 | 0.8572 | 도메인 축 종결(앵커 −0.15pt, 잡음 내) |

- exp1~11_01은 **구 test(11,271)** 기준이다(`11_01`만 정리 test 11,244). 정리 test 재계산값은 exp1 0.8683 · exp2 0.8599 · KoBERT 0.8500이며 서열·격차는 불변이다(`output/headline_cleaned_test.json`).

## KD 축 (신규) — 게이트 GREEN, student 대기

손실 축이 못 넘은 **k≥2 카디널리티**를 이종 앙상블 증류로 겨냥한다. 앙상블은 훈련 시점 teacher로만 쓰고 배포는 단일 student — 앙상블 배포([ADR-0005](docs/adr/0005-no-ensemble.md))의 재제안이 아니다. 계획·프로토콜 [knowledge-distillation.md](docs/experiments/knowledge-distillation.md).

- **게이트(무훈련) GREEN**: teacher(exp1/ASL/KoBERT) 로짓 앙상블이 정리 test(11,244)에서 최고 단일 대비 **micro +0.73pt · k≥2 +1.42pt**. 이득이 다양성(오류 탈상관, oracle-any top1 +4.70pt)에서 오고 **전 length-bin에 분포**(B0–B2 포함 → 2048 student 전이 가능). SSOT `output/kd_gate_ensemble.json`.
- **확정 설계**: teacher 3종 고정 · soft target = **확률공간** 가중 앙상블(exp1 0.5/ASL 0.2/KoBERT 0.3, val 선택) · 손실 `(1−λ)·focal + λ·BCE(p,q)` λ=0.5 · **student len2048**(전이 가능·8192 대비 저비용) · 필수 2런(2048 focal 통제 + 2048 KD).
- **다음 착수 = teacher soft target을 정리 train(201,616)에 덤프.** exp1@8192 추론이 주 비용(GPU 수 시간), ASL·KoBERT@512 저렴. 순차 샘플러로 행 순서를 `document_id`에 고정(순열 함정).

## 도메인 사전학습 축 — 종결([ADR-0013](docs/adr/0013-domain-pretraining-closure.md))

"장문 열화가 본질적 난이도라면 특허 코퍼스로 사전학습된 표현은 다른가"를 **자체 코퍼스 TAPT** 하나로 검증했고, 기각됐다. 실측·프로토콜·아티팩트 검증 절차는 [domain-pretraining.md](docs/experiments/domain-pretraining.md)가 SSOT.

- **집행된 MLM**(`13_01`): 자체 **train split만** MLM(라벨 무시, **test는 제외** — 누수 방지, val은 loss 관측 전용). train 총 160.4M 토큰을 `max_len=2048` **비겹침 청킹**으로 전량 노출(단순 절단이면 총 토큰의 5.87% 유실) → 201,616문서 = 211,159청크. 마스킹 0.30 · lr 5e-5 · eff_batch 128 · **5 epoch = 약 802M 토큰**. 산출 `ingyoun/A.X-patent-tapt-mlm@62818c2`(백본 키 `axenc_tapt`). val loss 0.4272 → 0.3742로 단조 하강, 과적합 반전 없음.
- **분류 결과**(`13_02`): 정리 test micro **0.8572** = 앵커 −0.15pt. **MLM loss가 이 축의 대리 지표가 아님**이 확인됐다 — MLM은 내내 좋아졌는데 하류 이득이 0이다.
- **재개하지 않는 이유**: (1) **MLM 체크포인트 교체** — 용량-반응 곡선의 양 끝(TAPT 0 epoch = 앵커 0.8588, 5 epoch = 0.8572)이 0.15pt 간격이라 중간 체크포인트가 판정을 뒤집으려면 곡선이 양 끝보다 0.4pt 위로 볼록해야 한다. MLM val loss 단조 하강이라 "중간이 최적" 신호가 없고, 그럴 유일한 기제인 파국적 망각도 관측되지 않았다(ep5가 잡음 내). 비용은 체크포인트당 8h 14m. (2) **코퍼스 동일성은 용량과 무관한 이유**다 — 없는 정보는 노출 시간으로 생기지 않는다. (3) **full DAPT**(외부 특허 코퍼스 수십억 토큰)는 개인 프로젝트 예산 밖이다.
- **기성 특허 백본 — 검토 후 불채택**: `KIPI-ai/KorPatElectra`가 HF에 공개돼 있다(특허 4.6M건·0.5B 문장 사전학습). `monologg/koelectra-base-v3-discriminator`와 짝지으면 `config.json`이 바이트 단위로 같아(12L/768/12H, vocab 35,000) 이례적으로 깨끗한 도메인 대조가 된다. 그럼에도 **레버가 아니다** — `max_position_embeddings=512` 하드캡이라 헤드라인의 +0.84pt 길이 성분에 접근 불가(운영 모델 불가), **gated + 비상업 한정 라이선스**로 apache-2.0 기준의 주 모델 선정([ADR-0003](docs/adr/0003-long-document-encoder.md))과 산출물 성격이 충돌, `vocab.txt`가 서로 달라 재는 것은 코퍼스 단독이 아니라 **코퍼스+어휘 묶음**이다. 풀 파인튜닝 2런(~16h GPU)과 토큰화 데이터셋 2종 신규 생성을 요구하는데 얻는 것은 성능이 아니라 설명이다. 증거가 필요해지면 `12_02`식 **동결 프로브**(hidden-state 덤프 + 선형 프로브, 추론만)로 부호와 크기를 먼저 싸게 잰다.
- **이전에 폐기한 대안**: 토크나이저 갈래(형태소+특허 vocab 재현·KoELECTRA 이식·복합명사 vocab 추가) — **세그멘테이션 이득이 2048 창에서 이미 소멸**(절단은 >2048 꼬리 ~4.9%뿐), 표현 이득은 사전학습 없이는 실현 불가. 임베딩-only 계속학습 — 신규 vocab이 없으면 명분 소멸.
- **비용 기준선**(다른 축에 재사용): MLM은 Colab L4 @2048 ≈ 15,900 tok/s(epoch ≈2.8h, 5 epoch ≈14h). 분류는 A40 @512 ≈ 41,500 tok/s(풀런 8h20m 역산) — A100 PCIe ~2배 속도지만 시간당 3.16배라 A40이 ~37% 저렴.

## 남은 일

1. **`11_01` 로짓은 행 축이 정상이다 — 재덤프 불필요.** `output/logits_modernbert-patent-len512-b128_{val,test}.npy`를 정리 데이터셋 행 순서의 라벨과 대면시키면 test micro/macro/sample이 SSOT(0.858759 / 0.856503 / 0.873791)와 1e-6까지 일치하고 val도 0.8623으로 정상이다(순열이면 ~0.006이 나온다). 정리 로짓의 행 축 SSOT는 `output/doc_ids_clean_{val,test}.json`이며 데이터셋 `document_id` 순서와 일치한다(구 `doc_ids_*`는 구 로짓 재현용 유지, `docs/data/data.md`「주의」). **비교선은 정리 test 재계산값(exp2 0.8599)**이다 — `11_01`은 정리 test(11,244)에서 평가되므로 구 test 수치(0.8601)와 직접 대면 안 된다.
2. **계층 손실(MCLoss) 1런이 다음 착수다** — 표적 질량 실측이 끝나 있고([hierarchy-loss.md](docs/experiments/hierarchy-loss.md)) 추론 구조를 바꾸지 않으므로 앵커 `11_01`에서 손실만 바꾼 단일 변수 대조가 된다. 판정선·잡음 기준은 도메인 축과 같다(+0.4pt, 표본 잡음 sd 0.18~0.21pt). **로짓 덤프는 훈련 경로 Trainer가 아니라 `11_03`식 추론 전용 경로로 한다**(「함정」 1 — `13_02`에서 재발했다).
3. **`11_01`에는 판정할 두 축이 겹쳐 있다** — 데이터 클리닝과 신 레시피가 동시에 바뀐 런이다. 클리닝 효과는 볼륨(train 0.04%)이 작아 aggregate에서 분리되지 않으므로([ADR-0010](docs/adr/0010-data-cleaning.md)), 연루 클래스(특히 EB01) per-class F1을 paired로 대조한다.
4. **RoBERTa·KoBERT 토큰화본 336 필터 미반영** — `ingyoun/patent-clean-text-roberta-tokenized`·`...-kobert-tokenized`는 정리 이전 상태다. 재토큰화 없이 같은 `document_id`로 필터링하면 되고, KLUE-RoBERTa 대조군이나 KoBERT 재현을 다시 돌릴 때 선행한다.
5. **잡음 하한 — 표본 성분은 실측 완료, 훈련 성분만 남았다**([eval-noise.md](docs/experiments/eval-noise.md)). 고정 test paired bootstrap으로 micro 델타의 표본 잡음 sd가 0.18~0.21pt임이 확정됐고, 길이(+0.84pt)·모델(+0.99pt)·손실 3종 열세는 모두 이를 넘으며 `11_01`−exp2(−0.12pt)는 잡음 내다. 남은 것은 **시드(훈련) 잡음**이며, 한 런만 집행한다면 대상은 `11_01` 재현(현행 레시피·손실 축 기준선, A40 ~8.3h)이다 — 여유가 +0.13pt뿐인 focal−BCE가 이 성분에 직접 걸린다. KLUE-RoBERTa 대조군(주장 방어용, 선택)은 그다음.

## 함정 (놓치기 쉬움)

- **`train_sampling_strategy="group_by_length"`는 eval·predict 로더에도 적용된다.** `Trainer._get_eval_sampler`가 같은 설정을 보고 `LengthGroupedSampler`를 반환하므로 `trainer.predict`의 **반환 행이 길이 그룹 순열**로 나온다. 평가 지표는 라벨을 같은 순서로 모으니 멀쩡하지만(`11_01` test micro 0.8588은 유효), 덤프된 로짓은 데이터셋 행 순서를 전제하는 하류 분석과 전부 어긋난다. 순열은 `torch.randperm(generator=None)` 기반이라 **사후 복원이 불가능**하다 — 재덤프뿐이다. `runner.predict_logits`의 방어(덤프 동안 순차 샘플러로 복원 + 반환 라벨로 행 순서 assert)는 **믿을 수 없다** — `13_02`에서 assert를 통과하고도 val·test 로짓이 모두 순열로 나왔다(`11_01`에 이은 두 번째 재발, 두 파일은 삭제). **로짓은 훈련이 끝난 뒤 추론 전용 경로(`TrainConfig.for_inference` → `_build_inference_trainer`)에서 별도 덤프하고, 같은 노트북에서 점추정을 SSOT와 대조해 그 자리에서 검증한다**(`11_03` 마지막 셀). 하류에서도 `scripts/eval_noise_bootstrap.py`가 모델마다 자동 대조한다. 순열 로짓의 징후는 micro ~0.006·top-1 ~0.006(우연 수준)인데 문서당 예측 수는 정상(1.22개)이라는 조합이다.
- **스케줄 길이가 다른 런을 에폭 눈금으로 비교하지 않는다.** `linear`+`warmup_ratio=0.1`은 총 스텝에 비례한다 — 2 epoch 탐색 런은 0.2 epoch에 피크 lr을 지나 곧바로 어닐링에 들어가고, 12 epoch 풀런은 1.2 epoch까지 워밍업 중이다. 같은 "1 epoch"이 lr 궤적에서 전혀 다른 자리이며, 조기 곡선 대조는 **12 epoch 런끼리** 한다.
- **`prep_cache`는 `{backbone}_len{max_len}`으로만 키잉된다.** 데이터셋 버전이 바뀌어도 볼륨에 캐시가 남아 있으면 원본 다운로드를 건너뛰어 **구 데이터로 학습된다.** 데이터셋을 갱신했으면 `/workspace/prep_cache/*`를 지우고 시작하고, `[schedule]` 출력의 step/epoch로 행 수를 역산해 확인한다(정리 데이터 = 1,576 step/epoch @ eff128).
- **정리 test(11,244)와 구 test(11,271)는 다른 셋이다.** 서로 다른 test에서 잰 micro를 나란히 놓지 않는다.
- **현행 대조 손실은 BCE가 아니라 `FocalLoss(alpha=0.25, gamma=2)`다.** `alpha`가 손실 전체에 곱해지는 상수라 클래스 균형 역할을 하지 않아 비대칭 처리가 사실상 γ 하나뿐이다(문서당 음성:양성 ≈156:1).
- **exp1은 12에폭을 전부 소진**하고 early stopping이 한 번도 발동하지 않았다(수렴이 아니라 예산 종료).
- **`classifier_pooling`은 이미 `mean`**(A.X-Encoder-base 기본값). CLS 병목 우려는 이 모델에 해당하지 않는다.
- **Hub에 push된 훈련 repo는 커밋 메시지로 완주를 판단할 수 없다.** `push_to_hub=True`+`hub_strategy="all_checkpoints"`는 저장 시점마다 "Training in progress…"로 커밋하고, 런이 정상 종료해도 "End of training" 커밋이 없을 수 있다. 루트 `model.safetensors`의 sha256을 마지막 `checkpoint-*/model.safetensors`와 대조하고 그 `trainer_state.json`의 `global_step`·`epoch`으로 확인한다(`ingyoun/A.X-patent-tapt-mlm@62818c2` 사례 — [domain-pretraining.md](docs/experiments/domain-pretraining.md)).

## 확정된 사실

- **주 비교 지표는 멀티라벨 micro-F1.** top-1 weighted-F1과 P@1/3/5는 벤더 baseline의 레거시이며 병기용(anchor).
- **임계값은 레버가 아니다.** global τ 오라클 micro 헤드룸 +0.0000~+0.0008. 문서 상대 임계값(`p ≥ α·p_max`)은 sample-F1을 +0.66~0.85pt 올리나 micro는 −0.23~−0.52pt로 떨어뜨린다.
- **카디널리티 헤드룸은 회수 불가로 종결.** 손실·추론 디코딩 두 경로 모두 부정이며, 분리에는 문서별 k가 필요하나 닫힌 갈래다.
- **앙상블은 채택하지 않는다.** 단일 모델 운영 요구. 3모델 로짓평균 앵커 weighted-F1 +0.71pt이나 추론 비용 3배.
- **추가 헤드룸의 레버는 모델·레시피다** — 측정된 최대 단일 이득은 길이(exp1 len8192, exp2 대비 +0.84pt).
- 나머지 닫힌 갈래는 `PROJECT.md` 「닫힌 갈래」 표를 따른다. **새 근거 없이 다시 제안하지 않는다.**

## 열린 질문

- **주 모델을 8192·2048·512 중 무엇으로 갈지** — 운영 단일 모델의 추론 비용 대 성능. KD 축이 열려 **2048 KD student**가 후보로 추가됐다(게이트: 앙상블 이득이 전 bin 분포 → 2048로 전이 가능·저비용). 8192는 +0.84pt 길이 이득이나 신 레시피 재훈련이 걸린다(8192 실측은 구 레시피뿐). student 길이는 [knowledge-distillation.md](docs/experiments/knowledge-distillation.md) 판정으로 결정한다.
- **훈련 길이·스케줄 형태**(linear 유지 여부) — exp1이 12에폭을 소진한 사실과 함께 검토한다.
- **KLUE-RoBERTa-base 대조군**(선택) — 성능이 아니라 주장 방어(크기 confound 제거)가 목적. 예산이 남을 때만.

## 작업 규약

- 노트북 작성은 **지시가 있을 때만** 한다. 훈련 노트북은 `TrainConfig` 주입만 하고 로직은 `src/patent_train`에 둔다.
- 검증은 산출물에서 재현되게 한다 — 판정 기준은 서술 통계가 아니라 **결정 질문에 직접 답하는 양**(오라클·회수율·학습 곡선)으로 세우고, `### verify` 셀에 assert와 SSOT 대조를 남긴다.
