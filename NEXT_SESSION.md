# 다음 세션 인계

> 읽는 순서: `PROJECT.md`(SSOT) → 이 문서. 수치 SSOT는 `output/*.json`, 손실 축은 `docs/experiments/loss-function.md`, 결정 경위는 `docs/adr/`.

## 지금 상태 — `11_01` 정리 데이터 풀런 훈련 중

손실 축이 종결됐고([ADR-0009](docs/adr/0009-loss-axis-closure.md)), 데이터 클리닝([ADR-0010](docs/adr/0010-data-cleaning.md))과 훈련 코드 src 전환([ADR-0011](docs/adr/0011-resource-constrained-methodology.md))이 끝난 뒤의 첫 훈련이다.

- **런**: `notebook/11_01_CleanData_Recipe.ipynb` — 정리 데이터 + 확정 레시피 focal 풀런(RunPod A40). **신 레시피(eff128/lr4.8e-4)의 첫 focal 풀런이자 정리 데이터의 첫 훈련**이다. 09 계열 실측 기준 소요 ~8.3h.
- **설정**: `backbone="axenc"` · focal(α=0.25, γ=2) · len512 · eff_batch 128 · lr 4.8e-4 · linear · warmup_ratio 0.1 · 12 epoch · `early_stop_epochs=2` · eval·save 에폭당 2회(788 step). 산출은 `ingyoun/A.X-patent-len512-b128`(Hub) · `modernbert-patent-len512-b128_metrics.json` · val/test 로짓 덤프.
- **진행 상황(1 epoch 시점 val)**: micro 0.7195 · macro 0.6890 · sample 0.6754 · empty 19.0% · anchor 0.7249 · val loss 0.000664. **exp2(`05_01`) 12에폭 곡선과 정합**이다(0.5 epoch 0.6380 대 0.6396 / 1.0 epoch 0.7195 대 0.7158, val loss는 더 낮음).
- **이후 대조점**([training-curves.md](docs/experiments/training-curves.md) exp2 곡선): 2 epoch에 micro ≈0.796 · empty ≤0.09, 3~5 epoch에 val loss 최저 통과(이후 상승은 정상), F1은 11.5 epoch까지 우상향.

## 완료된 축

| 축 | 결론 | SSOT |
| --- | --- | --- |
| 손실 함수 | **종결** — ZLPR 0.8493 · ASL 0.8362 · BCE 0.8538이 모두 focal exp2 0.8601 미달. 기제는 FP:FN 부호 뒤집힘(k=1 과대예측 vs k≥2 과소예측). DL2 미착수 확정 | [ADR-0009](docs/adr/0009-loss-axis-closure.md) · [loss-function.md](docs/experiments/loss-function.md) |
| 카디널리티 디코딩 | **음성** — raw 확률 기대-F1 plug-in이 k≥2를 +0.90pt 회수하나 k=1 과대예측과 분리 불가로 전역 −0.22pt. 오라클-k +1.60pt는 도달 불가 상한 | [cardinality-decoding.md](docs/experiments/cardinality-decoding.md)(`10_01`) |
| 장문 열화 | **디프리오리티** — 최장 문서도 정답 top-5 ~98% 잔존이라 표현 붕괴가 아니다. label-aware attention 풀링 헤드룸 <~0.5pt | [longdoc-degradation.md](docs/experiments/longdoc-degradation.md)(`10_02`) |
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
| `11_01` (A.X) | 512 | focal | eff128/lr4.8e-4 | 0.8588 | 정리 데이터·신 레시피 첫 focal 풀런 |

- exp1~11_01은 **구 test(11,271)** 기준이다(`11_01`만 정리 test 11,244). 정리 test 재계산값은 exp1 0.8683 · exp2 0.8599 · KoBERT 0.8500이며 서열·격차는 불변이다(`output/headline_cleaned_test.json`).

## KD 축 (신규) — 게이트 GREEN, student 대기

손실 축이 못 넘은 **k≥2 카디널리티**를 이종 앙상블 증류로 겨냥한다. 앙상블은 훈련 시점 teacher로만 쓰고 배포는 단일 student — 앙상블 배포([ADR-0005](docs/adr/0005-no-ensemble.md))의 재제안이 아니다. 계획·프로토콜 [knowledge-distillation.md](docs/experiments/knowledge-distillation.md).

- **게이트(무훈련) GREEN**: teacher(exp1/ASL/KoBERT) 로짓 앙상블이 정리 test(11,244)에서 최고 단일 대비 **micro +0.73pt · k≥2 +1.42pt**. 이득이 다양성(오류 탈상관, oracle-any top1 +4.70pt)에서 오고 **전 length-bin에 분포**(B0–B2 포함 → 2048 student 전이 가능). SSOT `output/kd_gate_ensemble.json`.
- **확정 설계**: teacher 3종 고정 · soft target = **확률공간** 가중 앙상블(exp1 0.5/ASL 0.2/KoBERT 0.3, val 선택) · 손실 `(1−λ)·focal + λ·BCE(p,q)` λ=0.5 · **student len2048**(전이 가능·8192 대비 저비용) · 필수 2런(2048 focal 통제 + 2048 KD).
- **다음 착수 = teacher soft target을 정리 train(201,616)에 덤프.** exp1@8192 추론이 주 비용(GPU 수 시간), ASL·KoBERT@512 저렴. 순차 샘플러로 행 순서를 `document_id`에 고정(순열 함정).

## 도메인 사전학습 축 (신규) — TAPT 채택, 사용자 선행 학습 후 진행 (보류)

"장문 열화가 본질적 난이도라면 특허 코퍼스로 사전학습된 표현은 다른가"를 검토한 결과, **자체 코퍼스 TAPT**를 단일 레버로 남겼다. 현재 사용자가 TAPT를 선행 학습한 뒤 진행하기로 해 **보류** 상태다.

- **성격**: 기본 표현 품질(백본) 축으로, 닫힌 장문·풀링·헤드 갈래의 재탕이 아니다. 단 동기는 "장문 열화 처방"이 아니라 "전 구간 level 상승"이다 — 3절 모델-vs-창 분해가 slope 개선을 배제한다(모델 교체는 slope을 못 편다, B3 모델 성분 +0.14pt). "본질적 난이도"는 *일반 도메인 백본에서* 조건부다.
- **폐기된 대안**: (1) 기성 KorPatBERT/KorPatELECTRA — 소속인증·사용협약으로 **이용 불가**, 게다가 512 컨텍스트. (2) 토크나이저 갈래(형태소+특허 vocab 재현·KoELECTRA 이식·복합명사 vocab 추가) — **세그멘테이션 이득이 2048 창에서 이미 소멸**(절단은 >2048 꼬리 ~4.9%뿐), 표현 이득은 사전학습 없이는 실현 불가. (3) 임베딩-only 계속학습 — 신규 vocab이 없으면 명분 소멸.
- **TAPT 설계(잠정)**: 자체 **train split만** MLM(라벨 무시, **test는 MLM에서 제외** — 누수 방지). 예산 ~0.34B 토큰/epoch × 3~6 epoch ≈ 1~2B → A40 ~7~14h · **~$3~6**. 시퀀스는 커리큘럼(1024~2048 위주 + 마지막 짧은 8192). 작은 코퍼스라 LR·epoch 보수적(MLM 과적합 함정).
- **판정 축**: TAPT→분류 파인튜닝 후 **exp1 대비 멀티라벨 micro·R-Precision**(길이 bin별로 slope 불변 재확인). KorPat 비교선이 닫혀 TAPT가 도메인 레버의 **첫 실측이자 후보 해법**을 겸한다. exp1을 유의 폭으로 못 이기면 축을 접는다.
- **비용 기준선**: A40 @512 ≈ 41,500 tok/s(분류 풀런 8h20m 역산). A100 PCIe ~2배지만 시간당 3.16배라 A40이 ~37% 저렴. full DAPT(3B 커리큘럼)는 A40 ~$12 — TAPT로 신호 확인 후에만 확장 검토.

## 남은 일

1. **`11_01` 로짓은 행 축이 정상이다 — 재덤프 불필요.** `output/logits_modernbert-patent-len512-b128_{val,test}.npy`를 정리 데이터셋 행 순서의 라벨과 대면시키면 test micro/macro/sample이 SSOT(0.858759 / 0.856503 / 0.873791)와 1e-6까지 일치하고 val도 0.8623으로 정상이다(순열이면 ~0.006이 나온다). 정리 로짓의 행 축 SSOT는 `output/doc_ids_clean_{val,test}.json`이며 데이터셋 `document_id` 순서와 일치한다(구 `doc_ids_*`는 구 로짓 재현용 유지, `docs/data/data.md`「주의」). **비교선은 정리 test 재계산값(exp2 0.8599)**이다 — `11_01`은 정리 test(11,244)에서 평가되므로 구 test 수치(0.8601)와 직접 대면 안 된다.
2. **판정할 두 축이 겹쳐 있다** — 이 런은 데이터 클리닝과 신 레시피가 동시에 바뀐다. 클리닝 효과는 볼륨(train 0.04%)이 작아 aggregate에서 분리되지 않으므로([ADR-0010](docs/adr/0010-data-cleaning.md)), 연루 클래스(특히 EB01) per-class F1을 paired로 대조한다.
3. **RoBERTa·KoBERT 토큰화본 336 필터 미반영** — `ingyoun/patent-clean-text-roberta-tokenized`·`...-kobert-tokenized`는 정리 이전 상태다. 재토큰화 없이 같은 `document_id`로 필터링하면 되고, KLUE-RoBERTa 대조군이나 KoBERT 재현을 다시 돌릴 때 선행한다.
4. **잡음 하한 — 표본 성분은 실측 완료, 훈련 성분만 남았다**([eval-noise.md](docs/experiments/eval-noise.md)). 고정 test paired bootstrap으로 micro 델타의 표본 잡음 sd가 0.18~0.21pt임이 확정됐고, 길이(+0.84pt)·모델(+0.99pt)·손실 3종 열세는 모두 이를 넘으며 `11_01`−exp2(−0.12pt)는 잡음 내다. 남은 것은 **시드(훈련) 잡음**이며, 한 런만 집행한다면 대상은 `11_01` 재현(현행 레시피·손실 축 기준선, A40 ~8.3h)이다 — 여유가 +0.13pt뿐인 focal−BCE가 이 성분에 직접 걸린다. KLUE-RoBERTa 대조군(주장 방어용, 선택)은 그다음.

## 함정 (놓치기 쉬움)

- **`train_sampling_strategy="group_by_length"`는 eval·predict 로더에도 적용된다.** `Trainer._get_eval_sampler`가 같은 설정을 보고 `LengthGroupedSampler`를 반환하므로 `trainer.predict`의 **반환 행이 길이 그룹 순열**로 나온다. 평가 지표는 라벨을 같은 순서로 모으니 멀쩡하지만(`11_01` test micro 0.8588은 유효), 덤프된 로짓은 데이터셋 행 순서를 전제하는 하류 분석과 전부 어긋난다. 순열은 `torch.randperm(generator=None)` 기반이라 **사후 복원이 불가능**하다 — 재덤프뿐이다. `runner.predict_logits`가 덤프 동안 순차 샘플러로 되돌리고 반환 라벨로 행 순서를 assert한다. 덤프를 하류에서 쓰기 전에 **점추정을 SSOT와 대조**해 행 축을 확인한다(`scripts/eval_noise_bootstrap.py`가 모델마다 자동으로 한다).
- **스케줄 길이가 다른 런을 에폭 눈금으로 비교하지 않는다.** `linear`+`warmup_ratio=0.1`은 총 스텝에 비례한다 — 2 epoch 탐색 런은 0.2 epoch에 피크 lr을 지나 곧바로 어닐링에 들어가고, 12 epoch 풀런은 1.2 epoch까지 워밍업 중이다. 같은 "1 epoch"이 lr 궤적에서 전혀 다른 자리이며, 조기 곡선 대조는 **12 epoch 런끼리** 한다.
- **`prep_cache`는 `{backbone}_len{max_len}`으로만 키잉된다.** 데이터셋 버전이 바뀌어도 볼륨에 캐시가 남아 있으면 원본 다운로드를 건너뛰어 **구 데이터로 학습된다.** 데이터셋을 갱신했으면 `/workspace/prep_cache/*`를 지우고 시작하고, `[schedule]` 출력의 step/epoch로 행 수를 역산해 확인한다(정리 데이터 = 1,576 step/epoch @ eff128).
- **정리 test(11,244)와 구 test(11,271)는 다른 셋이다.** 서로 다른 test에서 잰 micro를 나란히 놓지 않는다.
- **현행 대조 손실은 BCE가 아니라 `FocalLoss(alpha=0.25, gamma=2)`다.** `alpha`가 손실 전체에 곱해지는 상수라 클래스 균형 역할을 하지 않아 비대칭 처리가 사실상 γ 하나뿐이다(문서당 음성:양성 ≈156:1).
- **exp1은 12에폭을 전부 소진**하고 early stopping이 한 번도 발동하지 않았다(수렴이 아니라 예산 종료).
- **`classifier_pooling`은 이미 `mean`**(A.X-Encoder-base 기본값). CLS 병목 우려는 이 모델에 해당하지 않는다.

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
