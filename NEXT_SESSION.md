# 다음 세션 인계

> 읽는 순서: `PROJECT.md`(SSOT) → 이 문서. 수치 SSOT는 `output/*.json`, 손실 축은 `docs/experiments/loss-function.md`, 결정 경위는 `docs/adr/`.

## 지금 상태 — 계층 손실 축 종결, **다음 착수는 KD teacher soft target 덤프**

`14_01`(MCLoss 그룹 항)이 12 epoch 완주했고 실측·기제 분해·세 갈래 판정이 전부 기록돼 축이 닫혔다([ADR-0014](docs/adr/0014-hierarchy-loss-closure.md), SSOT [hierarchy-loss.md](docs/experiments/hierarchy-loss.md) · `output/hierarchy_loss_mass.json`·`output/hierarchy_loss_grad_budget.json`). **남은 활성 축은 KD와 모델·레시피뿐이다**(「KD 축」 절).

- **결과**: 정리 test micro **0.8467** = 앵커 `11_01`(0.8588) 대비 **−1.20pt**. 네 지표 모두 음수(macro −1.22 · sample −0.53 · anchor weighted −0.70pt). 문서 단위 paired bootstrap 4,000회 **CI95 [−1.604, −0.802]** · P(Δ≥0)=0.0000으로 시드 축 델타(−0.176pt)의 약 5.8배 — 도메인 축(−0.15pt, 잡음 내 = 효과 없음)과 달리 **실제 하락**이다. 그룹 항이 직접 최적화하는 `Lno` 유도 축도 평탄(0.9079→0.9074).
- **기제 = 표적이 아니라 전달의 실패.** 표적은 얇지 않았다(FP의 60.4%). 세 겹으로 갈린다.
  1. **기울기 예산 포획** — λ를 초기화 1배치의 손실 **값** 일치로 고정했는데, focal은 `α(1−p_t)^γ`·1/188로 자기소멸하고 그룹 항은 max 위의 로그손실이라 소멸하지 않는다. 기울기 비 group/focal이 초기화 0.34~0.36x → 종점 **2.12x**(그룹 항 몫 67.9%). 그 결과 `14_01`은 **앵커의 목적함수인 focal에서도 더 나쁜 지점**에 도착했다(0.001363 대 0.000852).
  2. **포화된 목적** — 앵커 종점에서 양성 그룹 88.7%·음성 그룹 99.4%가 이미 충족인데 그룹 항 기울기의 46.9%가 거기 실린다(결정 못 바꾸는 확신 부풀리기). 로짓 기하가 그대로 움직였다: 정답 +1.495 · 정답 `Lno` 밖 −3.811 · sd 1.708→2.434(시드 대조는 모두 |0.07| 이하).
  3. **형제 축 사각지대** — 정답 `Lno` 안 비정답 143,897개는 두 항 어디에도 안 닿는다(양성 그룹이라 음성 항 없음, 비정답이라 max가 안 가져감). 이 밴드가 임계 쪽으로 +0.435 떠올라 within `Lno` FP +253이 났고, 조건부 형제 top-1이 **−0.53pt [−0.87, −0.18]** 로 시드 잡음(−0.12pt [−0.46, +0.21]) 밖으로 나갔다. `Lno` 축 하락(−0.27pt)은 잡음권이다.
- **구성은 옮기고 총량은 못 줄임.** 작동점을 맞추면(arm τ 0.620) 정답 `Lno` 밖 FP가 1,211→1,148(−63)로 의도한 방향이나 대가가 within `Lno` FP +61과 FN +324다. 어느 작동점에서도 micro가 회복되지 않아 캘리브레이션이 아니라 순위 손실이다.
- **하위 갈래 상한이 산술로 닫혔다**: λ 변형은 훈련 곡선이 대리 실험이고(24 eval 지점 중 **23개 음수**, 1~3 epoch 평균 −1.00pt, 유일한 양은 작동점 아티팩트) 상한이 "앵커와 같아짐"이다. BCE 기반 교체는 λ가 12.6배가 돼 그룹 항 몫을 74.9~80.3%로 **키우고**, 기반선도 −0.53pt에 표적 구성이 focal과 같다(cross FP 1,212 대 1,211). 표적 효과 −63을 공짜로 얹는 도달 불가 가정에서도 focal 기반 +0.20pt · BCE 기반 −0.33pt로 형식 하한 +0.4pt에 미달한다.
- **아티팩트 위치**: 지표 `output/modernbert-patent-len512-mcl_metrics.json` · 로짓 `output/logits_modernbert-patent-len512-mcl_{val,test}.npy`(행 축 정상 — 로짓 재계산 micro가 훈련 잡 지표와 4자리 일치) · 모델 `ingyoun/A.X-patent-len512-mcl` · 실행 기록 `notebook_output/14_01_HierLoss_MCLoss.ipynb`.

### 직전 축 — 도메인 사전학습 종결

`13_02`(TAPT 백본 분류 파인튜닝)가 완주해 도메인 사전학습 축이 **기준 미달로 종결**됐다([ADR-0013](docs/adr/0013-domain-pretraining-closure.md), 실측 SSOT [domain-pretraining.md](docs/experiments/domain-pretraining.md)).

- **`13_02` 결과**: 정리 test micro **0.8572**(macro 0.8545 · sample 0.8724 · anchor weighted 0.8198)로 앵커 `11_01`(0.8588) 대비 **−0.15pt**. 판정선 +0.4pt 미달이고 네 지표 부호가 모두 음수이나 폭은 표본 잡음(sd 0.18~0.21pt) 안이다 — "해롭다"가 아니라 **효과 없음**이다.
- **곡선에도 이득 없음**: eval 24지점 짝지은 델타 평균 −0.03pt · sd 0.43pt, 3 epoch까지 평균 +0.04pt. 초기 우위로 보이던 ep1.0(+0.69)·ep1.5(+0.67)는 ep2.0(−1.29)이 되돌린다. TAPT의 통상 이득인 **초기 수렴 가속조차 없다**(첫 eval 0.6385 대 0.6380).
- **기제 = 코퍼스 동일성**: 같은 train split 201,616 문서에서 TAPT 5 epoch가 802M 토큰인 반면 분류 파인튜닝 12 epoch는 **1,116M 토큰**(len512 절단, 문서당 평균 461)이다. 파인튜닝이 같은 텍스트를 1.4배 더, 라벨 감독과 함께 본다 — MLM이 넣을 정보가 없다.
- **⚠️ `13_02` 로짓은 행 순열로 깨져 삭제했다.** 팟의 `src/patent_train`이 구 사본이라 `predict_logits`의 행 순서 방어가 없는 코드로 돌았다(「함정」 1). 런이 낸 지표는 유효하나 paired bootstrap CI·길이 bin slope는 미집행으로 남는다. 모델은 Hub(`ingyoun/A.X-patent-len512-tapt`)에 있으니 필요하면 `11_03`식 추론 전용 경로로 재덤프한다.

## 완료된 축

| 축 | 결론 | SSOT |
| --- | --- | --- |
| 손실 함수 | **종결** — ZLPR 0.8493 · ASL 0.8362 · BCE 0.8538이 모두 focal exp2 0.8601 미달. 기제는 FP:FN 부호 뒤집힘(k=1 과대예측 vs k≥2 과소예측). DL2 미착수 확정 | [ADR-0009](docs/adr/0009-loss-axis-closure.md) · [loss-function.md](docs/experiments/loss-function.md) |
| 카디널리티 디코딩 | **음성** — raw 확률 기대-F1 plug-in이 k≥2를 +0.90pt 회수하나 k=1 과대예측과 분리 불가로 전역 −0.22pt. 오라클-k +1.60pt는 도달 불가 상한 | [cardinality-decoding.md](docs/experiments/cardinality-decoding.md)(`10_01`) |
| 장문 열화 | **디프리오리티** — 최장 문서도 정답 top-5 ~98% 잔존이라 표현 붕괴가 아니다. label-aware attention 풀링 헤드룸 <~0.5pt | [longdoc-degradation.md](docs/experiments/longdoc-degradation.md)(`10_02`) |
| 계층 확장(`Lno` 게이트) | **하드 top-1 게이트 불채택** — 단일 `Lno` 게이트의 micro recall 상한 0.8590이 exp1 현재 recall 0.8697 아래(4모델 동일). 기존 2단계 추정(`Lno` 정확도 × 전체 오라클)은 1단계 실패를 이중 계상한 결함 추정량이고, 조건부로 교정하면 flat과 **항등**이라 판정력이 0이었다. 라벨 형상도 하드 게이팅을 배제한다(두 단계 모두 다중레이블이어야 하고, 그러면 표현력이 flat과 같다). **게이트 없는 형태(보조 손실)는 `14_01`로 측정돼 기각됐고**([ADR-0014](docs/adr/0014-hierarchy-loss-closure.md)), **훈련된 조건부 2단계**는 추론 구조 변경을 동반해 이 행에 걸린다. 2단계 상한 +3.48pt는 sibling 질량인데 주 지표 결손의 다수는 형제가 아니라 cross-`Lno` 두 번째 라벨이다(k≥2 FN 1,064 중 cross 589·형제 475) | [ADR-0002](docs/adr/0002-flat-multilabel-hierarchy.md) · [modernbert-comparison.md](docs/experiments/modernbert-comparison.md)「오류 구조」 · [data.md](docs/data/data.md)「계층 형상」(`scripts/hierarchy_conditional.py`·`scripts/multilabel_shape.py`) |
| 도메인 사전학습 | **종결** — 자체 코퍼스 TAPT가 정리 test micro 0.8572로 앵커 0.8588 대비 −0.15pt(판정선 +0.4pt 미달). 기제는 코퍼스 동일성(TAPT 802M 대 파인튜닝 1,116M 토큰, 같은 문서). MLM 체크포인트 교체·기성 `KorPatElectra` 모두 불채택 | [ADR-0013](docs/adr/0013-domain-pretraining-closure.md) · [domain-pretraining.md](docs/experiments/domain-pretraining.md)(`13_01`·`13_02`) |
| 계층 손실(보조 손실) | **종결** — MCLoss 그룹 항이 정리 test micro 0.8467로 앵커 대비 −1.20pt(잡음 밖). 기제는 기울기 예산 포획(종점 67.9%)·포화된 목적(기울기의 46.9%가 이미 충족된 그룹)·형제 축 사각지대(조건부 형제 top-1 −0.53pt). λ 변형·BCE 기반 교체 모두 상한이 판정선 아래. 계층 축은 추론 구조·훈련 신호 양쪽에서 닫혔다 | [ADR-0014](docs/adr/0014-hierarchy-loss-closure.md) · [hierarchy-loss.md](docs/experiments/hierarchy-loss.md)(`14_01`) |
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
| `11_04` (A.X seed153) | 512 | focal | eff128/lr4.8e-4 | 0.8570 | 시드 축 측정용 앵커 재현(Δ −0.176pt) |
| `14_01` (A.X MCLoss) | 512 | focal+MCL λ0.0444 | eff128/lr4.8e-4 | 0.8467 | 계층 손실 1런(앵커 −1.20pt, **잡음 밖**) — 축 종결([ADR-0014](docs/adr/0014-hierarchy-loss-closure.md)) |

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

1. **KD teacher soft target 덤프 — 다음 착수.** 계층 손실 축이 닫혀 활성 축은 KD와 모델·레시피뿐이다. 게이트는 GREEN이고 설계가 확정돼 있다(「KD 축」 절) — 정리 train 201,616 문서에 teacher 3종 로짓을 덤프하는 것이 다음 GPU 작업이다. exp1@8192 추론이 주 비용이고 ASL·KoBERT@512는 저렴하며, 순차 샘플러로 행 순서를 `document_id`에 고정한다(순열 함정).
   - **재사용 가능한 도구**: `scripts/hierarchy_loss_mass.py`에 `paired_bootstrap`(문서 단위 CI)과 `matched_operating_point`(작동점 정규화), `scripts/hierarchy_loss_grad_budget.py`에 기울기 예산·포화도·순위 국소화(P@1을 `Lno` 축과 조건부 형제 축으로 분해)가 들어 있다. arm 추가는 두 스크립트 모두 상단 `MODELS`/`TAGS` 사전만 손대면 된다.
2. **오류 부분집합의 관찰 지표 규약은 교정됐다 — 어떤 축이든 절대량·문서당으로 잰다.** `14_01`이 훈련 중 관찰 지표를 share(표적/전체 FP)로 뒀던 것이 결함이었다(「함정」의 share 항목). 교정된 프로토콜은 [hierarchy-loss.md](docs/experiments/hierarchy-loss.md)「프로토콜」에 있고 KD 런에도 그대로 적용한다. `14_01` 노트북은 실행 기록이라 수정하지 않았다.
3. **`11_01` 로짓은 행 축이 정상이다 — 재덤프 불필요.** `output/logits_modernbert-patent-len512-b128_{val,test}.npy`를 정리 데이터셋 행 순서의 라벨과 대면시키면 test micro/macro/sample이 SSOT(0.858759 / 0.856503 / 0.873791)와 1e-6까지 일치하고 val도 0.8623으로 정상이다(순열이면 ~0.006이 나온다). 정리 로짓의 행 축 SSOT는 `output/doc_ids_clean_{val,test}.json`이며 데이터셋 `document_id` 순서와 일치한다(구 `doc_ids_*`는 구 로짓 재현용 유지, `docs/data/data.md`「주의」). **비교선은 정리 test 재계산값(exp2 0.8599)**이다 — `11_01`은 정리 test(11,244)에서 평가되므로 구 test 수치(0.8601)와 직접 대면 안 된다.
4. **`11_01`에는 판정할 두 축이 겹쳐 있다** — 데이터 클리닝과 신 레시피가 동시에 바뀐 런이다. 클리닝 효과는 볼륨(train 0.04%)이 작아 aggregate에서 분리되지 않으므로([ADR-0010](docs/adr/0010-data-cleaning.md)), 연루 클래스(특히 EB01) per-class F1을 paired로 대조한다.
5. **RoBERTa·KoBERT 토큰화본 336 필터 미반영** — `ingyoun/patent-clean-text-roberta-tokenized`·`...-kobert-tokenized`는 정리 이전 상태다. 재토큰화 없이 같은 `document_id`로 필터링하면 되고, KLUE-RoBERTa 대조군이나 KoBERT 재현을 다시 돌릴 때 선행한다.
6. **KLUE-RoBERTa 대조군**(주장 방어용, 선택 항목 — [klue-roberta.md](docs/experiments/klue-roberta.md)). 예산이 남을 때만 집행한다.

## 함정 (놓치기 쉬움)

- **팟의 `src/patent_train`을 로컬 최신본으로 교체하고 시작한다.** 볼륨·이미지에 남은 구 사본이 그대로 import되면 **로컬에서 고친 코드가 반영되지 않은 채** 런이 돈다. `13_02`가 이 경로로 순열 로짓을 냈다 — `runner.predict_logits`의 행 순서 방어(순차 샘플러 복원 + 반환 라벨 assert)는 로컬에 이미 들어와 있었으나 팟 사본이 구본이었다. 런 초반에 `patent_train.__file__`과 방어 코드 존재를 찍어 확인한다(`11_03` 3번 셀 방식).
- **스케줄 길이가 다른 런을 에폭 눈금으로 비교하지 않는다.** `linear`+`warmup_ratio=0.1`은 총 스텝에 비례한다 — 2 epoch 탐색 런은 0.2 epoch에 피크 lr을 지나 곧바로 어닐링에 들어가고, 12 epoch 풀런은 1.2 epoch까지 워밍업 중이다. 같은 "1 epoch"이 lr 궤적에서 전혀 다른 자리이며, 조기 곡선 대조는 **12 epoch 런끼리** 한다.
- **`prep_cache`는 `{backbone}_len{max_len}`으로만 키잉된다.** 데이터셋 버전이 바뀌어도 볼륨에 캐시가 남아 있으면 원본 다운로드를 건너뛰어 **구 데이터로 학습된다.** 데이터셋을 갱신했으면 `/workspace/prep_cache/*`를 지우고 시작하고, `[schedule]` 출력의 step/epoch로 행 수를 역산해 확인한다(정리 데이터 = 1,576 step/epoch @ eff128).
- **정리 test(11,244)와 구 test(11,271)는 다른 셋이다.** 서로 다른 test에서 잰 micro를 나란히 놓지 않는다.
- **오류의 부분집합을 share(부분/전체)로 관찰하지 않는다 — 절대량·문서당으로 잰다.** 분모인 전체 FP·FN이 함께 움직여 방향이 뒤집혀 보인다. `14_01`의 `fp_cross_lno_share`가 val 곡선에서 0.365→0.544로 올라 표적 악화로 읽혔으나 표적 절대량은 평탄했고(문서당 0.094→0.106) 움직인 것은 분모였다(전체 FP 문서당 0.258→0.195). test 최종 지점에서는 같은 결함이 반대 방향이었다 — share 0.604→0.553으로 개선처럼 보이는데 절대량은 1,211→1,294로 늘었다.
- **예측량이 다른 두 런의 오류 칸을 τ=0.5에서 직접 비교하지 않는다.** empty rate·문서당 예측 라벨이 다르면 모든 칸이 함께 움직인다. arm의 τ를 조정해 기준선의 전체 FP나 문서당 예측 수에 맞추고 대조한다(`scripts/hierarchy_loss_mass.py`의 `matched_operating_point`). `14_01`에서 이 정규화가 표적 증감의 부호를 뒤집었다(cross FP +83 → −63). 이 τ는 진단용이며 임계 정책이 아니다.
- **보조 손실의 계수는 손실 *값*이 아니라 *기울기 몫*으로 맞춘다.** 두 항의 소멸 속도가 다르면 값 일치는 계통 오차가 된다 — focal은 `α(1−p_t)^γ`로 자기소멸하고 그룹 항은 소멸하지 않아, 초기화에서 값을 1:1로 맞춘 λ가 종점에서 기울기 2.12x(그룹 항 몫 67.9%)가 됐다. 초기화 시점은 하필 보조항의 영향이 **최소**인 지점이다(`scripts/hierarchy_loss_grad_budget.py`).
- **하락을 국소화할 때는 임계값 무관 지표를 쓴다.** P@1을 `Lno` top-1과 조건부 within-`Lno` top-1로 갈라 시드 쌍둥이(`11_04`)와 대면시키면 순위 손실이 어느 축에 실렸는지 드러난다 — `14_01`에서 `Lno` 축(−0.27pt)은 잡음권이고 형제 축(−0.53pt)만 잡음 밖이었다.
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
