# 프로젝트 스펙 — P1 · 특허 과학기술표준분류 분류

> 이 문서가 프로젝트 목표·접근·제약의 **SSOT**다.  
> 데이터 상세는 `[docs/data/data.md](./docs/data/data.md)`, 훈련 인프라는 `[docs/](./docs/README.md)`.

## 목표

특허 문헌을 **과학기술표준분류 188개 중분류**로 자동 분류하는 **인코더 분류기**를 만들고, 공식 baseline 대비 개선을 정량 입증한다. 17개 대분류와의 **계층 일관성**을 확보한다.

**산출물의 성격**: 최종 결과물은 **운영 가능한 단일 모델의 성능**과 **그 성능을 끌어올린 방법론**이다. 특정 가설의 엄밀한 증명이 아니다. long-document는 프로젝트의 출발점이 된 아이디어이고 이미 실측으로 지지됐으므로(아래 「long-document 축」), 그 축을 더 엄밀하게 검증하는 데 예산을 쓰지 않는다. 실험 선택의 기준은 **성능 개선 기대치**이며, 기법은 근거 문헌을 먼저 확보한 뒤 적용한다.

## 접근

- **문서당 다중 레이블(multi-label) 188-way 분류.** 한 특허가 여러 중분류에 대응한다(고유 문서의 ~~14%가 2~~10개 `Mno`, 평균 ~1.2개 — `docs/data/data.md`). 188-way **sigmoid + Focal Loss**(γ=2, `problem_type=multi_label_classification`)로 flat baseline을 세운다. 대분류(`Lno`)는 **예측된 각** `Mno` → `Lno` 매핑으로 유도한다(별도 Lno 헤드 불필요 → 계층 비일관성 회피).
  - 현행 손실 구현의 `alpha`는 손실 전체에 곱해지는 상수라 **클래스 균형 역할을 하지 않는다**(표준 α-balanced focal은 양성·음성에 서로 다른 가중을 준다). 문서당 음성:양성이 약 156:1인데 비대칭 처리는 사실상 γ 하나뿐이다 — 손실 축은 이 비대칭·카디널리티를 겨냥했으나 focal을 넘지 못해 종결됐고([ADR-0009](docs/adr/0009-loss-axis-closure.md), 실측 `docs/experiments/loss-function.md`), 그 **k≥2 헤드룸은 이제 이종 앙상블 KD 축이 승계**한다(`docs/experiments/knowledge-distillation.md`).
- **계층 구조(대분류 조건부) 확장은 하지 않는다 — flat 유지.** 하드 `Lno` 게이트는 정답 양성 라벨의 14.1%를 도달 불가로 만들어 micro recall 상한을 0.8590으로 묶는데 exp1의 현재 recall이 0.8697이라, 주 지표에서 시작부터 손해다(정답 `Lno`가 2개 이상인 문서 8.77%, k≥2 문서로는 58.4%). P@1 축에서는 계층 마스킹이 flat과 **항등으로 동치**여서 즉시 이득도 없다. 형제 혼동 자체는 우연의 5.2배로 잦으나(36.6% vs 우연 7.0%), 그 사실이 조건부 구조의 이득을 뜻하지는 않는다(`docs/experiments/modernbert-comparison.md` 「오류 구조」).
  - 라벨 형상도 하드 게이팅을 배제한다 — 다중레이블 문서가 순수 cross-`Lno` 46.1% / 순수 within-`Lno` 41.4% / 혼합 12.5%로 갈려, `Lno`·`Mno` 두 단계를 모두 다중레이블로 두어야 데이터를 표현한다(`Lno`당 단일 `Mno`면 전체 문서의 8.22%가 표현 불가·recall 상한 0.9138). 그런데 둘 다 다중레이블이면 표현력이 flat 188-way와 같아진다 — **계층으로 얻을 수 있는 것은 표현력이 아니라 파라미터화·손실 구조뿐이다**(`docs/data/data.md` 「다중레이블의 계층 형상」).
  - 다만 **훈련된 조건부 2단계의 이득은 측정된 적이 없다** — 로짓 마스킹 시뮬레이션은 위 동치 때문에 flat을 재현할 뿐 탐지 능력이 없다. 경계값만 알려져 있다: 2단계 최적화의 P@1 상한은 1단계 정확도 0.9398(flat 대비 +3.48pt = sibling 오류 질량), 현행 조건부 정확도는 0.9630, 소프트 top-2 게이트의 recall 상한은 0.9672다. 단 주 지표 결손의 다수는 형제가 아니라 cross-`Lno` 두 번째 라벨이다(k≥2 FN 1,064 중 cross 589·형제 475).
- ⚠️ **독립된 두 헤드(대분류·중분류 별도 예측) 구성은 피한다** — 계층 비일관성을 유발. `Mno` 다중 예측 + `Lno` 매핑으로 일관성을 유지한다.

## long-document 축 — 실측 완료

512 토큰 truncation에 묶인 기존 baseline을 **장문 인코더**로 개선한다는 것이 프로젝트의 출발 아이디어였다. 아래 실측으로 지지됐고, **추가 검증은 진행하지 않는다**(「닫힌 갈래」).

입력은 명칭(`invention_title`)·`ipc_main`·요약(`abstract`)·청구항(`claims`)을 공백으로 이어 붙인 고정 조합이며, 전 실험이 같은 조합을 쓴다. 장문의 실체는 주로 `claims`다(별도 상세설명 필드는 데이터에 없음 — `docs/data/data.md`).

**실측 1차 결과**(고정 test 11,271, **KoBERT 재현선** 대비 — 공식 0.8249 아님, 상세 `docs/experiments/modernbert-comparison.md`): A.X-Encoder(8192)가 멀티라벨 micro **0.8502 → 0.8685**(+1.83pt), 앵커 top-1 weighted **0.8148 → 0.8256**으로 재현선을 넘었다. 512 control(exp2)로 분해하면 **컨텍스트 길이에 +0.84pt**(exp1−exp2 — 같은 모델·같은 토크나이저라 길이에 통제 귀속), **모델 성분에 +0.99pt**(exp2−KoBERT)다. 창 확장 효과는 길이 bin에서 단조 증가(exp1−exp2 micro Δ: B0 +0.46 → B3 +2.64pt)해 **장문 가설을 지지한다.** 모델 성분은 아키텍처·사전학습·토크나이저에 더해 토크나이저 압축이 512 창에 만든 커버리지 우위(절단 문서에서 ~10% 더 많은 본문 — 실측 `docs/experiments/no-train-analysis.md` C)가 섞인 값이라 순수 아키텍처 이득으로 읽지 않으며, 그 안의 배분은 주장하지 않는다. headline은 "장문 + 더 나은 한국어 인코더의 결합 효과"로 서술하되, 주 동력은 512 창이 버리던 본문의 회복이다.

## 모델

- **주 모델: `skt/A.X-Encoder-base`** (Hugging Face, 확정). ModernBERT 아키텍처의 **한국어(+영어) 인코더** — 프로젝트의 long-document 축에 정확히 부합.
  - 스펙(config 확인): `model_type=modernbert`, **149M 파라미터**, `**max_position_embeddings=16,384`**(표준 ModernBERT 8,192의 2배), hidden 768 / 22 layers / 12 heads, vocab 50,000, bf16.
  - attention: local window 128 + **global attention every 3 layers**, RoPE(local θ=10,000 / global θ=160,000).
  - `AutoModelForMaskedLM`(fill-mask) 기반 → 분류 head를 얹어 fine-tune. **라이선스 apache-2.0**(가용성·라이선스 확인 완료).
- **대조군: KLUE-RoBERTa-base (512 토큰)** — **선택 항목.** hidden 768·base급으로 A.X와 크기를 맞춰(~110M vs 149M) 모델 성분에서 크기 confound를 걷어내는 용도다. 성능 향상이 아니라 주장 방어가 목적이므로 예산이 남을 때만 집행한다(`docs/experiments/klue-roberta.md`).
- **KoBERT** — 공식 baseline 모델이나 2026 기준 레거시. 재현·비교용 참조로만.

> 입력 최대 길이는 16,384까지 가능하나, 실제 시퀀스 길이는 필드 조합·메모리와의 트레이드오프로 실험한다(전 문서를 16k로 패딩할 필요 없음).

## 평가 프로토콜 (baseline 주의)

**0.8249의 정체(참고 코드 `소스코드/03_model_test.ipynb` 실측):** baseline(KoBERT)은 **훈련은 멀티레이블**(FocalLoss=BCE 기반)이나 **평가는 top-1**(`argmax` 1개) 예측을 단일화한 정답과 sklearn `f1_score`로 잰다. 결과 **Micro 0.8261 / Macro 0.8038 / weighted 0.8249** — 즉 **`0.8249` = full test(24,525건)의 weighted-F1(top-1)**이다.

- **그 숫자를 절대 기준으로 쓰지 말 것.** baseline의 데이터 분할은 **`documentId`가 train·val 양쪽에 존재하는 데이터 누수** 위험이 있어, **누수 없는 데이터셋을 새로 생성**해 비교선을 세운다. 또 `300,240`은 고유 문서 수가 아니라 **(문서,레이블) 쌍 수**이며 로컬 zip은 다른(작은) 스냅샷이다(고유 224,328건 — `docs/data/data.md`).
- **비교선은 자체 수립**: 재생성한 데이터에서 **test split을 고정**(재생성 split: train 201,895 / val 11,162 / test 11,271)하고, **KoBERT baseline을 그 test set 위에서 직접 재현**해 비교한다(0.8249 숫자를 그대로 쓰지 않음).
- **주 비교 지표 = 멀티라벨 micro-F1**(τ=0.5, 문서별 다중-핫 예측). 개선 판정과 체크포인트 선택 모두 이 축에서 한다 — 표준성·비교가능성 때문이다. 지표 서열은 **micro(주 헤드라인) · sample-F1(제품-대면 보조, 문서 단위 집합 일치) · macro(188 클래스 견고성 점검) · empty rate(캘리브레이션)**이며 LRAP·R-Precision을 함께 본다.
  - **sample > micro는 고카디널리티(다라벨) 문서가 코퍼스 평균보다 어렵다는 신호다** — 직접 근거는 k별 분해(`docs/experiments/modernbert-comparison.md`, k≥2 문서가 성능을 끌어내림)가 소유한다.
  - **micro−macro 격차는 ~0.3pt로 작다** — 평탄화된 분포(중분류당 1,300~2,600건, 실제 출원 분포와 다름) 탓에 micro가 이 데이터에서 가리는 클래스별 약점은 크지 않다. 결과 보고 시 한계로 병기한다.
  - **여기서의 weighted-F1은 멀티라벨 클래스-weighted이며, baseline의 top-1 weighted-F1(0.8249)과 다른 양이다** — 병기 시 반드시 축을 구분해 표기한다.
- **top-1 예측 weighted-F1·P@1/3/5는 벤더 baseline의 레거시 지표다.** 공식 baseline이 top-1로 평가했기에 동일 계산으로 재현·병기하지만, 프로젝트의 개선 판정 기준이 아니다. `docs/experiments/kobert-baseline.md`가 "headline"이라 부르는 top-1 weighted-F1은 **baseline 재현 범위에 한정된 표현**이다.
- **분할은 `documentId`(=고유 특허) 단위.** 제공된 `Training`/`Validation` 폴더는 **7,822개 문서가 양쪽에 겹쳐**(누수) 그대로 쓸 수 없다 — 전체를 고유 문서로 집계한 뒤 재분할한다.

## 닫힌 갈래 (재검토하지 않는다)

측정으로 결론이 난 항목이다. **새로운 근거 없이 다시 제안하지 않는다.**

| 갈래 | 결론 | 근거 |
| --- | --- | --- |
| 계층 구조(대분류 조건부) 확장 — **하드 게이트 계보** | 불채택 | 단일 `Lno` 게이트의 micro recall 상한 0.8590 < exp1 현재 recall 0.8697이라 주 지표에서 즉시 손해이고, `Lno`당 단일 `Mno`는 라벨 형상이 배제한다(전체 8.22% 표현 불가). P@1 축에서는 마스킹이 flat과 항등 동치라 이득도 없다. **게이트 없는 형태(계층 정규화·보조 손실)와 훈련된 조건부 2단계는 미측정**이며 되열려면 실제 훈련이 필요하다(상한 +3.48pt, 조건부 현재 0.9630) — `docs/experiments/modernbert-comparison.md` 「오류 구조」. 게이트 없는 형태는 **`docs/experiments/hierarchy-loss.md`(MCLoss 그룹 항)가 집행한다** — 추론 구조를 바꾸지 않으므로 이 행의 폐쇄(하드 게이트 계보) 밖이다 |
| 절대 임계값 정책(global τ · per-class τ) | global τ 적용, 그 이상 없음 | global τ는 오라클 헤드룸이 +0.03~0.19pt로 레버가 아니다. per-class τ는 헤드룸 +1.5pt이나 클래스당 양성 표본 ~71개로 추정이 불가능하다(손익분기 ~140) — `docs/experiments/no-train-analysis.md` B |
| 로짓 캘리브레이션(temperature·아핀 변환) | 불채택 | 임계값 정책과 **결정 등가**다 — 온도는 τ=0.5에서 소거되고, 아핀 이동은 per-class τ와 같다. 길이 bin별 조건부도 오라클 헤드룸이 +0.08~0.24pt에 그친다 |
| 앙상블 | 불채택 | 운영 환경이 단일 모델을 요구한다. 3모델 로짓평균이 앵커 weighted-F1을 0.8256 → 0.8327(+0.71pt)로 올리나 추론 비용이 3배다. **단, 앙상블을 훈련 시점 teacher로만 쓰고 배포는 단일 student인 KD는 별개 활성 축이다**(추론 단일 모델 유지 — `docs/experiments/knowledge-distillation.md`) |
| 입력 필드 조합 실험 | 후순위 | long-document 축의 추가 검증에 해당하며 목표가 아니다. 전 실험은 고정 조합을 쓴다 |
| long-document 가설 추가 검증 | 종료 | 「long-document 축」에서 실측 완료 |
| 도메인 사전학습(백본 표현 품질) | 불채택 | 자체 코퍼스 TAPT가 정리 test micro 0.8572로 앵커 `11_01`(0.8588) 대비 −0.15pt — 판정선 +0.4pt 미달이고 초기 수렴 가속조차 없다. 기제는 **코퍼스 동일성**: TAPT 5 epoch 802M 토큰 대 분류 파인튜닝 12 epoch 1,116M 토큰을 같은 train split에서 보므로 MLM이 넣을 정보가 없다. MLM 체크포인트 교체는 용량-반응 양 끝이 0.15pt 간격이라 근거가 없고, 기성 `KIPI-ai/KorPatElectra`는 512 캡(길이 성분 접근 불가)·gated 비상업 라이선스로 운영 모델이 될 수 없다 — [ADR-0013](docs/adr/0013-domain-pretraining-closure.md), `docs/experiments/domain-pretraining.md` |
| 표현·풀링(풀링 교체·label-aware attention 헤드) | 불채택 | hidden-state 덤프 + 고정 풀링 동결 프로브에서 항희석 풀링이 최장 bin에서 mean 대비 +0.32pt(게이트 아래)·concat 무상보 — 장문 열화는 표현·풀링·헤드로 회수 안 되는 본질적 난이도([ADR-0012](docs/adr/0012-representation-pooling-closure.md), `docs/experiments/longdoc-degradation.md`「풀링 실측」) |

## 스코프·한계

- **멀티모달은 스코프 밖.** 도면 이미지가 있으나 사용하지 않는다.
- 중분류당 대체로 1,300~2,600건으로 **인위적으로 평탄화된 분포** — 실제 출원 분포와 다르다. 결과 해석 시 한계로 명시한다.

## 데이터

AI Hub 71531(과학기술표준분류 대응 특허 데이터). 300,240건, 17대분류/188중분류, 출처 KIPRIS. 승인 필요·내국인 한정·샘플 제공.
입력은 명칭·요약·청구항(상세설명 필드 없음), 라벨은 JSON(`Lno`/`Mno`, 문서당 다중 가능). 레이아웃·스키마·조인·카테고리 상세 → `[docs/data/data.md](./docs/data/data.md)`.