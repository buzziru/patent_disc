# 프로젝트 스펙 — P1 · 특허 과학기술표준분류 분류

> 이 문서가 프로젝트 목표·접근·제약의 **SSOT**다. (`owner/`는 초기 브리핑·참고 자료일 뿐 SSOT가 아니다.)
> 데이터 상세는 [`docs/data/data.md`](./docs/data/data.md), 훈련 인프라는 [`docs/`](./docs/README.md).

## 목표

특허 문헌을 **과학기술표준분류 188개 중분류**로 자동 분류하는 **인코더 분류기**를 만들고, 공식 baseline 대비 개선을 정량 입증한다. 17개 대분류와의 **계층 일관성**을 확보한다.

## 접근

- **문서당 다중 레이블(multi-label) 188-way 분류.** 한 특허가 여러 중분류에 대응한다(고유 문서의 ~14%가 2~10개 `Mno`, 평균 ~1.2개 — `docs/data/data.md`). 188-way **sigmoid + BCE**(`problem_type=multi_label_classification`)로 flat baseline을 세운다. 대분류(`Lno`)는 **예측된 각** `Mno` → `Lno` lookup으로 유도한다(별도 Lno 헤드 불필요 → 계층 비일관성 회피).
- 오류 분석에서 **같은 대분류 내 형제 혼동**이 지배적이면 계층 구조(대분류 조건부)로 확장한다.
- ⚠️ **독립된 두 헤드(대분류·중분류 별도 예측) 구성은 피한다** — 계층 비일관성을 유발. `Mno` 다중 예측 + `Lno` lookup으로 일관성을 유지한다.

## 차별점 — long-document 처리

명칭(`invention_title`)·요약(`abstract`)·청구항(`claims`) 중 **입력 필드를 선택**하고, 512 토큰 truncation에 묶인 기존 baseline의 한계를 **장문 인코더**로 개선하는 것을 축으로 삼는다. 장문의 실체는 주로 `claims`(별도 상세설명 필드는 데이터에 없음 — `docs/data/data.md`). 입력 필드 조합이 핵심 실험 변수.

## 모델

- **주 모델: `skt/A.X-Encoder-base`** (Hugging Face, 확정). ModernBERT 아키텍처의 **한국어(+영어) 인코더** — 프로젝트의 long-document 축에 정확히 부합.
  - 스펙(config 확인): `model_type=modernbert`, **149M 파라미터**, **`max_position_embeddings=16,384`**(표준 ModernBERT 8,192의 2배), hidden 768 / 22 layers / 12 heads, vocab 50,000, bf16.
  - attention: local window 128 + **global attention every 3 layers**, RoPE(local θ=10,000 / global θ=160,000).
  - `AutoModelForMaskedLM`(fill-mask) 기반 → 분류 head를 얹어 fine-tune. **라이선스 apache-2.0**(가용성·라이선스 확인 완료).
- **대조군: KLUE-RoBERTa-large (512 토큰)** — truncation 대 장문의 이득을 대조하는 안정적 비교선.
- **KoBERT** — 공식 baseline 모델이나 2026 기준 레거시. 재현·비교용 참조로만.

> 입력 최대 길이는 16,384까지 가능하나, 실제 시퀀스 길이는 필드 조합·메모리와의 트레이드오프로 실험한다(전 문서를 16k로 패딩할 필요 없음).

## 평가 프로토콜 (baseline 주의)

**0.8249의 정체(참고 코드 `소스코드/03_model_test.ipynb` 실측):** baseline(KoBERT)은 **훈련은 멀티레이블**(FocalLoss=BCE 기반)이나 **평가는 top-1**(`argmax` 1개) 예측을 단일화한 정답과 sklearn `f1_score`로 잰다. 결과 **Micro 0.8261 / Macro 0.8038 / weighted 0.8249** — 즉 **`0.8249` = full test(24,525건)의 weighted-F1(top-1)**이다. (종전 "10.9% 서브셋" 서술은 재현 코드와 모순되어 폐기.)

- **그 숫자를 절대 기준으로 쓰지 말 것.** baseline의 실제 입력(`datadam_20230116.plk`, `class04`=188중분류)은 리포에 **없어**(참고 폴더 `patent_label_data.plk`는 KSIC 업종분류 44-class로 무관) **동일 test set 재현이 불가**하다. 또 `300,240`은 고유 문서 수가 아니라 **(문서,레이블) 쌍 수**이며 우리 로컬 zip은 다른(작은) vintage(고유 224,328건 — `docs/data/data.md`).
- **비교선은 자체 수립**: 우리 데이터에서 **test split을 고정**(이미 업로드: test 11,217)하고, **KoBERT baseline을 그 test set 위에서 직접 재현**해 비교한다(0.8249 숫자를 그대로 쓰지 않음).
- **주 비교 지표 = top-1 예측 weighted-F1**(baseline과 동일 계산). 멀티레이블 프레이밍용으로 **micro/macro-F1(임계값 0.5, 검증셋 튜닝)·P@1/3/5**를 병기한다.
- **분할은 `documentId`(=고유 특허) 단위.** 제공된 `Training`/`Validation` 폴더는 **7,822개 문서가 양쪽에 겹쳐**(누수) 그대로 쓸 수 없다 — 전체를 고유 문서로 집계한 뒤 재분할한다.

## 스코프·한계

- **멀티모달은 스코프 밖.** 도면 이미지가 있으나 사용하지 않는다.
- 중분류당 대체로 1,300~2,600건으로 **인위적으로 평탄화된 분포** — 실제 출원 분포와 다르다. 결과 해석 시 한계로 명시한다.

## 데이터

AI Hub 71531(과학기술표준분류 대응 특허 데이터). 300,240건, 17대분류/188중분류, 출처 KIPRIS. 승인 필요·내국인 한정·샘플 제공.
입력은 명칭·요약·청구항(상세설명 필드 없음), 라벨은 JSON(`Lno`/`Mno`, 문서당 다중 가능). 레이아웃·스키마·조인·카테고리 상세 → [`docs/data/data.md`](./docs/data/data.md).
