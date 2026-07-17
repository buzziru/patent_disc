# 무훈련 분석 3종 — 계획 (오류 분해 · 임계값 튜닝 · 토크나이저 분석)

> **목적**:  (A) 계층 확장 여부를 결정할 오류 구조 데이터, (B) macro-F1 성능 레버(임계값 정책), (C) 「길이 vs 모델 분해」(`[modernbert-comparison.md](./modernbert-comparison.md)`)에서 모델 성분에 섞인 커버리지 성분의 기제 근거. 결과 수치는 `output/*.json`을 SSOT로 두고 해석은 비교 문서에 반영한다.

## 공통 전제 — 로짓 재확보 (A·B의 선행 의존성)

- 평가 노트북(`03_02`·`04_03`·`05_02`)의 `LogitsRunner`는 로짓을 `logits_{tag}_{split}.npy`로 캐시하지만, 캐시 경로가 Colab `/content/output/`(휘발)이라 **로컬** `output/`**에는 metrics JSON만 있고 로짓 캐시가 없다.**
- **val 로짓은 어떤 실험에서도 생성된 적이 없다**(평가는 test만 추론). B(임계값 튜닝)는 val 로짓이 필수.
- 조치: **추론 전용 Colab 잡 1회**로 3모델 × {test, val} 로짓 6개를 일괄 덤프하고 로컬 `output/`으로 다운로드한다.
  - 체크포인트(HF Hub): `ingyoun/kobert-patent-baseline` / `ingyoun/A.X-patent-maxlen512` / `ingyoun/A.X-patent-maxlen8192`.
  - tag(기존 규약 유지): `kobert-patent-baseline_len512` / `modernbert-patent-len512` / `modernbert-patent-len8192`.
  - ModernBERT 계열은 평가 노트북과 동일한 절단 규칙(훈련 `max_len`으로 `x[:max_len-1]+[eos]`)과 dtype 규칙(fp32 로드 + autocast bf16)을 적용한다 — `[modernbert.md](./modernbert.md)` 공통 프로토콜의 함정 두 가지가 그대로 적용된다.
  - `DataLoader(shuffle=False)`이므로 로짓 행 순서 = 데이터셋 행 순서. `document_id`로 라벨·길이축과 조인한다.
  - 추론만이라 저렴 — 가장 무거운 A.X-8192도 기존 평가 노트북 1회 실행 규모(test 11,271 + val 11,162).
- 파일 규약: `output/logits_{tag}_{split}.npy` (`split` ∈ {test, val}).
- 검증: 재덤프한 test 로짓으로 기존 지표를 재계산해 `output/total_metrics_{tag}.json`과 4자리 일치를 확인한 뒤 분석에 사용한다(체크포인트·절단·순서가 동일하다는 증거).



## A. 오류 분해 — sibling vs cross-Lno · 라벨 개수 bin (`notebook/06_01`)

**답할 질문**: 오분류가 같은 대분류 내 형제(`Mno`는 다르나 `Lno` 동일) 혼동인가, 대분류를 넘는 혼동인가. `PROJECT.md`의 분기점 — "형제 혼동이 지배적이면 계층 구조(대분류 조건부)로 확장" — 을 결정할 데이터를 만든다.

- **입력**: test 로짓 3종 + test 멀티핫 라벨 + `Mno`→`Lno` 매핑(라벨 인덱스↔`Mno` 매핑은 전처리 산출물 — `../data/data.md`). 훈련 불필요.
- **방법**:
  - **앵커(top-1) 오류 분해**: top-1 예측 `Mno`가 정답 라벨 집합에 없는 문서에서, 예측 `Mno`의 `Lno`가 정답 `Mno`들의 `Lno` 집합에 포함되면 **sibling**, 아니면 **cross-Lno**로 분류. 17×17 `Lno` 혼동 행렬 병행(어느 대분류 쌍이 새는지).
  - **멀티라벨(τ=0.5) 오류 분해**: FP 각각에 같은 기준 적용. FN도 병기(놓친 라벨의 `Lno`가 예측 `Lno` 집합에 있는가 — "대분류는 맞췄으나 중분류를 놓침" 신호).
  - **라벨 개수 bin**: 정답 라벨 수 1 vs ≥2로 micro/sample-F1·R-Precision 분해 — 비교 문서 「한계」에서 예고된 항목을 함께 해소한다.
- **판정 기준(사전 고정)**: exp1의 앵커 오류 중 sibling 비율이 **≥50%면 계층 확장 검토 개시, <50%면 flat 유지**. 3모델 공통 경향인지 교차 확인(판정은 exp1 기준, 나머지는 참고).
- **산출물**: `notebook/06_01_Lno_Analysis.ipynb`(로컬 CPU로 충분 — 로짓·라벨 연산뿐), 수치 `output/error_analysis_{tag}.json`, 해석은 비교 문서에 절 추가.
- **verify**: sibling + cross-Lno = 총 오류 수, 총 오류율이 1−P@1과 정합.



## B. 임계값 튜닝 — global τ · per-class τ (`notebook/06_02`)

**답할 질문**: τ=0.5가 버리는 성능이 얼마인가. 근거 신호는 empty rate 비단조(1.16→1.79→1.35%)와 threshold-free 랭킹 지표의 우위 — 비교 문서 「멀티라벨 지표 비교」가 "아직 안 쓴 성능 레버"로 지목한 항목.

- **프로토콜(사전 고정)**:
  - **val에서 튜닝, test에 1회 적용.** test로 τ를 고르지 않는다.
  - **3모델에 동일 정책을 적용**한다 — 한 모델에만 적용하면 모델 간 비교가 불공정해진다.
  - 기존 τ=0.5 수치는 프로토콜 앵커로 유지·병기한다(SSOT 수치 교체가 아니라 레버의 정량화).
- **방법**:
  1. **global τ**: grid(0.05~0.95 step 0.05, 최적 근방 0.01 세분)로 val micro/macro 각각 최대화.
  2. **per-class τ_c**: 클래스별 val F1 최대화. **과적합 가드**: val 양성 표본 수가 `n_min` 미만인 클래스는 global τ로 fallback(`n_min`은 val 내 소규모 sweep으로 선정, 시작값 30).
  3. **빈 예측 처리와의 조합**: "빈 예측 문서에 argmax 1개 강제" 규칙과 τ 정책의 결합 효과를 1회 비교.
- **리포트**: 모델별 {τ=0.5, global τ, per-class τ} × {micro, macro, sample-F1, empty rate} 매트릭스 + **val→test 일반화 격차**(val 이득 대비 test 이득 — per-class τ 과적합 진단).
- **기대·해석**: 주 목표는 macro 개선(희소 클래스의 낮게 깔린 확률 분포 보정). LRAP·R-Precision은 τ와 무관하므로 불변이어야 한다 — 파이프라인 sanity check로 사용.
- **산출물**: `notebook/06_02_Tau_Tuning.ipynb`(로컬 CPU로 충분), `output/threshold_{tag}.json`(정책·τ 벡터·지표), 해석은 비교 문서에 절 추가.
- **verify**: τ=0.5 재계산치가 기존 SSOT와 일치, 랭킹 지표 불변.



## C. 토크나이저 fertility·coverage 분석 (`notebook/06_03`) — 완료

**답한 질문**: "A.X가 같은 512 창에 더 많은 본문을 담는다"(모델 성분에 섞인 커버리지 성분의 기제)를 평균이 아닌 **분포**로 입증할 수 있는가. 아울러 다음 훈련(KLUE-RoBERTa-large@512 대조군)에서 같은 confound가 재발하지 않도록 해석 틀을 선확보한다.

- **의존성 0** — 로짓·GPU 불필요. 로컬 CPU에서 `ingyoun/patent-clean-text`와 토크나이저 3종만으로 실행(test 11,271 × 3토크나이저 ≈ 60초).
- **대상 토크나이저**: KoBERT(`monologg/kobert`, vocab 8,002) / A.X(`skt/A.X-Encoder-base` revision `9708f9c4`, vocab 49,999) / **KLUE-RoBERTa-large**(`klue/roberta-large` revision `28d91120`, vocab 32,000 — 다음 대조군 실험 대비 선측정).
- **입력 텍스트**: baseline과 동일 조합(`invention_title + ipc_main + abstract + claims` 공백 join, 빈 필드 skip). 범위는 **test split 11,271 기준**(비교 축과 정합).
- **측정 방법**: 토큰↔문자 정렬은 **어절(공백) 단위 토큰화**로 잡는다. `offset_mapping`은 KoBERT 토크나이저가 slow라 반환하지 않고, `decode` 길이는 WordPiece(A.X·KLUE)가 구두점 주위에 공백을 넣어 원문 대비 최대 +25% 부풀어(SentencePiece인 KoBERT는 ≈1.00) 검증 대상 주장 방향으로 편향된다. 어절별 조각 수 합이 문서 통째 토큰화와 정확히 일치함을 verify로 고정한다.
- **지표**:
  1. **fertility**: 문서별 토큰 수 ÷ 문자 수 분포, 토크나이저별. 같은 창에 담기는 콘텐츠 양의 비는 이 값의 비로 예측된다.
  2. **coverage@512**: 512 창에 조각이 전부 들어온 마지막 어절까지의 원문 문자 수 ÷ 전체 문자 수 — 문서별 분포와 `kobert_len` bin별 분해. 절대 격차와 **KoBERT 대비 상대이득**을 함께 본다(주장이 상대량이므로).
  3. **열화 신호**: 과분절 — 어절당 서브워드 조각 수 분포와 3+ 조각 어절 비율. `[UNK]` 비율은 병행 측정하되 **KoBERT에는 적용되지 않는 지표**다(SentencePiece라 UNK가 원리적으로 발생하지 않아 실측 0.0000%). vocab 8,002의 표현 손실은 UNK가 아니라 과분절로 나타난다.
- **산출물**: `notebook/06_03_Tokenizer_Analysis.ipynb`(로컬 `.venv` 커널), 수치 SSOT `output/tokenizer_analysis.json`. 해석은 `[modernbert-comparison.md](./modernbert-comparison.md)` 「3-모델 bin 비교」, KLUE 선측정은 `../data/data.md` 「KLUE-RoBERTa 토크나이저」에 반입했다.
- **verify(통과)**: 어절별 조각 수 합 == 문서 통째 토큰 수(3종 전부) · KoBERT `seq_len` == 데이터셋 `kobert_len`(SSOT) 11,271/11,271 일치 · A.X−KoBERT 길이 차가 `../data/data.md` 기존 실측(mean +88.4 / median +64 / p90 +185 / p99 +517 / min −1,463 / max +2,287)과 완전 일치. 상류 전처리(`02_02`·`04_01`)와 독립 경로로 계산돼 재현 증거가 된다.

**결과 요약** (수치는 `output/tokenizer_analysis.json`):

- **커버리지 우위는 ~10%다** — 절단 문서(8,074건) 한정 coverage 상대이득 **+10.0%**. fertility 비(KoBERT 0.6045 ÷ A.X 0.5449 = 1.109)가 예측하는 +10.9%와 정합한다. test 전체 평균으로는 +6.0%이나, 이득이 원천적으로 불가능한 B0(28.4%)이 희석한 값이라 기제 논의에는 절단 문서 기준을 쓴다.
- **B0에서 우위는 정확히 0** — 절단이 없으면 커버리지 우위가 무의미하다는 예측의 확증(Δ −0.0002).
- **과분절이 유일한 열화 신호** — KoBERT 어절당 2.778조각 / 3+조각 43.15% / 1조각 어절 21.06%, A.X 2.503 / 37.81% / 30.42%. A.X가 어절을 통째로 담는 비율이 1.44배다.



## 실행 순서

1. **C — 완료.** 로짓 무관·로컬 실행. 결과는 위 「결과 요약」·`output/tokenizer_analysis.json`.
2. **A**(test 로짓 확보 후) — 계층 확장 분기 판정이 다음 아키텍처 결정을 좌우하므로 로짓 도착 후 최우선.
3. **B**(val 로짓 확보 후) — τ 정책 수립. A와 같은 로짓을 쓰므로 같은 잡의 산출물로 이어서 진행.

세 분석 모두 완료 시점에 다음 훈련 1회(KLUE-RoBERTa-large@512, `PROJECT.md` 대조군)의 해석 틀 — 오류 구조 기준선, 공정한 τ 정책, 토크나이저 커버리지 보정 — 이 갖춰진다.