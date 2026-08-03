# 프로젝트 문서 (`docs/`)

특허 과학기술표준분류 분류(`[PROJECT.md](../PROJECT.md)`, SSOT) 진행 중 **참고·정리용 문서**를 모으는 곳.
인프라 운영뿐 아니라 데이터 분석 노트, 실험 결과, 의사결정 기록 등을 여기에 쌓는다.
(사람이 쓰든 Claude가 쓰든) 재사용할 만한 내용은 대화에 묻지 말고 이 폴더에 문서로 남긴다.

## 폴더 구조

문서가 늘어남에 따라 범주별 하위 폴더로 분류한다. 인덱스(이 파일)는 루트에 유지.

- `infra/` — GPU 훈련을 외부 잡으로 돌리는 방법(로컬은 CPU 전용)
- `data/` — 데이터 레이아웃·스키마·전처리 파이프라인
- `experiments/` — baseline 재현·실험별 설정/결과·프로토콜
- `adr/` — 사안별 결정 기록·회고(결정의 경위·대안·결과)

## 현재 문서

### 인프라 운영 (`infra/`)

GPU 훈련을 외부 잡으로 돌리는 방법 — 이 세션은 **로컬 Windows(CPU 전용)** 이라 훈련은 외부에서 한다.

- `[infra/lightning-jobs.md](./infra/lightning-jobs.md)` — Lightning AI Jobs: **로컬에서 Python SDK로 커스텀 Docker 이미지 잡을 제출**(스튜디오 스냅샷 대신). `lightning` CLI는 Windows 미지원 → SDK 사용
- `[infra/colab-jobs.md](./infra/colab-jobs.md)` — Google Colab: `colab` CLI로 헤드리스 L4 GPU에서 self-contained 스크립트 실행
- `[infra/runpod-jobs.md](./infra/runpod-jobs.md)` — RunPod: `uv.lock`을 굳힌 커스텀 Docker 이미지로 RTX 4090/L4 팟에서 `.ipynb` 훈련(Trainer 런타임 의존성·flash-attn gpu 그룹·비용 가드)
- `[infra/studio-performance.md](./infra/studio-performance.md)` — ⚠️ **과거 Lightning cloudspace 기록**(로컬 무관): 스튜디오 느림/"CPU 과부하" 진단, 컨테이너 vs 호스트 CPU 구분, 노이지 네이버 대처

### 데이터 (`data/`)

- `[data/data.md](./data/data.md)` — AI Hub 71531 레이아웃, JSON 스키마, 원천↔라벨 조인, 17대분류/188중분류, 다중레이블·분포·baseline 주의 + 다중레이블의 계층 형상(cross-`Lno` vs within-`Lno`, 계층 설계 제약 — `scripts/multilabel_shape.py`·`output/multilabel_shape.json`) + IPC 필드 관계(`ipc_main`은 `ipc_all`의 첫 코드 · 보조 코드의 잔여 신호는 잡음 수준이며 부호가 다중레이블 반대 — `scripts/ipc_field_analysis.py`·`output/ipc_field_analysis.json`)
- `[data/data-pipeline.md](./data/data-pipeline.md)` — 2계층 파이프라인: zip 스트리밍 파싱 → 정제 텍스트 HF Hub push → 소비 시 토큰화

### 실험·기록 (`experiments/`)

- `[experiments/kobert-baseline.md](./experiments/kobert-baseline.md)` — KoBERT baseline을 고정 test set 위에서 직접 재현해 **비교 기준점**(top-1 weighted-F1 + 멀티라벨·길이구간) 수립
- `[experiments/modernbert.md](./experiments/modernbert.md)` — A.X-Encoder(ModernBERT) 실험 **계획·프로토콜·비교 축**(허브: 공통 프로토콜·dtype/절단 함정·실험 목록)
- `[experiments/modernbert-results.md](./experiments/modernbert-results.md)` — A.X-Encoder **실험별 실측**(exp1 full 8192 / exp2 512 control)
- `[experiments/modernbert-comparison.md](./experiments/modernbert-comparison.md)` — A.X-Encoder **교차 비교·결론**(길이 vs 모델 분해 · 오류 수준 차집합 · 3모델 bin · 커버리지 기제 실측 · 멀티라벨 지표 비교 · 라벨 개수 bin · 오류 구조와 계층 확장 판정 — 2단계 추정량의 결함·조건부 재추정 포함(`scripts/hierarchy_conditional.py` · `output/hierarchy_conditional.json`·`output/hierarchy_stage1_rules.json`) · 임계값 정책)
- `[experiments/klue-roberta.md](./experiments/klue-roberta.md)` — **KLUE-RoBERTa-base 대조군(선택 항목 — 후순위)** 계획·프로토콜(512 창 절단 필수 · 절단 규칙 · 크기·토크나이저 confound)
- `[experiments/no-train-analysis.md](./experiments/no-train-analysis.md)` — **무훈련 분석 3종 완료**(오류 분해 sibling/cross-Lno · 임계값 튜닝 global/per-class τ · 토크나이저 fertility·coverage) + 로짓 재확보 절차
- `[experiments/loss-function.md](./experiments/loss-function.md)` — **손실 함수 축**(카디널리티 회수 겨냥): 후보 평가·프로토콜 + ZLPR·ASL·BCE 실측(전부 focal 대비 열세 → 축 종결, [ADR-0009](./adr/0009-loss-axis-closure.md))
- `[experiments/cardinality-decoding.md](./experiments/cardinality-decoding.md)` — **카디널리티 디코딩**(문서별 기대-F1 plug-in): 손실 종결 후 남은 카디널리티 헤드룸을 추론 결정 규칙으로 회수 시도 → **음성**(raw에 신호 존재·k=1 vs k≥2 분리 AUC 0.89이나 운영점 비대칭으로 k=1 과대예측과 분리 불가, micro 후퇴). 사후 디코딩·IPC·손실 세 각도 기준 오라클-k +1.60pt는 도달 불가 상한으로 확정, 공동학습 k-head만 미검증 갈래로 보류(`notebook/10_01`)
- `[experiments/hierarchy-loss.md](./experiments/hierarchy-loss.md)` — **계층 손실(MCLoss) 축**(cross-`Lno` 오류를 훈련 신호로): `PROJECT.md` 계층 행이 열어 둔 "게이트 없는 형태"의 첫 훈련 실측. 추론 구조 불변(flat + `Mno`→`Lno` 유도 = MCM), 손실에만 17개 `Lno` 그룹 항 추가. 표적 질량 실측 — 정답 `Lno` 밖 FP 60.0~60.4% · 그룹 전체 놓친 FN 76.5~78.4%(4모델 공통)이고 음성 항은 k=1(과대예측)·양성 항은 k≥2(과소예측)에 실린다. **`14_01` 1런 실측으로 축 종결**([ADR-0014](./adr/0014-hierarchy-loss-closure.md)) — 정리 test micro 0.8467 = 앵커 −1.20pt(paired bootstrap CI95 [−1.60, −0.80], 잡음 밖)이고 `Lno` 유도 축도 평탄하다. 작동점을 맞추면 음성 항 표적만 −63 움직이고 그 대가를 within `Lno` FP·FN이 상계한다(구성은 이동, 총량은 불변). **실패 기제**는 표적 부족이 아니라 전달 실패 — λ를 초기화 손실 **값**으로 고정해 기울기 비가 0.35x(초기화) → 2.12x(종점, 그룹 항 몫 67.9%)로 뒤집히고 arm이 focal 자체에서도 나쁜 지점에 도착하며, `max` 정의상 형제 밴드가 두 항 어디에도 안 닿아 조건부 형제 top-1이 −0.53pt로 잡음 밖 하락한다. λ 변형(훈련 곡선 24지점 중 23지점 음수 = 대리 실험)·BCE 기반 교체(그룹 몫 74.9~80.3%로 악화)도 상한이 판정선 아래로 산술 확정(`scripts/hierarchy_loss_grad_budget.py` · `output/hierarchy_loss_grad_budget.json`). 관찰 지표를 share로 두면 방향이 뒤집혀 보이는 결함과 작동점 정규화 절차도 이 문서가 소유한다(`scripts/hierarchy_loss_mass.py` · `output/hierarchy_loss_mass.json`). 근거 문헌 C-HMCNN(NeurIPS 2020). **「문헌 대조」 절이 계열 전체의 기대값을 소유한다** — 깊이 2 · 노드 다수 · 훈련 표본 다수 · 불균형 없음은 계층 이득이 사라지는 조건의 조합이고(CoNLL 2024에서 flat BERT+BCE 대비 최고 계층 모델의 격차 +0.19pt), 계층 손실 계열이 개선하는 것은 오류 수가 아니라 심각도라(CHAMP) flat leaf micro-F1에 등록되지 않는다. 잔여 계열(조건부 시그모이드·계층 인지 표현)의 기대값과 λ 규칙이 알려진 안티패턴인 근거도 이 절에 있다
- `[experiments/longdoc-degradation.md](./experiments/longdoc-degradation.md)` — **장문 열화 진단**(표현 vs 결정, label-aware attention 게이트): 무훈련 로짓 게이트로 저하가 카디널리티와 독립이나 표현 붕괴가 아님(최장 문서도 정답 top-5 ~98% 잔존)을 보여 풀링 헤드룸을 <~0.5pt로 경계 → label-aware attention은 기대 이득 얇은 레버로 디프리오리티(`notebook/10_02`)
- `[experiments/knowledge-distillation.md](./experiments/knowledge-distillation.md)` — **지식 증류(KD) 축 — 종결**([ADR-0015](./adr/0015-kd-closure.md), 이종 앙상블 → 단일 2048 student 설계): 앙상블 헤드룸을 앙상블 배포 없이 단일 모델로 회수하려던 경로. 무훈련 헤드룸 게이트는 **GREEN**(정리 test 앙상블 micro +0.73pt·k≥2 +1.42pt, `output/kd_gate_ensemble.json`)이었으나, 주 비용 앞에 집행한 **착수 전 게이트 둘**이 축을 닫았다 — ① **전이 구조**(`output/kd_transfer_structure.json`·`scripts/kd_transfer_structure.py`): 이득은 새 정보이나(exp1 재조정의 도달 불가 오라클 +0.13pt) 순 182개 원소가 임계 근처에 몰려 있고(여유 0.2 이상만 채택 시 +0.01pt), 승자가 길이·k·확신도로 예측되지 않으며(oracle gap 25.11pt 중 0.4~2.1% 회수), 104/188 클래스에 흩어져 있다. ② **기울기 예산**(`output/kd_grad_budget.json`·`scripts/kd_grad_budget.py`): λ=0.5가 종점 기울기의 93%를 distill 항에 주면서 이득 밴드에는 1.2%만 쓰고, 이 쏠림은 λ로 고쳐지지 않는다(설계 λ는 0.017~0.065). 온도 T는 dark knowledge와 예산 포획을 함께 키운다. teacher 포화 실측(exp1 soft target이 문서당 중간대 0.71개·앙상블 2.01개)도 이 문서가 소유한다([training-curves.md](./experiments/training-curves.md))
- `[experiments/domain-pretraining.md](./experiments/domain-pretraining.md)` — **도메인 사전학습(TAPT) 축**(자체 특허 코퍼스 MLM 계속학습 → 분류 파인튜닝): MLM 5 epoch 완주 실측(val loss 0.4272→0.3742, 약 802M 토큰, Colab L4 ~14h)과 아티팩트 검증 절차(`ingyoun/A.X-patent-tapt-mlm@62818c2`, 토크나이저 동일성·헤드 초기화 대칭). **분류 실측 기준 미달로 축 종결**([ADR-0013](./adr/0013-domain-pretraining-closure.md)) — 정리 test micro 0.8572로 앵커 `11_01`(0.8588) 대비 −0.15pt, 판정선 +0.4pt 미달. 훈련 곡선의 초기 우위도 잡음(3 epoch까지 평균 +0.04pt, eval별 델타 sd 0.43pt). 실패 기제는 **코퍼스 동일성**(TAPT 802M 토큰 대 분류 파인튜닝 1,116M 토큰을 같은 문서에서 본다). MLM 체크포인트 교체·`KorPatElectra` 재개 모두 불채택 근거 포함. ⚠️ `13_02` 로짓 덤프는 행 순열로 깨져 폐기(팟 `src` 미동기화로 행 순서 방어가 빠진 구 사본이 돌았다)
- `[experiments/eval-noise.md](./experiments/eval-noise.md)` — **잡음 하한 두 성분**(평가 표본 + 훈련 시드): ① 고정 test paired bootstrap(GPU 0)으로 정리 test 11,244에서 micro 델타의 표본 잡음 sd 0.18~0.21pt(`scripts/eval_noise_bootstrap.py` · `output/eval_noise_bootstrap.json`). ② `11_01` 시드 재현(`11_04`, seed 42 대 153, 시드 외 전 설정 동일)으로 훈련 잡음 측정 — 시드 델타 micro −0.18pt가 구간 [−0.55, +0.21]로 0을 포함하고 `D² < σ_표본²`이라 **훈련 잡음이 평가 잡음 바닥 아래로 분해되지 않는다**(관측 |Δ| 0.176pt ≈ 순수 표본 기대 0.157pt). 따라서 표본 구간을 넓힐 근거가 없어 ①의 판정이 그대로 선다. 시드 스프레드는 epoch 1–2의 2.1pt에서 epoch 11–12의 0.12pt로 수렴 — 부분 학습 곡선으로 설정을 비교하면 안 되는 근거(`scripts/seed_variance.py` · `output/seed_variance.json`). ③ 두 번째 시드로 손실 축 재측정 — **focal−BCE가 +0.52pt(유의) → +0.34pt(CI [−0.05, +0.74], 미유의)로 뒤집혀** γ 이득이 미확정으로 강등, ZLPR·ASL·길이·모델 성분은 여유가 넓어 불변. 운영 판정선 **micro 델타 0.6pt 미만은 시드 1런씩으로 확정하지 않는다**. ⚠️ 자유도 1이라 "σ_훈련=0"이 아니라 "이 test로는 안 보인다"이며 다른 레시피로 옮길 근거 없음
- `[experiments/training-curves.md](./experiments/training-curves.md)` — **훈련 곡선 판독**(4런 공통: val loss 3~7 epoch 최저 후 상승, F1은 끝까지 개선 → 정상 형상): 손실 질량 분해로 상승의 정체 규정(원소 0.1%가 손실의 75~88% · 확신 오답 1,500~2,000개가 73~82%), τ 아티팩트·val 과적합 해석 배제, best 선정·early stop을 손실 아닌 주 지표에 거는 근거와 감시 대조점. 대가는 확률 포화(양성의 70~77%가 p≥0.9)로 KD soft target에 전이(`scripts/loss_mass_decomposition.py` · `output/loss_mass_decomposition.json`)
- `[experiments/confident-errors.md](./experiments/confident-errors.md)` — **확신 오답 진단**(손실 질량 73~82%를 무는 원소 1,461개의 정체 — [ADR-0016](./adr/0016-confident-error-diagnosis.md)): 교차 모델 합의로 **모델 고유 실패를 배제**한다(시드 쌍둥이 86.3% → 다른 아키텍처 KoBERT 81.7%, 앵커 단독 1.6%. 대조군인 경계 오답은 전원 동조 21.1%·단독 14.3%). 공기 검정은 확신 FP를 **라벨 누락 후보**로 규정하고(전원 동조분 lift 6.62 ≈ 입력 동일 라벨 충돌 실측 5.56, 앵커 단독 0.52), 확신 FN을 **문서에서 겉도는 이질 라벨**로 규정한다(같은 k≥2 모집단에서 3.57 대 맞힌 정답 19.11). **"focal이 오답 확신을 키운다"는 반증된다** — γ만 뺀 BCE가 확신 오답을 1.75배 만들고 앵커 집합의 75.3%가 BCE에서도 확신 오답이다(손실은 질량을 집중시킬 뿐 집합을 만들지 않는다). 잡음 후보가 전부 실제 잡음이면 측정 micro가 `11_01` +2.14pt · exp1 +1.80pt 하향 편향(가정적 값 — 정답 재작성 금지). 두 극단의 상세 분류(G절, `scripts/confident_error_classes.py` · `output/confident_error_classes.json`)는 **잡음의 위치가 클래스가 아니라 문서**임을 보인다 — 전원 동조 FP 419개가 329개 쌍에 흩어지고(상위 10쌍 10.7%), 라벨 충돌 실측 74클래스와의 겹침도 관측/기대 1.08이며, **앵커 단독 24개는 시드 쌍둥이가 100% 맞힌다**(모델 귀속분 사실상 0). 재라벨링 검토 후보 884행·752문서를 `output/relabel_candidates.csv`로 산출(반영하지 않음). **성능 보고에 실을 한계 기술 문안은 「보고용 요약」 절이 소유**한다. ⚠️ 공기 검정의 confound(모델이 학습한 공기 구조와 미분리)·k=1 확신 FN 364개 미분류·클래스 축 검정의 낮은 검정력이 이 진단의 한계(`scripts/confident_error_diagnosis.py` · `output/confident_error_diagnosis.json`)
- `[experiments/final-run.md](./experiments/final-run.md)` — **배포 모델 확정 런**(`16_01`, max_len 4096)의 결정·근거: 창 크기 선택(유실 토큰 1.60% 대 `micro_batch` 2배 — `scripts/length_cost.py`·`output/length_cost.json`), LR range test 판정(전환점 512 8.84e-3 · 8192 7.13e-3, 운영 lr이 15~18배 아래 — `output/lr_range_verdict.json`), lr이 `eff_batch`만 따르고 `micro_batch`와 무관한 이유, 착수하지 않는 것(512 HPO·스케줄러 교체·토큰 예산 배칭), 실행 후 할 일
- (예정) 데이터 EDA/필드 길이 분석, 클래스 불균형 노트, 실험별 설정·결과 요약, 모델/입력 필드 선택 근거

### 분석 스크립트 (`scripts/`) — 전부 GPU 0, 덤프된 로짓·라벨만 쓴다

각 스크립트는 `output/<같은 이름>.json`에 수치를 남기고, 위 실험 문서가 그 해석을 소유한다. 새 런(arm)을 얹을 때는 대개 상단 `MODELS`/`TAGS` 사전 한 줄만 고치면 된다.

| 스크립트 | 재사용 가능한 도구 |
| --- | --- |
| `hierarchy_loss_mass.py` | `paired_bootstrap`(문서 단위 CI) · `matched_operating_point`(작동점 정규화 — 예측량이 다른 두 런의 오류 칸을 τ=0.5에서 직접 비교하지 않기 위한 것) |
| `hierarchy_loss_grad_budget.py` | 손실 항별 기울기 예산 · 포화도 · 순위 국소화(P@1을 `Lno` 축과 조건부 형제 축으로 분해) |
| `kd_transfer_structure.py` | 여유 밴드 분해 · 재조정 도달(캘리브레이션 배제) · 라우팅 전이(val→test) · 집중도(관측/기대) |
| `kd_grad_budget.py` | 손실 항별 기울기 몫 · 축퇴도 · 신호 국소화 · 설계 λ 역산 |
| `confident_error_diagnosis.py` | 교차 모델 합의(대조군 포함) · train 공기(lift) 검정 · 손실 질량 분해 · 측정 편향 |
| `confident_error_classes.py` | 클래스 쌍·라벨 축 분해 · 시드 쌍둥이 대조 · 재라벨링 후보 목록(CSV) |
| `loss_mass_decomposition.py` | 원소별 focal 질량 집중도 · 확률 포화도 · τ 스윕 |
| `eval_noise_bootstrap.py` · `seed_variance.py` | 평가 표본 잡음 · 훈련 시드 잡음(판정선의 근거) |
| `length_cost.py` | 입력 길이 분포 · `max_len`별 유실/비용/`micro_batch` 여유 |
| `hierarchy_conditional.py` · `multilabel_shape.py` · `ipc_field_analysis.py` | 계층 조건부 추정량 · 다중레이블 형상 · IPC 필드 신호 |

### 결정 기록·회고 (`adr/`)

프로젝트의 주요 결정을 사안별로 남긴다 — 무엇을 왜 결정했고 어떤 대안을 접었으며 결과가 무엇이었나. 포트폴리오·회고의 SSOT.

- `[adr/README.md](./adr/README.md)` — ADR 인덱스·규약·결정 카탈로그(사안 목록·상태)

## 훈련 인프라 요약

문서당 다중 레이블(multi-label) 188-way 인코더 분류(주 모델 `skt/A.X-Encoder-base` — 한국어 ModernBERT 16k / 대조군 KLUE-RoBERTa-base 512, 선택 항목). 단일 GPU로 충분, multi-GPU 불필요.


| 경로                     | 언제                                                                                                         |
| ---------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Colab L4 (24GB)**    | 기본 경로. 장문(최대 16,384 토큰) `skt/A.X-Encoder-base`는 512 토큰 대비 메모리를 많이 써 24GB가 안전. `colab exec -f nb.ipynb`로 실행 |
| **Lightning Job**      | 로컬에서 SDK로 **커스텀 Docker 이미지** 잡 제출. L4보다 큰 머신(A100/H100)·긴 런타임·스팟이 필요할 때. Job 실행 시간만 과금                     |
| **로컬 Windows(CPU)**     | 데이터 전처리·토큰화·통계·잡 제출/제어 등 GPU 불필요 작업. **GPU 훈련 불가.**                                                       |


> ⚠️ **idle GPU는 계속 과금된다.** Colab은 끝나면 반드시 `colab stop`(또는 self-clean되는 `colab run`), Lightning Job은 종료 시 머신이 자동 회수.

## 환경 값 (로컬 + Lightning)


| 항목              | 값                                                                                        |
| --------------- | ---------------------------------------------------------------------------------------- |
| 로컬 머신           | Windows 11, **GPU 없음(CPU 전용)**. uv 프로젝트 `.venv`(Python 3.12), 인터프리터 `.venv\Scripts\python.exe` |
| 로컬 도구           | `uv` 설치. **Docker 미설치**(이미지 빌드 시 필요). `lightning` CLI는 Windows 미지원 → **Lightning은 SDK로** |
| org             | `paraise-org`                                                                            |
| teamspace       | `ml`                                                                                     |
| teamspace owner | `paraise-org` (org)                                                                      |
| 로컬 인증 계정        | `paraise-edu` — teamspace `ml`의 **멤버**(owner 아님)                                          |
| 인증 env          | `LIGHTNING_USER_ID`·`LIGHTNING_API_KEY`(잡 제출), `HUGGINGFACEHUB_API_TOKEN`(데이터). `.env`에 저장 |


> ⚠️ 로그인 `paraise-edu`는 owner가 아니다 → 로컬에서 잡 제출 시 owner를 명시(SDK `Teamspace(name="ml", org="paraise-org")`). 상세는 `[infra/lightning-jobs.md](./infra/lightning-jobs.md)` 트러블슈팅.

