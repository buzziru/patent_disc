# 프로젝트 문서 (`docs/`)

특허 문헌을 과학기술표준분류 **17개 대분류 / 188개 중분류**로 자동 분류하는 인코더 모델을 만들고, 공식 baseline 대비 개선을 정량 입증하는 프로젝트다. 한 특허가 여러 중분류에 해당할 수 있어(고유 문서의 14.1%) 문제는 다중 라벨 188-way 분류이며, 주 모델은 한국어 ModernBERT인 `skt/A.X-Encoder-base`(16k 컨텍스트)다.

**현재 결과**: 자체 재현한 KoBERT baseline의 다중 라벨 micro-F1 **0.8502** → 장문 인코더(8192 토큰) **0.8685**(+1.83pt). 개선분은 컨텍스트 길이 +0.84pt와 모델·토크나이저 +0.99pt로 분해된다. **산출물 모델은 비용 대비 손익 분기인 4096 토큰 창으로 정리 데이터 위에서 훈련한 `16_01`이며, 정리 test micro-F1 0.8660**(KoBERT 재현선 대비 +1.60pt · 512 기준 런 대비 +0.72pt)이다 — [experiments/final-run.md](./experiments/final-run.md).

이 폴더는 그 과정에서 나온 **데이터 분석·실험 실측·의사결정 기록**을 모은 곳이다. 전체 명세(목표·접근·평가 프로토콜)는 [PROJECT.md](../PROJECT.md)에 있다.

## 처음 읽는다면

1. [GLOSSARY.md](./GLOSSARY.md) — 기호·용어·런 코드 대조표. 나머지 문서가 전부 이 약속 위에 있다.
2. [PROJECT.md](../PROJECT.md) — 무엇을 왜 만드는가.
3. [data/data.md](./data/data.md) — 데이터가 어떻게 생겼고 어떤 함정이 있는가.
4. [experiments/kobert-baseline.md](./experiments/kobert-baseline.md) — 비교선을 어떻게 세웠는가.
5. [experiments/modernbert-comparison.md](./experiments/modernbert-comparison.md) — 개선이 어디서 왔는가.
6. [adr/README.md](./adr/README.md) — 무엇을 결정했고 무엇을 접었는가.

## 문서 지도

### 데이터 (`data/`)

| 문서 | 내용 |
| --- | --- |
| [data.md](./data/data.md) | 레이아웃·스키마·라벨 분포·분할 누수·데이터 클리닝 |
| [data-pipeline.md](./data/data-pipeline.md) | zip 스트리밍 파싱 → HF Hub 업로드 → 소비 시 토큰화 |

### 실험 (`experiments/`)

| 문서 | 내용 |
| --- | --- |
| [kobert-baseline.md](./experiments/kobert-baseline.md) | KoBERT baseline 자체 재현 — 비교 기준선 수립 |
| [modernbert.md](./experiments/modernbert.md) | 장문 인코더 실험의 계획·공통 프로토콜·함정 |
| [modernbert-results.md](./experiments/modernbert-results.md) | 실험별 실측치(8192 / 512 대조) |
| [modernbert-comparison.md](./experiments/modernbert-comparison.md) | 교차 비교와 결론 — 길이 대 모델 분해, 오류 구조 |
| [klue-roberta.md](./experiments/klue-roberta.md) | KLUE-RoBERTa 대조군 계획(후순위) |
| [no-train-analysis.md](./experiments/no-train-analysis.md) | 저장된 로짓만으로 한 세 가지 무훈련 분석 |
| [loss-function.md](./experiments/loss-function.md) | 손실 함수 축 — ZLPR·ASL·BCE 실측과 종결 |
| [cardinality-decoding.md](./experiments/cardinality-decoding.md) | 추론 시 예측 개수 결정 규칙 — 음성 결과 |
| [hierarchy-loss.md](./experiments/hierarchy-loss.md) | 계층 손실(MCLoss) 축 — 실측과 실패 기제 |
| [longdoc-degradation.md](./experiments/longdoc-degradation.md) | 긴 문서에서 성능이 떨어지는 원인 진단 |
| [knowledge-distillation.md](./experiments/knowledge-distillation.md) | 지식 증류 축 — 착수 전 게이트로 종결 |
| [domain-pretraining.md](./experiments/domain-pretraining.md) | 특허 코퍼스 사전학습(TAPT) — 실측과 종결 |
| [eval-noise.md](./experiments/eval-noise.md) | 평가 표본 잡음과 훈련 시드 잡음 — 판정선의 근거 |
| [training-curves.md](./experiments/training-curves.md) | 훈련 곡선 판독 — 손실 상승과 F1 개선의 공존 |
| [confident-errors.md](./experiments/confident-errors.md) | 확신 오답의 정체 — 라벨 잡음 진단과 측정 편향 |
| [final-run.md](./experiments/final-run.md) | 배포 모델 확정 런 — 창 크기·레시피 결정과 최종 실측 |

### 결정 기록 (`adr/`)

무엇을 왜 결정했고, 어떤 대안을 왜 접었으며, 결과가 무엇이었는지를 사안별로 남긴다. 목록과 상태는 [adr/README.md](./adr/README.md)의 카탈로그에 있다.

### 인프라 (`infra/`)

로컬 머신은 **Windows CPU 전용**이라 GPU 훈련을 외부 잡으로 돌린다. 그 운영 기록이다.

| 문서 | 내용 |
| --- | --- |
| [colab-jobs.md](./infra/colab-jobs.md) | Google Colab — `colab` CLI로 헤드리스 L4 실행 |
| [lightning-jobs.md](./infra/lightning-jobs.md) | Lightning AI — Python SDK로 커스텀 Docker 이미지 잡 제출 |
| [runpod-jobs.md](./infra/runpod-jobs.md) | RunPod — 커스텀 이미지로 RTX 4090/L4 팟 훈련 |
| [studio-performance.md](./infra/studio-performance.md) | 과거 Lightning cloudspace 진단 기록(현재 작업과 무관) |

## 분석 스크립트 (`scripts/`)

전부 GPU 없이 돌아간다 — 저장된 로짓과 라벨만 읽는다. 각 스크립트는 `output/<같은 이름>.json`에 수치를 남기고, 그 해석은 위 실험 문서가 담는다. 새 런을 얹을 때는 대개 상단 `MODELS`/`TAGS` 사전 한 줄만 고치면 된다.

| 스크립트 | 재사용 가능한 도구 |
| --- | --- |
| `hierarchy_loss_mass.py` | `paired_bootstrap`(문서 단위 신뢰구간) · `matched_operating_point`(예측량이 다른 두 런의 오류를 공정하게 비교하기 위한 작동점 정규화) |
| `hierarchy_loss_grad_budget.py` | 손실 항별 기울기 예산 · 포화도 · 순위 국소화 |
| `kd_transfer_structure.py` | 여유 밴드 분해 · 재조정 도달 상한 · 라우팅 전이 · 집중도 |
| `kd_grad_budget.py` | 손실 항별 기울기 몫 · 축퇴도 · 신호 국소화 · 설계 λ 역산 |
| `confident_error_diagnosis.py` | 교차 모델 합의 · 공기 검정 · 손실 질량 분해 · 측정 편향 |
| `confident_error_classes.py` | 클래스 쌍 분해 · 시드 쌍둥이 대조 · 재라벨링 후보 목록(CSV) |
| `loss_mass_decomposition.py` | 원소별 손실 질량 집중도 · 확률 포화도 · `τ` 스윕 |
| `eval_noise_bootstrap.py` · `seed_variance.py` | 평가 표본 잡음 · 훈련 시드 잡음 |
| `error_analysis_final.py` | 배포 모델 상세 오류 분석 — `error_analysis` 기법 전량 + 비교선 4런 paired 대조·길이 bin 델타 |
| `length_cost.py` | 입력 길이 분포 · `max_len`별 유실 토큰·비용·배치 여유 |
| `hierarchy_conditional.py` · `multilabel_shape.py` · `ipc_field_analysis.py` | 계층 조건부 추정량 · 다중 라벨 형상 · IPC 필드 신호 |

## 훈련 인프라 요약

단일 GPU로 충분하며 multi-GPU는 필요 없다.

| 경로 | 언제 쓰나 |
| --- | --- |
| **Colab L4 (24GB)** | 기본 경로. 장문 모델은 512 토큰 대비 메모리를 많이 써 24GB가 안전하다. `colab exec -f nb.ipynb`로 실행 |
| **Lightning Job** | 로컬에서 SDK로 커스텀 Docker 이미지 잡을 제출한다. L4보다 큰 머신·긴 런타임·스팟이 필요할 때. 실행 시간만 과금 |
| **RunPod** | `uv.lock`을 굳힌 이미지로 RTX 4090/L4 팟에서 노트북 훈련 |
| **로컬 Windows(CPU)** | 데이터 전처리·통계·잡 제어 등 GPU가 필요 없는 작업. GPU 훈련은 불가 |

> ⚠️ **idle GPU는 계속 과금된다.** Colab은 끝나면 반드시 `colab stop`(또는 self-clean되는 `colab run`)을 부른다. Lightning Job은 종료 시 머신이 자동 회수된다.

## 환경 값

| 항목 | 값 |
| --- | --- |
| 로컬 머신 | Windows 11, GPU 없음(CPU 전용). uv 프로젝트 `.venv`(Python 3.12), 인터프리터 `.venv\Scripts\python.exe` |
| 로컬 도구 | `uv` 설치. Docker 미설치(이미지 빌드 시 필요). `lightning` CLI는 Windows 미지원이라 Lightning은 SDK로 쓴다 |
| org | `paraise-org` |
| teamspace | `ml` (owner: `paraise-org`) |
| 로컬 인증 계정 | `paraise-edu` — teamspace `ml`의 멤버(owner 아님) |
| 인증 env | `LIGHTNING_USER_ID`·`LIGHTNING_API_KEY`(잡 제출), `HUGGINGFACEHUB_API_TOKEN`(데이터). `.env`에 저장 |

로그인 계정이 owner가 아니므로 로컬에서 잡을 제출할 때는 owner를 명시해야 한다(`Teamspace(name="ml", org="paraise-org")`). 상세는 [infra/lightning-jobs.md](./infra/lightning-jobs.md)의 트러블슈팅 절에 있다.

## 문서를 쌓는 규칙

- 재사용할 지식·결과는 대화가 아니라 이 폴더에 문서로 남긴다.
- 인프라를 운영하며 알게 된 사실(오류·해결·플래그·머신 동작)은 해당 `infra/*.md`를 즉시 갱신한다.
- 서술은 3인칭·사실 중심으로 하고, 수정 이력은 문서에 남기지 않는다(상세는 `CLAUDE.md` 「문서 서술 규칙」).
