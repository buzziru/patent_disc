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
| [data-pipeline.md](./data/data-pipeline.md) | zip 스트리밍 파싱 → HF Hub 업로드 → 소비 시 토큰화 · **AI Hub 원본에서 데이터셋을 재생성하는 절차** |

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

로컬 머신은 **Windows CPU 전용**이라 GPU 훈련을 외부에서 돌린다. 그 운영 기록이다.

**훈련은 두 단계를 거쳤다** — 코드가 확정되기 전의 초기 실험·디버깅은 **Colab**에서, 훈련 코드를 `src/patent_train` 패키지로 굳힌 뒤의 장시간 런은 전부 **RunPod 팟**에서 돌았다. 전환 이유는 Colab 헤드리스 세션이 ~20분에 회수돼 무인 장시간 훈련이 되지 않기 때문이다.

| 문서 | 내용 | 사용 |
| --- | --- | --- |
| [runpod-jobs.md](./infra/runpod-jobs.md) | RunPod — 로컬에서 빌드한 이미지로 팟 템플릿을 만들고 볼륨에 코드·캐시를 두고 훈련 | **주 경로** |
| [colab-jobs.md](./infra/colab-jobs.md) | Google Colab — `colab` CLI로 헤드리스 L4 실행 | 초기 실험·디버깅 |
| [lightning-jobs.md](./infra/lightning-jobs.md) | Lightning AI — Python SDK로 커스텀 Docker 이미지 잡 제출 | 미사용(검증 기록) |
| [studio-performance.md](./infra/studio-performance.md) | 과거 Lightning cloudspace 진단 기록 | 미사용(참고) |

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
| **RunPod 팟** | 훈련의 주 경로. 로컬 `uv.lock`을 굳힌 이미지로 템플릿을 만들고, 볼륨(`/workspace`)에 `src/`와 HF 캐시를 두고 노트북을 돌린다. SSH가 끊겨도 컨테이너가 살아 무인 장시간 런이 된다 |
| **Colab L4 (24GB)** | 코드가 확정되기 전의 짧은 실험·디버깅. `colab exec -f nb.ipynb` 한 줄로 왕복이 가장 짧다 |
| **로컬 Windows(CPU)** | 데이터 전처리·통계·오류 분석·잡 제어 등 GPU가 필요 없는 작업. GPU 훈련은 불가 |

> ⚠️ **idle GPU는 계속 과금된다.** Colab은 끝나면 반드시 `colab stop`(또는 self-clean되는 `colab run`)을 부른다. RunPod은 Stop해도 **볼륨 요금이 계속 청구**되므로 며칠 이상 쉴 때는 Terminate가 대개 이득이다([runpod-jobs.md](./infra/runpod-jobs.md)「비용」).

## 공개 산출물 (Hugging Face)

공개하는 것은 모델 둘뿐이다.

| 리포 | 역할 |
| --- | --- |
| [`ingyoun/A.X-patent-len4096-op`](https://huggingface.co/ingyoun/A.X-patent-len4096-op) | 배포 모델(`16_01`). `label_mappings.json`을 함께 담는다 |
| [`ingyoun/kobert-patent-baseline`](https://huggingface.co/ingyoun/kobert-patent-baseline) | 재현한 KoBERT 비교선 — 개선 주장의 검증 수단 |

**실험 런 모델과 가공 데이터셋은 공개하지 않는다.** 따라서 아래 문서들이 적은 리포 ID(`A.X-patent-maxlen8192`·`-len512-*`·`-tapt-mlm`·`-seed153`·`patent-clean-text*` 등)는 조회되지 않으며, **당시 실행의 기록**으로 읽는다. 데이터셋을 AI Hub 원본에서 재생성하는 절차는 [data/data-pipeline.md](./data/data-pipeline.md)「가공 데이터셋은 배포하지 않는다 — 재현 경로」에 있다.

로짓은 저장소에 동봉돼 있어(`output/logits_*.npy`) 이 모델들을 받지 않아도 `scripts/`의 분석은 전부 재현된다.

## 환경 값

| 항목 | 값 |
| --- | --- |
| 로컬 머신 | Windows 11, GPU 없음(CPU 전용). uv 프로젝트 `.venv`(Python 3.12), 인터프리터 `.venv\Scripts\python.exe` |
| 로컬 도구 | `uv` · Docker(훈련 이미지를 로컬에서 빌드해 Docker Hub로 push) · `colab` CLI · `runpodctl` |
| 훈련 이미지 | 로컬 `uv.lock`을 `uv sync --frozen`으로 굳힌 커스텀 이미지. 시맨틱 버전 태그로 고정하고 RunPod 팟 템플릿이 이를 참조한다 |
| 팟 볼륨 | `/workspace` — 훈련 패키지 `src/`와 HF 캐시(`HF_HOME=/workspace/hf_cache`)를 둔다 |
| 인증 env | `HUGGINGFACEHUB_API_TOKEN`(데이터·모델), `WANDB_API_KEY`(관측), `RUNPOD_API_KEY`(팟 제어). `.env`에 저장하고 팟에는 환경 변수로 주입한다 |

Lightning 계정 값(org `paraise-org` · teamspace `ml` · 인증 계정 `paraise-edu`)은 [infra/lightning-jobs.md](./infra/lightning-jobs.md)가 소유한다 — 이 프로젝트의 훈련에는 쓰이지 않았다.

## 문서를 쌓는 규칙

- 재사용할 지식·결과는 대화가 아니라 이 폴더에 문서로 남긴다.
- 인프라를 운영하며 알게 된 사실(오류·해결·플래그·머신 동작)은 해당 `infra/*.md`를 즉시 갱신한다.
- 서술은 3인칭·사실 중심으로 하고, 수정 이력은 문서에 남기지 않는다(상세는 `CLAUDE.md` 「문서 서술 규칙」).
