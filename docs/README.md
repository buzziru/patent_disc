# 프로젝트 문서 (`docs/`)

특허 과학기술표준분류 분류(`[PROJECT.md](../PROJECT.md)`, SSOT) 진행 중 **참고·정리용 문서**를 모으는 곳.
인프라 운영뿐 아니라 데이터 분석 노트, 실험 결과, 의사결정 기록 등을 여기에 쌓는다.
(사람이 쓰든 Claude가 쓰든) 재사용할 만한 내용은 대화에 묻지 말고 이 폴더에 문서로 남긴다.

## 폴더 구조

문서가 늘어남에 따라 범주별 하위 폴더로 분류한다. 인덱스(이 파일)는 루트에 유지.

- `infra/` — GPU 훈련을 외부 잡으로 돌리는 방법(로컬은 CPU 전용)
- `data/` — 데이터 레이아웃·스키마·전처리 파이프라인
- `experiments/` — baseline 재현·실험별 설정/결과·의사결정 기록

## 현재 문서

### 인프라 운영 (`infra/`)

GPU 훈련을 외부 잡으로 돌리는 방법 — 이 세션은 **로컬 Windows(CPU 전용)** 이라 훈련은 외부에서 한다.

- `[infra/lightning-jobs.md](./infra/lightning-jobs.md)` — Lightning AI Jobs: **로컬에서 Python SDK로 커스텀 Docker 이미지 잡을 제출**(스튜디오 스냅샷 대신). `lightning` CLI는 Windows 미지원 → SDK 사용
- `[infra/colab-jobs.md](./infra/colab-jobs.md)` — Google Colab: `colab` CLI로 헤드리스 L4 GPU에서 self-contained 스크립트 실행
- `[infra/studio-performance.md](./infra/studio-performance.md)` — ⚠️ **과거 Lightning cloudspace 기록**(로컬 무관): 스튜디오 느림/"CPU 과부하" 진단, 컨테이너 vs 호스트 CPU 구분, 노이지 네이버 대처

### 데이터 (`data/`)

- `[data/data.md](./data/data.md)` — AI Hub 71531 레이아웃, JSON 스키마, 원천↔라벨 조인, 17대분류/188중분류, 다중레이블·분포·baseline 주의
- `[data/data-pipeline.md](./data/data-pipeline.md)` — 2계층 파이프라인: zip 스트리밍 파싱 → 정제 텍스트 HF Hub push → 소비 시 토큰화

### 실험·기록 (`experiments/`)

- `[experiments/kobert-baseline.md](./experiments/kobert-baseline.md)` — KoBERT baseline을 고정 test set 위에서 직접 재현해 **비교 기준점**(top-1 weighted-F1) 수립
- (예정) 데이터 EDA/필드 길이 분석, 클래스 불균형 노트, 실험별 설정·결과 요약, 모델/입력 필드 선택 근거

## 훈련 인프라 요약

문서당 다중 레이블(multi-label) 188-way 인코더 분류(주 모델 `skt/A.X-Encoder-base` — 한국어 ModernBERT 16k / 대조군 KLUE-RoBERTa-large 512). 단일 GPU로 충분, multi-GPU 불필요.


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

