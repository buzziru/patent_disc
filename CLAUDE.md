# [CLAUDE.md](http://CLAUDE.md)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

특허 문헌을 **과학기술표준분류 188개 중분류로 자동 분류하는 인코더 분류기**를 만들고, 공식 baseline 대비 개선을 정량 입증한다. 문서당 **다중 레이블(multi-label)** 188-way 분류(17대분류/188중분류) — 한 특허가 여러 중분류에 대응(고유 문서의 ~ 14%가 2~10개 Mno). 주 모델은 `skt/A.X-Encoder-base`(한국어 ModernBERT, 16k 컨텍스트, apache-2.0).

**산출물은 운영 가능한 단일 모델의 성능과 그것을 끌어올린 방법론이며, 특정 가설의 증명이 아니다.** long-document는 출발점이 된 아이디어이고 이미 실측이 끝났다 — 그 축의 추가 검증(입력 필드 실험·대조군 등)은 목표가 아니다. 실험 선택 기준은 **성능 개선 기대치**이며, 기법은 근거 문헌을 먼저 확보한 뒤 적용한다. 전체 명세는 `PROJECT.md`(SSOT) — 작업 전 필독.

## Always follow

- **작업 전** `PROJECT.md`**를 읽는다.** 목표·접근·모델·평가 프로토콜의 SSOT다(`owner/`는 참고 자료일 뿐 SSOT 아님).
- **다중 레이블**: 학습 타깃은 문서별 `Mno` 다중-핫(188-class, sigmoid + Focal Loss γ=2). 대분류 `Lno`는 **예측된 각** `Mno`→`Lno` 매핑으로 유도한다.
- **재사용할 지식·결과는 대화가 아니라** `docs/`**에 문서로 남긴다.** 특히 **인프라를 운영하며 알게 된 사실(오류·해결·플래그·머신 동작 등)은 해당** `docs/*.md`**를 즉시 최신화**한다.
- **이 세션은 로컬 Windows 머신 — GPU 없음(CPU 전용).** 훈련은 외부 GPU에서 돌린다. **주 경로는 RunPod 팟**이다 — 로컬 `uv.lock`을 굳힌 커스텀 이미지로 팟 템플릿을 만들고, 볼륨 `/workspace`에 `src/`와 HF 캐시를 둔다(`docs/infra/runpod-jobs.md`). 코드 확정 전의 짧은 실험은 Colab이 맡았다(`docs/infra/colab-jobs.md`). **Lightning은 이 프로젝트의 훈련에 쓰이지 않았다.**
- **환경은 uv 프로젝트** `.venv`**(Python 3.12).** 패키지 설치는 `uv add`(시스템 pip/conda 금지), 실행은 `uv run …` 또는 `.venv\Scripts\python.exe`(Windows venv 레이아웃). 모델·토크나이저·데이터 코드는 이 `.venv`에서 돌린다. **Jupyter/ipynb 커널도** `.venv` **인터프리터를 선택**한다(`.venv\Scripts\python.exe`). Colab·도커 이미지 호환 위해 3.12 고정 — `uv.lock`이 버전의 SSOT.
- `data/`는 압축 상태로 `zipfile` 스트리밍(419MB, gitignored) — **대량 unzip 금지**. 전처리 1회로 토크나이즈해 HF Hub에 올리고 원격 GPU에서 streaming 소비한다(`docs/data/data-pipeline.md`). **가공 데이터셋은 공개 배포하지 않는다** — 재생성 절차는 같은 문서의 「재현 경로」에 있다.
- **개발·훈련은** `.ipynb` **중심.** Colab에서 `colab exec -f nb.ipynb`로 그대로 실행하니 `.py`로 옮길 필요 없다(상세 `docs/infra/colab-jobs.md`).



## 문서 서술 규칙

문서·주석은 **제3자 독자**를 전제로 작성한다.

- **1인칭 지칭('우리' 등)을 쓰지 않는다.** 주체를 내세우지 말고 사실·절차로 서술한다(예: "우리는 재분할한다" → "`documentId` 단위로 재분할한다").
- **한영 혼용 표현을 쓰지 않는다.** 단, 영어 기술용어를 전부 한국어로 옮기라는 뜻은 아니다 — `split`·`test`·`val`·`baseline`처럼 통용되는 일반 용어와 `KoBERT`·`ModernBERT`처럼 대체 불가한 고유명은 그대로 둔다. `vintage`·`lookup`처럼 자연스러운 한국어 대응어가 있는 표현만 바꾼다(vintage→스냅샷/버전, lookup→조회/매핑).
- **수정 이력을 문서에 남기지 않는다.** 틀린 사항을 지적받아 고칠 때 "원래 X였으나 Y로 정정", "모순 제거", "이전 값 삭제" 같은 변경 서술을 쓰지 말고 **최종 내용만** 사실로 서술한다(변경 경위는 커밋·대화에 남고, 문서엔 결과만 남긴다).



## Hard boundary

- **공식 baseline F1 0.8249를 절대 기준으로 비교하지 말 것** — 이는 **top-1 예측의 weighted-F1**(full test 24,525건 실측)이다. baseline 데이터 분할은 `documentId`**가 train·val 양쪽에 존재하는 데이터 누수** 위험이 있어, **누수 없는 데이터셋을 새로 생성**하고 **자체 test 고정 + KoBERT 자체 재현**으로 비교선을 세운다(`PROJECT.md` 평가 절).
- **독립된 두 분류 헤드(대분류·중분류 별도 예측)를 만들지 말 것** — 계층 비일관성을 유발한다. Mno 다중 예측 + `Mno`→`Lno` 매핑으로 해결.
- **앙상블을 제안하지 말 것** — 운영 환경이 단일 모델을 요구한다. 이득은 실측됐으나(앵커 weighted-F1 +0.71pt) 채택하지 않는다.
- `PROJECT.md` **「닫힌 갈래」의 항목을 새 근거 없이 다시 제안하지 말 것** — 계층 확장·임계값 정책·캘리브레이션·앙상블·입력 필드 실험·long-document 가설 추가 검증. 측정으로 결론이 난 항목이다.
- **멀티모달(도면 이미지)은 스코프 밖.**
- 평탄화된 분포(중분류당 1,300~2,600건)는 실제 출원 분포와 다르다 — 결과에 한계로 명시.



## Routing table

작업에 필요한 문서를 카테고리 단위로 찾는 표. **파일별 세부 설명·전체 목록은** `docs/README.md`**(문서 인덱스)를 참고한다.**


| 필요할 때                                     | 문서                                    |
| ----------------------------------------- | ------------------------------------- |
| 프로젝트 전체 명세(목표·접근·모델·baseline 주의) — **SSOT** | `PROJECT.md`                          |
| 문서 인덱스·환경 값(org/teamspace/studio) — 아래 각 문서의 세부 설명 | `docs/README.md`                      |
| 데이터 레이아웃·스키마·전처리 파이프라인                     | `docs/data/`                          |
| 실험 계획·프로토콜·실측(baseline 재현·ModernBERT·손실 함수·대조군·무훈련 분석) | `docs/experiments/`                   |
| 결정 기록·회고(사안별 ADR)                         | `docs/adr/README.md`                  |
| GPU 훈련을 외부에서(RunPod 주 경로 · Colab 초기 실험)     | `docs/infra/`                         |


