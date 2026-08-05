# 특허 과학기술표준분류 자동 분류

한국 특허 문헌을 **과학기술표준분류 188개 중분류로 분류하는 다중 레이블 인코더 분류기**다. 한 특허가 여러 중분류에 대응하며(고유 문서의 14.1%가 2개 이상), 17개 대분류는 예측된 중분류에서 유도해 계층 일관성을 보장한다.

**배포 모델: [`ingyoun/A.X-patent-len4096-op`](https://huggingface.co/ingyoun/A.X-patent-len4096-op)** — `skt/A.X-Encoder-base`(한국어 ModernBERT)를 4,096 토큰 창에서 파인튜닝했다.

**평가 탐색기: [ingyoun/patent-eval-demo](https://ingyoun-patent-eval-demo.static.hf.space)** — 저장된 로짓으로 채점 방식·임계값·문서 길이별 결과를 직접 조작해 보는 정적 데모다(소스 `hf-spaces/patent-eval-demo/`).

## 결과

test 11,244건 · τ=0.5 · 같은 문서·같은 순서에서 측정.

| 런 | micro-F1 | macro | sample | empty | P@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| KoBERT (512) — 재현한 비교선 | 0.8500 | 0.8467 | 0.8653 | 1.17% | 0.8935 |
| A.X-Encoder (512) | 0.8588 | 0.8565 | 0.8738 | 1.17% | 0.9000 |
| **A.X-Encoder (4096) — 배포 모델** | **0.8660** | **0.8638** | **0.8835** | **0.83%** | **0.9051** |

- 창 크기만 다른 기준 런 대비 **+0.72pt**(문서 단위 paired bootstrap CI95 [+0.33, +1.10])로 시드 잡음(±0.2pt) 밖이다.
- 이득이 길이 구간에서 단조 증가한다(≤512 +0.29pt → 1024–2048 +1.58pt) — 창 확장의 서명과 일치한다.
- KoBERT 재현선 대비 **+1.60pt**(CI95 [+1.19, +2.00]).

상세한 실측·오류 분석은 [docs/experiments/final-run.md](docs/experiments/final-run.md)에 있다.

## 한계

성능 수치를 인용할 때 함께 읽어야 하는 것들이다.

- **측정 F1은 라벨 잡음만큼 하향 편향돼 있다** — 잡음 후보가 전부 실제 잡음이면 참 성능은 최대 +2.1pt 높다([confident-errors.md](docs/experiments/confident-errors.md)).
- **훈련 분포가 인위적으로 평탄화돼 있다**(중분류당 1,300~2,600건) — 실제 특허 출원 분포와 다르다.
- **원 데이터 배포처의 baseline F1 0.8249와 직접 비교할 수 없다.** 그 값은 다른 test 집합에서 잰 top-1 weighted-F1이고, 그 데이터 분할에는 문서 누수 위험이 있다. 그래서 누수를 제거한 데이터를 새로 만들고 KoBERT를 직접 재현해 비교선을 세웠다([ADR-0001](docs/adr/0001-comparison-baseline.md)).

## 저장소 구조

| 경로 | 내용 |
| --- | --- |
| [PROJECT.md](PROJECT.md) | 목표·접근·모델·평가 프로토콜의 SSOT |
| [docs/](docs/README.md) | 문서 인덱스 — 데이터·실험·결정 기록(ADR)·훈련 인프라 |
| `notebook/` | 전처리·분석 노트북 |
| `notebook_output/` | 훈련·평가 실행 기록(출력 포함) |
| `src/patent_train/` | 훈련 하니스 — 노트북은 `TrainConfig`만 주입한다 |
| `scripts/` | 오류 분석·진단 스크립트 |
| `output/` | 지표·오류 분석 결과(json)와 **모델별 로짓 덤프(npy)** |
| `hf-spaces/patent-eval-demo/` | 공개 평가 탐색기(정적) — 데이터는 `scripts/build_demo_payload.py`가 생성한다 |

**모델별 로짓과 정답 라벨을 저장소에 포함한다**(`output/logits_*.npy` · `output/gold_labels_*.json`). 그래서 **GPU도, 모델 가중치도, 데이터셋도 없이** 임계값 스윕·오류 분석·모델 간 비교가 그대로 돈다. 배포 모델의 전체 오류 분석은 아래 한 줄이다.

```bash
uv run python scripts/error_analysis_final.py
```

`scripts/` 14개 중 **12개가 네트워크 없이 재현된다**. 나머지 둘은 입력 길이 분포를 재는 `length_cost.py`(토큰화 데이터셋 필요)와 IPC 필드를 원본 zip에서 읽는 `ipc_field_analysis.py`다. 동봉하는 정답 라벨은 `document_id`·`label_ids`·길이 bin뿐이며 **원문 텍스트는 포함하지 않는다**.

## 환경

uv 프로젝트이며 Python 3.12로 고정돼 있다(`uv.lock`이 버전의 SSOT).

```bash
uv sync
```

로컬 머신은 Windows CPU 전용이라 훈련은 외부 GPU에서 돌린다. **주 경로는 RunPod 팟**이다 — 로컬 `uv.lock`을 굳힌 커스텀 Docker 이미지로 팟 템플릿을 만들고, 볼륨에 훈련 패키지와 HF 캐시를 두고 노트북을 실행한다. 코드가 확정되기 전의 짧은 실험은 Colab이 맡았다. 절차는 [docs/](docs/README.md)의 인프라 문서에 있다.

로짓·지표 분석은 GPU 없이 로컬에서 돌아간다.

## 데이터

AI Hub 71531 「과학기술표준분류 대응 특허 데이터」(<https://www.aihub.or.kr>)를 원천으로 한다.

**원본의 재배포 제한 때문에 가공 데이터셋(정제 텍스트·토큰화본)은 배포하지 않는다.** 따라서 코드에 적힌 데이터셋 ID는 그대로 조회되지 않으며, 해당 파일에는 안내 표기가 붙어 있다. AI Hub 원본에서 재생성하는 절차는 [docs/data/data-pipeline.md](docs/data/data-pipeline.md)「가공 데이터셋은 배포하지 않는다 — 재현 경로」에 있다.

라벨 매핑만 필요하면 재생성이 필요 없다 — `output/label_mappings.json`에 188개 중분류의 정렬 순서(= 모델 출력 열 순서)와 중분류→대분류 대응이 들어 있다.

## 라이선스

Apache-2.0([LICENSE](LICENSE)). 베이스 모델 `skt/A.X-Encoder-base`와 같은 라이선스다. 학습 데이터는 AI Hub 이용 약관의 적용을 받으며 이 저장소에 포함되지 않는다.
