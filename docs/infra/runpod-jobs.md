# RunPod — 커스텀 Docker 이미지로 GPU 훈련

> `uv.lock`을 이미지에 굳혀(`uv sync --frozen`) 로컬·Colab·Lightning과 **비트 단위 동일 환경**을 재현하고, RTX 4090/L4급 팟에서 `.ipynb`를 돌리는 경로.
> 대안 경로 = [Colab Job](./colab-jobs.md)(기본) · [Lightning Job](./lightning-jobs.md).

## 언제 RunPod을 쓰나

- **Colab L4 헤드리스 회수(~20분, `colab-jobs.md` 실측)를 피하고 싶은 장시간 런.** 팟은 SSH가 끊겨도 컨테이너가 죽지 않아 `tmux`+무인 훈련이 가능하다.
- **환경을 이미지로 완전히 고정**하고 싶을 때 — 매 세션 `!pip`로 버전 드리프트가 나는 Colab과 달리, 이미지 태그가 곧 환경이다.
- **비용은 초 단위 종량제.** 팟이 RUNNING인 모든 초에 GPU가 과금되고, Stop해도 볼륨 디스크는 계속 청구된다 → 아래 「비용」 절.

## 3층 분리 — 무엇을 이미지에 넣고 무엇을 빼나

기준은 "변경 주기"다.

| 층 | 내용 | 위치 | 이유 |
| --- | --- | --- | --- |
| 환경 | CUDA·torch·transformers·flash-attn 등 의존성 | **이미지**(`uv.lock` 재현) | 거의 안 바뀜. 바뀌면 재현성 붕괴 |
| 코드 | 훈련 노트북(`notebook/*.ipynb`) | 이미지 **밖** — 접속 후 반입/실행 | 커밋마다 바뀜. 이미지 재빌드 루프를 피함 |
| 데이터·산출물 | 토큰화 데이터셋·체크포인트·HF 캐시 | `/workspace` 볼륨 | 크고, AI Hub 재배포 제약. 이미지에 넣으면 pull이 느려짐 |

- **데이터는 이미지에 절대 넣지 않는다** — 팟 안에서 HF Hub streaming으로 받는다(`ingyoun/patent-clean-text-modernbert-tokenized`, 설계: [`data-pipeline.md`](../data/data-pipeline.md)). AI Hub 71531은 재배포 제약이 있어 public 이미지에 섞이면 안 된다.
- **HF/모델 캐시는 `/workspace` 볼륨**으로 유도(`HF_HOME=/workspace/hf_cache`, 이미지 ENV에 고정). 컨테이너 디스크(`/`·`/app`·`/root`)는 Stop 시 소실되지만 볼륨은 남아 재다운로드가 없다.

## 이미지 (`Dockerfile`) — 환경 SSOT = `uv.lock`

프로젝트 루트 `Dockerfile`이 이미지를 정의한다. 핵심 설계:

- **베이스 = `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`.** 엔트리포인트에 SSH/Jupyter/`PUBLIC_KEY` 처리가 이미 들어 있어 "SSH 설정"이라는 변수를 제거한다. 베이스의 시스템 torch·Python은 **쓰지 않고**, 훈련은 아래 `/opt/venv`(uv.lock 재현본)로 돌린다.
- **환경 설치 = 한 줄:**
  ```dockerfile
  RUN uv sync --frozen --no-dev --group gpu
  ```
  - `--frozen`: `uv.lock`을 **재해석 없이 그대로 설치**. lock이 `pyproject.toml`과 어긋나면 빌드가 즉시 실패한다(재현성 가드). 빌드 전 로컬에서 `uv lock --check`로 일관성을 확인한다.
  - `--group gpu`: `flash-attn` 프리빌트 휠(`cu12torch2.11`·`cp312`·linux 마커)을 설치. 이 그룹은 **기본 `uv sync`에 안 들어가므로 반드시 명시**한다. 프리빌트라 nvcc 불필요 → 이미지에 CUDA devel 툴킷을 담지 않는다.
  - `--no-dev`: `nbdime` 등 dev 그룹 제외.
- **베이스는 py3.11인데 venv는 3.12다.** `requires-python = ">=3.12,<3.13"`(+ cp312 flash-attn 휠)이라, `uv sync`가 베이스에 없는 **관리형 CPython 3.12를 자동 다운로드**해 `/opt/venv`를 만든다. 그래서 `UV_PYTHON_DOWNLOADS`를 끄면 안 된다.
- **torch는 linux에서 cu128 빌드**로 받는다(`[tool.uv.sources]`의 `sys_platform == 'linux'` 마커 → `pytorch-cu128` 인덱스). PyPI 기본 torch 2.11은 cu13이라 cu12 flash-attn 휠과 CUDA 메이저가 어긋난다. 로컬 Windows는 PyPI CPU torch를 유지.
- **ipykernel 등록**: `patent_disc`(display `patent_disc (/opt/venv)`). 노트북 커널로 이것을 선택한다.
- **엔트리포인트/CMD를 덮지 않는다** — 베이스 기본 CMD가 SSH/Jupyter를 띄운다. 오버라이드하면 접속이 끊긴다(무인 배치 모드에서만 오버라이드).

## ⚠️ Trainer의 런타임 의존성은 **명시 의존성**이어야 한다 (실측)

`transformers`를 `dependencies`에 넣었다고 훈련이 도는 게 아니다. HF `Trainer`는 **런타임에 별도 패키지를 import**하는데, 이들이 `pyproject.toml`에 없으면 lock/이미지에도 없어 팟에서야 터진다. `/opt/venv`는 격리돼 있어 **베이스 이미지의 시스템 패키지(system wandb 등)도 보이지 않는다** — venv에 명시적으로 있어야 한다.

이 프로젝트에서 실제로 밟은 두 지뢰:

| 패키지 | 무엇이 요구하나 | 없으면 터지는 지점 | 에러 |
| --- | --- | --- | --- |
| `accelerate` | `from transformers import Trainer` | import 단계 | `ImportError`(accelerate 요구) |
| `wandb` | `TrainingArguments(report_to="wandb")` | **Trainer 생성 시점** | `RuntimeError: WandbCallback requires wandb to be installed` |

- `report_to="wandb"`가 명시되면 Trainer 생성자가 `WandbCallback`을 즉시 인스턴스화하고, 그 `__init__`이 `is_wandb_available()`가 False면 바로 `RuntimeError`를 던진다. `transformers 5.13` 소스로 실측 확인.
- **둘 다 메인 `dependencies`에 고정한다**(dev/gpu 그룹 아님 — `--no-dev`로도 항상 설치되도록). 현재 lock: `accelerate 1.14.0`, `wandb 0.28.1`.
- **일반 규칙**: `report_to`·`optim`·`gradient_checkpointing` 등 Trainer 옵션을 바꿀 때, 그 기능이 끌어오는 런타임 패키지가 `pyproject.toml`에 있는지 먼저 확인한다. 없으면 추가 → `uv lock` → **이미지 재빌드**.

## 빌드 전 로컬 검증 (GPU 없이, CPU에서)

로컬 Windows/CPU에서 아래를 통과시키면 GPU 요금을 내며 오타를 찾는 일이 없다.

1. **lock ↔ pyproject 일관성**(`--frozen` 빌드 실패 예방):
   ```powershell
   uv lock --check   # EXIT 0 이어야 빌드의 --frozen 이 통과
   ```
2. **의존성 그래프·버전 충돌**(로컬 `.venv`에서, `--group gpu`는 linux 마커라 로컬에선 flash-attn을 건너뜀):
   ```powershell
   .venv\Scripts\python.exe -c "import transformers, accelerate, wandb, datasets, sklearn, sentencepiece; print(transformers.__version__)"
   ```
   - ⚠️ `flash_attn`은 **로컬(비-linux)에서 설치되지 않으므로** import 검증이 불가하다. `attn_implementation="flash_attention_2"` 경로는 **팟에서만** 확인된다(접속 직후 `python -c "import flash_attn"`).
3. **TrainingArguments 인자 유효성**(transformers 5.x는 API가 바뀐 항목이 있다 — 예: `group_by_length`(bool)가 `train_sampling_strategy="group_by_length"`로 변경). 로컬 `.venv`의 실제 transformers로 대조한다.
4. **eval 샘플러 분기**(로짓 행 순서 회귀 방지): 더미 `datasets.Dataset`으로 `Trainer._get_eval_sampler`가 `train_sampling_strategy="group_by_length"`에선 `LengthGroupedSampler`, `"sequential"`에선 `SequentialSampler`를 반환하는지 assert한다. 이 설정은 **train뿐 아니라 eval·predict 로더에도 적용**되어, `group_by_length`인 채 로짓을 덤프하면 행이 길이 그룹 순열로 나온다(지표는 멀쩡, 로짓만 어긋남 — 복원 불가). `predict_logits`가 덤프 동안 `"sequential"`로 되돌리고 반환 라벨로 행 순서를 assert하는 것이 방어선이다.

## 팟 생성·접속 (`runpodctl`)

사전 준비(1회): API 키(`runpodctl config --apiKey=…`), SSH 공개키 등록(`runpodctl ssh add-key`), Docker Hub 로그인. 이미지 push:

```powershell
$sha = git rev-parse --short HEAD
docker build -t <DOCKERHUB_USER>/patent-disc:$sha .
docker push  <DOCKERHUB_USER>/patent-disc:$sha
```
- 태그를 `latest`로 두지 않는다 — 재현성에서 `latest`는 "어떤 이미지인지 모른다"와 같다. git short SHA 또는 시맨틱 버전.

팟 생성(웹 콘솔 또는 CLI). 주요 설정값:

| 항목 | 값 | 근거 |
| --- | --- | --- |
| GPU | RTX 4090(24GB) 1장 또는 L4 | 188-way 인코더 파인튜닝에 24GB면 충분(len8192는 여유 확인 필요) |
| Container Image | `<DOCKERHUB_USER>/patent-disc:<sha>` | |
| **CUDA Version 필터** | **≥ 12.8** | 이미지가 cu12.8을 요구 — 호스트 드라이버가 낮으면 `OCI runtime create failed`로 컨테이너가 아예 안 뜬다 |
| Volume Disk / Mount | 20GB+ / `/workspace` | 이미지 ENV `HF_HOME=/workspace/hf_cache`와 **일치해야** 캐시가 볼륨에 남는다 |
| Container Disk | 20GB+ | devel 베이스 압축 해제분 + 여유 |
| Environment Variables | `HF_TOKEN`, `WANDB_API_KEY` | 이미지에 넣지 않고 팟에서 주입. `WANDB_PROJECT`는 이미지 ENV에 이미 박힘 |

접속 후 반드시 확인:
```bash
nvidia-smi
/opt/venv/bin/python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.get_device_name(0))"
echo $HF_HOME        # /workspace/hf_cache 여야 함
df -h /workspace
```
- `torch.cuda.is_available()`가 False면 CUDA 필터 문제 — 팟을 지우고 필터를 올려 재생성.

## 노트북 실행

훈련 코드는 `src/patent_train` 패키지에 있고(config·data·model·losses·metrics·trainer·runner·probe, ADR-0011 코드 성숙 전환), 노트북은 이를 임포트해 config 하나로 실행한다 — 손실/모델/레시피 변형은 `TrainConfig` 필드만 바꾸면 되고 코드는 불변이다:

> ⚠️ **런 시작 전에 팟의 `src/patent_train`을 로컬 최신본으로 교체한다.** 볼륨·이미지에 남은 구 사본이 그대로 import되면 로컬에서 고친 코드가 반영되지 않은 채 돈다 — `13_02`가 이 경로로 행 순열 로짓을 냈다(아래 「eval 샘플러 분기」 방어가 로컬에는 있었으나 팟 사본에 없었다). 임포트 직후 `patent_train.__file__`을 찍어 반입 경로를 확인한다.

```python
import sys; sys.path.insert(0, "src")            # /workspace/src (또는 uv pip install -e . 로 최상위 설치)
from patent_train import TrainingRunner, TrainConfig, probe_batches

cfg = TrainConfig(backbone="axenc",                          # backbones.BACKBONES 키(인코더+데이터셋)
                  loss="focal", loss_params={"alpha": 0.25, "gamma": 2}, max_len=512,
                  learning_rate=4.8e-4, weight_decay=0.01, warmup_ratio=0.1, early_stop_epochs=2,
                  eff_batch=128, micro_batch=128, eval_micro_batch=512,   # 최적화 레시피는 노트북에서 명시
                  tag="modernbert-patent-len512", repo_final="ingyoun/A.X-patent-len512",
                  out_path="/workspace/output/modernbert-len512", search=True)  # fast-fail
runner = TrainingRunner(cfg)
runner.load_data()        # 토크나이저 + 원본 데이터셋(prep 캐시 있으면 원본 다운로드 생략)
runner.prepare_data()     # max_len 절단 → prep 캐시
runner.load_model()       # (= 위 셋을 잇는 축약: runner.setup())
probe_batches(runner.model, runner.data.tokenizer.vocab_size, cfg.max_len)   # 선택: micro_batch OOM 탐침
runner.build_trainer()
runner.train()                    # 훈련만
print(runner.evaluate("test"))    # 평가만 — 메트릭 dict 반환(출력은 호출부)
runner.save_metrics()             # runner.metrics(split 전체) → {tag}_metrics.json
runner.push_to_hub()              # 가중치는 Hub로만. 로컬 사본이 필요할 때만 runner.save_model()
runner.predict_logits("test")     # logits_{tag}_test.npy 덤프 → error_analysis 인계
```

`TrainConfig`의 시간 축은 모두 **에폭 단위**다 — `epochs`·`evals_per_epoch`(에폭당 eval·save 횟수)·`early_stop_epochs`(개선 없이 견디는 에폭 수). step 수와 `EarlyStoppingCallback`의 eval 단위 patience 환산은 `runner.build_trainer`가 하고, 환산 결과를 `[schedule]` 한 줄로 출력한다. `epochs`는 풀런(`search=False`)에서 필수 입력이고, 짧은 탐색 런(`search=True`)에서만 2로 유도된다. bf16은 레시피가 아니라 백본 제약이라 `Backbone` 스펙에 있다(flash-attention-2 필수 조건).

패키지(및 실행 노트북)는 이미지 밖에 있으므로 접속 후 반입한다(git clone 또는 SCP/`runpodctl send`). 실행은 **`/opt/venv` 커널**로:

- Jupyter: 커널에서 `patent_disc (/opt/venv)` 선택.
- 커맨드라인: `cd /app && uv run --no-sync jupyter …`(`--no-sync`로 lock 재해석 없이 `/opt/venv` 사용) 또는 `/opt/venv/bin/python`.
- **산출물은 반드시 `/workspace` 아래**(`out_path`·`output_dir`)로 — 컨테이너 디스크에 쓰면 Stop 시 소실. 최종 모델은 `push_to_hub`로 HF Hub에 올리면 팟을 지워도 남는다.
- **fast-fail**: 풀런 전 `SEARCH=True`(짧은 런)로 GPU·메모리·파이프라인을 먼저 검증한 뒤 `SEARCH=False` 풀런.
- 무인 장시간 런은 SSH 끊김에 견디도록 `tmux` 안에서 실행.

## 관측·wandb

`report_to="wandb"` + `WANDB_PROJECT`(이미지 ENV) + `WANDB_API_KEY`(팟 env)면 별도 로그인 없이 붙는다. 훈련 프로세스가 HTTP로 wandb.ai에 직접 푸시하므로 브라우저 어디서든 loss·F1·lr을 실시간 관측한다. 안 붙으면 `env | grep WANDB`로 주입 여부부터 확인 — 팟 env 오타가 가장 흔한 원인.

## 비용 — Stop ≠ Terminate

- **Stop**: GPU 반납 + `/workspace` 볼륨 보존. 컨테이너 디스크는 소실. **볼륨 요금은 정지 중에도 계속 청구.**
- **Terminate(delete)**: 완전 삭제 — `/workspace`도 사라진다. 데이터·체크포인트는 HF Hub에 있으므로 잃을 것이 없다.
- ⚠️ **실행 중인 팟을 Edit하면 리셋**되어 볼륨 밖 데이터가 지워진다. 이미지 태그를 바꾸려면 Edit 대신 **새 팟**으로 띄운다.
- 며칠 이상 쉴 거면 Terminate가 대개 이득(볼륨 월 요금 vs 이미지 재pull 수 분의 GPU 요금).

## 트러블슈팅 — 증상별

| 증상 | 원인 | 조치 |
| --- | --- | --- |
| `ImportError`(accelerate) / `RuntimeError: WandbCallback requires wandb` | Trainer 런타임 의존성이 명시 의존성에 없음 | `pyproject.toml`에 추가 → `uv lock` → **이미지 재빌드**(위 「런타임 의존성」 절) |
| `uv sync --frozen` 빌드 실패 | lock ↔ pyproject 불일치 | 로컬에서 `uv lock` 후 커밋, 재빌드. 빌드 전 `uv lock --check` |
| `import flash_attn` 실패(팟) | `--group gpu` 미설치 또는 torch/CUDA 메이저 불일치 | Dockerfile의 `uv sync … --group gpu` 확인. torch=cu128 / 휠=cu12 / 팟 CUDA≥12.8 |
| `OCI runtime create failed` / `cuda.is_available()`=False | 호스트 CUDA 드라이버 < 이미지 요구 | 팟 삭제 후 CUDA Version 필터 ≥12.8로 재생성 |
| `TypeError: unexpected keyword` (TrainingArguments) | transformers 5.x API 변경 | 로컬 `.venv`의 실제 버전으로 인자명 대조(예: `train_sampling_strategy`) |
| 지표는 정상인데 덤프 로짓 기반 오류 분석이 전부 0 근처 | `train_sampling_strategy="group_by_length"`가 eval·predict 로더에도 적용되어 반환 행이 길이 그룹 순열. 방어가 있는데도 재발했다면 팟 `src`가 구 사본이다(`patent_train.__file__` 확인) | 사후 복원 불가 — `src` 동기화 후 `predict_logits`(순차 샘플러 복귀 + 행 순서 assert)로 재덤프. 훈련은 불필요하고 Hub 모델로 추론만 다시 돌린다 |
| 재시작 후 HF 모델 재다운로드 | `HF_HOME`이 볼륨 밖 | Volume Mount가 `/workspace`, `echo $HF_HOME`이 그 아래인지 |
| 체크포인트 소실 | `output_dir`이 컨테이너 디스크 | `/workspace` 아래로 |
| wandb 미기록 | 팟 env 미주입 | `env \| grep WANDB` |

진단 원칙: **로컬(CPU)에서 재현되면 이미지 문제, 안 되면 팟/호스트 문제.** import·버전·API 오류는 GPU 없이 로컬에서 잡는다 — GPU 위에서 이걸 디버깅하는 게 돈이 새는 가장 흔한 경로다.
