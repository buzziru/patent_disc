# Lightning AI Jobs — 로컬에서 커스텀 Docker 이미지로 GPU 훈련

> 작업은 **로컬 Windows 머신(CPU 전용)** 에서 하고, GPU 훈련은 **로컬에서 Lightning Python SDK로 비동기 Job을 제출**해 돌린다.
> 환경 재현은 **`.venv`를 담은 커스텀 Docker 이미지**로 통일한다(스튜디오 스냅샷 대신). 대안 경로 = [Colab](./colab-jobs.md).
>
> ⚠️ 이전 버전은 "이 세션 = Lightning cloudspace 스튜디오 안"을 전제로 `--studio` 스냅샷 잡을 썼다. 로컬로 옮기면서 **이미지 기반 잡**으로 전환했다. 스튜디오 인터랙티브 세션 진단은 [studio-performance.md](./studio-performance.md)(과거 기록).

## 결론 먼저 — 가능 여부 (SDK v2026.07.03 실측)

로컬에 `lightning-sdk`를 설치해 `Job.run` API를 직접 확인했다.

- ✅ **스튜디오 없이 Docker 이미지만으로 잡 실행 가능.** `Job.run(image=..., command=..., machine=...)`. docstring 원문: *"The docker image to run the job with. **Mutually exclusive with studio.**"* `command`는 이미지 사용 시 **optional**(없으면 컨테이너 엔트리포인트+기본 커맨드 실행).
- ✅ **로컬에서 제출 가능.** SDK가 Windows에 정상 설치·import(`pywin32` 포함). 인증은 env `LIGHTNING_USER_ID` + `LIGHTNING_API_KEY`.
- ⚠️ **`lightning` CLI는 네이티브 Windows에서 작동하지 않는다** — `simple_term_menu → termios`(Unix 전용) import로 크래시. **로컬에선 Python SDK를 쓴다**(아래 예제). CLI가 꼭 필요하면 WSL2/Linux에서.

## `Job.run` 전체 시그니처 (v2026.07.03)

```
Job.run(name, machine, cloud=None, command=None, studio=None, image=None,
        teamspace=None, org=None, user=None, env=None, interruptible=False,
        image_credentials=None, cloud_account_auth=False, entrypoint=None,
        path_mappings=None, max_runtime=None, reuse_snapshot=True, scratch_disks=None)
```

중요한 인자:

| 인자 | 의미 |
|---|---|
| `image` | 실행할 Docker 이미지(레지스트리 경로). **`studio`와 상호 배타**. |
| `command` | 컨테이너 안에서 실행할 커맨드. 이미지 사용 시 생략 가능(엔트리포인트 실행). |
| `machine` | 머신 타입(`Machine.L4` 등). |
| `teamspace` / `org` / `user` | 잡을 붙일 teamspace와 그 소유자. **로컬엔 ambient 컨텍스트가 없으니 반드시 명시**(아래). |
| `env` | 컨테이너 환경변수(예: `HF_TOKEN`, `WANDB_API_KEY`). |
| `image_credentials` | private 이미지 pull용 **Lightning 시크릿 이름**. |
| `cloud_account_auth` | 레지스트리가 클라우드 제공자(ECR 등)일 때 True. |
| `entrypoint` | 기본 `sh -c`. 빈 문자열로 두면 이미지 자체 엔트리포인트 사용. |
| `path_mappings` | 컨테이너 경로 ↔ Lightning data-connection 매핑. 데이터를 HF streaming으로 받으면 불필요. |
| `interruptible` | 스팟 인스턴스(저렴, 선점 가능 → **체크포인트 필수**). |
| `max_runtime` | 머신 할당 상한(초). 기본 3시간. |

## 이 환경의 값

| 항목 | 값 |
|---|---|
| org (teamspace owner) | `paraise-org` |
| teamspace | `ml` |
| 로컬 인증 계정 | `paraise-edu` — teamspace `ml`의 **멤버**이며 owner가 아니다 |
| 이전 스튜디오 | `patent` / `patent_edu` — 이미지 기반 잡에선 **사용하지 않음**(참고용) |

> ⚠️ 로그인 `paraise-edu`는 owner가 아니다. 로컬에서 제출할 땐 ambient 컨텍스트가 없어 **`teamspace`+`org`(또는 `user`)를 반드시 명시**해야 한다. 생략하면 `paraise-edu/ml`로 조회해 `Teamspace ... does not exist`로 실패한다(트러블슈팅 절).

## 로컬 사전 준비

1. **Lightning SDK 설치**: `uv add lightning-sdk`(프로젝트 의존성으로) 또는 잡 제출 스크립트에서 `uv run --with lightning-sdk …`.
2. **인증 키**: lightning.ai → Settings → **API Keys**에서 `LIGHTNING_USER_ID`·`LIGHTNING_API_KEY`를 발급해 로컬 `.env`에 저장. ✅ 완료·검증됨(로컬 Windows에서 인증→`Teamspace` 조회 성공, 2026-07-07).
3. **Docker**: **Docker Desktop 설치·기동 완료**(`docker` 29.6.1, 데몬 UP). 이미지 빌드는 GPU가 필요 없으니 CPU Windows에서 가능.
4. **HF 토큰**: 데이터 streaming용. `.env`의 `HUGGINGFACEHUB_API_TOKEN`을 잡 `env`로 주입.

**`.env`의 Lightning 키(검증됨)** — teamspace/owner slug는 **대소문자·정확한 slug**여야 조회된다(오타 시 `Teamspace ... does not exist`):

```
LIGHTNING_USER_ID=…            # 인증
LIGHTNING_API_KEY=…            # 인증
LIGHTNING_TEAMSPACE=ml         # ⚠️ 소문자 (ML 아님)
LIGHTNING_TEAMSPACE_OWNER=paraise-org  # ⚠️ org slug (paraise 아님)
```

> 확인된 사실: 로그인 계정은 org `paraise-edu-org`의 멤버이고, 대상 teamspace `ml`의 owner org는 `paraise-org`다. SDK는 `Teamspace(name="ml", org="paraise-org")`로 해석된다.

## 커스텀 Docker 이미지 만들기

목표: `.venv`(Python 3.12 + `uv.lock` 고정 — torch cu13, transformers 등)를 그대로 담은 **CUDA 지원 이미지**를 만들어 레지스트리에 올린다.

- **베이스**: CUDA 런타임 이미지(예: `nvidia/cuda:12.x-runtime-ubuntu22.04`) 위에 Python 3.12 + uv. GPU 머신의 드라이버와 **CUDA 버전을 맞춘다**(torch가 잡는 cu13/cu12 확인).
- **의존성 고정**: 이미지 안에서 `uv sync --frozen`(또는 `uv pip install -r`)로 `uv.lock`을 재현 → 로컬·Colab·잡 3곳 버전 일치.
- **코드**: 훈련 코드(`.ipynb`/`.py`)를 이미지에 넣거나, 얇은 이미지 + 잡에서 코드 clone/download. 데이터는 이미지에 넣지 않는다(HF streaming).
- **레지스트리**: Lightning이 pull 가능한 곳에 push.
  - **공개 Docker Hub** → `Job.run(image="<user>/<img>:<tag>")`로 바로 참조(자격증명 불필요).
  - **비공개** → Lightning 시크릿에 자격증명을 등록하고 `image_credentials="<secret-name>"`. 클라우드 레지스트리(ECR 등)면 `cloud_account_auth=True`.

> 스모크 테스트: 먼저 `machine=Machine.CPU_SMALL`로 이미지가 pull·부팅되는지 확인 → 그다음 GPU 머신에서 few-step 훈련으로 파이프라인 검증(쿼터 보호).

## 사용법 — Python SDK (로컬 Windows 권장 경로)

```python
import os
from dotenv import load_dotenv
from lightning_sdk import Job, Machine, Teamspace

load_dotenv()  # LIGHTNING_USER_ID / LIGHTNING_API_KEY / HUGGINGFACEHUB_API_TOKEN

job = Job.run(
    name="patent-train-001",                       # teamspace 내 고유해야 함
    machine=Machine.L4,                            # OOM이면 Machine.A100
    image="<user>/patent-train:cu13-py312",        # 커스텀 이미지
    command="python -m src.train --config configs/axenc.yaml",
    teamspace=Teamspace(name="ml", org="paraise-org"),  # owner 명시(로그인이 owner 아님)
    env={"HF_TOKEN": os.environ["HUGGINGFACEHUB_API_TOKEN"]},  # 데이터 streaming 인증
    interruptible=False,                           # 스팟이면 True + 체크포인트
    max_runtime=14400,                             # 4h 상한
    # image_credentials="dockerhub-creds",         # private 이미지면
)
print(job.name)
# 상태/로그: job.status, job.logs
```

- **인터프리터 경로**: 커맨드는 **컨테이너 안**에서 실행되므로 로컬 `.venv\Scripts\python`이 아니라 **이미지의 python**(`python` 또는 이미지 내 venv 경로)을 부른다.
- **`name`은 매 실행 고유**해야 한다(artifact 경로가 이름 기준).

## 데이터·산출물

- **데이터 반입 = 컨테이너 안에서 HF Hub streaming.** 스튜디오 FS가 없으므로 `load_dataset("<user>/patent-tokenized", split=..., streaming=True)`로 받는다(`HF_TOKEN` env 주입). 설계: [`data-pipeline.md`](../data/data-pipeline.md). (대안: `path_mappings`로 Lightning data-connection 마운트.)
- **산출물 회수**: Job은 별도 머신에서 돌아 로컬 FS에 자동 병합되지 **않는다**. 잡 작업디렉터리를 미러한 **artifact 경로**(`job.artifact_path`, 대략 `/teamspace/jobs/<job-name>/artifacts/...`)에 남는다. 필요한 파일(체크포인트·메트릭·리포트)을 회수하거나, 스크립트 말미에서 **HF Hub / 외부 스토리지로 직접 push**하는 편이 로컬 회수보다 단순하다.
- **로그**: SDK `job.logs`.

## 머신 타입·과금

- **머신**(`Machine` enum): GPU `T4_SMALL`, `T4`, `L4`, `L4_X_2/4/8`, `L40S(_X_…)`, `A100(_X_…)`, `H100(_X_…)`, `H200…`, `B200_X_8` 등. CPU `CPU_SMALL`~`CPU_X_16`, 데이터 전처리 `DATA_PREP/_MAX/_ULTRA`.
- 단일 GPU 인코더 훈련은 `L4`(24GB)로 시작. `skt/A.X-Encoder-base`(ModernBERT, 최대 16k 토큰) 장문 배치가 OOM이면 `A100`(40GB)으로 올린다.
- **과금**: 잡 실행 시간만 과금, 종료 시 머신 회수(상시 idle 비용 없음). `interruptible=True` = 스팟(저렴, 선점 가능 → **체크포인트 필수**). `max_runtime`으로 상한.

## 운영 팁

- **CLI 인자로 넘기기 까다로운 설정**(리스트·중첩 오버라이드 등)은 커맨드에 직접 쓰지 말고 **config 파일**로 빼서 이미지에 넣거나 잡에서 받아 이름만 넘긴다.
- **스팟(`interruptible=True`) 사용 시 체크포인트 저장을 반드시 켠다** — 선점되면 처음부터 다시 돈다.
- **이미지 변경 시 태그를 올린다**(`:cu13-py312-v2`) — 같은 태그 재사용은 캐시/혼동을 부른다.
- **end-to-end 검증**: 이미지 push → CPU 스모크 → GPU few-step → `Completed` → artifact/HF 산출물 확인.

## 트러블슈팅

- **`Teamspace paraise-edu/ml does not exist ... member of organizations: []`**: teamspace `ml`의 owner는 org `paraise-org`인데 owner를 생략해 `paraise-edu/ml`로 조회한 것 → `Teamspace(name="ml", org="paraise-org")`로 명시. (토큰/로그인 정상이어도 발생 — 인증 문제 아님.)
- **`lightning` CLI가 Windows에서 `NotImplementedError: "Windows" is currently not supported` / `No module named 'termios'`**: CLI는 Unix 전용. **SDK를 쓰거나** WSL2/Linux에서 실행.
- **이미지 pull 실패(private)**: `image_credentials` 시크릿 이름·자격증명 확인. 클라우드 레지스트리면 `cloud_account_auth=True`.
- **컨테이너에서 GPU 미인식**: 이미지 CUDA 버전이 머신 드라이버와 안 맞음 → torch가 잡는 CUDA(`torch.version.cuda`)와 베이스 이미지 CUDA를 맞춘다.

## Sources

[Batch jobs SDK](https://lightning.ai/docs/overview/batch-jobs/sdk) · [Submit jobs](https://lightning.ai/docs/overview/scale-with-batch-jobs/submit-jobs) · [Custom Docker images](https://lightning.ai/docs/overview/ai-studio/custom-docker-images) · [Artifacts](https://lightning.ai/docs/overview/artifacts) · [SDK reference](https://lightning.ai/docs/overview/sdk-reference) · SDK v2026.07.03 `Job.run` docstring(로컬 실측)
