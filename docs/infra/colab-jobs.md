# Google Colab — 헤드리스 GPU 훈련 (`colab` CLI)

> **L4 24GB** GPU에서 훈련을 돌리는 경로. 노트북 UI 업로드가 아니라 **`colab` CLI**로 원격 VM에서 로컬 `.ipynb`/`.py`를 실행한다(`colab exec -f`는 둘 다 지원) = 에이전트로 자동화 가능. 이 프로젝트는 **`.ipynb` 중심**.
> 대안 경로 = [Lightning Job](./lightning-jobs.md).

## 언제 Colab을 쓰나

- 단일 GPU로 충분한 인코더 훈련의 **기본 경로**. 장문 인코더(`skt/A.X-Encoder-base`, ModernBERT 계열, 최대 16,384 토큰)는 512 토큰 대비 메모리를 많이 써 L4 24GB가 안전.
- ⚠️ accelerator는 tier-gated — L4 쿼터가 없으면 `--gpu T4`(16GB) fallback 또는 Lightning L4 Job으로.

## 설치·인증 (1회)

- **설치**: `uv tool install google-colab-cli` (또는 `pip install google-colab-cli`). 로컬(Windows)에 아직 미설치.
- **인증 = ADC**(Application Default Credentials, 헤드리스 기본). Colab 백엔드는 **4개 스코프 전부** 필요 — 빠지면 401/403:
  ```bash
  gcloud auth application-default login \
    --scopes=openid,\
  https://www.googleapis.com/auth/cloud-platform,\
  https://www.googleapis.com/auth/userinfo.email,\
  https://www.googleapis.com/auth/colaboratory
  ```
  각 스코프 이유: `userinfo.email`(세션 백엔드 `colab.research.google.com`, 없으면 401) · `colaboratory`(keep-alive RuntimeService `colab.pa.googleapis.com`, 없으면 403) · `openid`+`cloud-platform`(gcloud가 요구).
  - ⚠️ 이 `gcloud ...`는 **브라우저 동의가 필요** → 에이전트가 아니라 **사용자가 직접**(세션에서 `! gcloud ...`) 실행한다.
- **인증 검증(1샷)**: `colab whoami`(활성 이메일·스코프·만료 출력) 또는 `colab sessions`. `colab.pa.googleapis.com`에 403이면 거의 항상 `colaboratory` 스코프 누락.
- ⚠️ **`colab auth` ≠ CLI 인증.** `colab auth`는 VM 안쪽 GCP 크레덴셜 주입(노트북 코드가 GCS/BigQuery 호출용)이며 인터랙티브다. CLI 401/403 해결책이 **아니다** — 그건 위 `gcloud` 스코프 문제.

## 멘탈 모델 (먼저 읽기)

- **session == 임대 VM 위의 살아있는 Jupyter 커널.** `colab new`가 과금 VM을 할당, `colab stop`이 반환. 24h keep-alive cap 외에는 자동 회수가 없어 **stop 안 하면 컴퓨트가 계속 소모된다.**
- **커널 state는 같은 세션의 `colab exec` 호출 간 유지된다.** import·변수·함수가 살아남으므로 매 호출 재import 불필요. 리셋은 `colab stop` 또는 `colab restart-kernel`.
- **기본 작업 디렉터리 = `/content`.** 파일 작업은 절대경로(`/content/...`) 권장.
- **`colab`은 fire-and-forget.** 각 명령이 인증→한 가지 작업→종료. keep-alive는 백그라운드 데몬이 담당.

## 실행 단위: `.ipynb`(기본) vs `.py`(one-shot 배치)

**이 프로젝트는 노트북(`.ipynb`) 중심으로 작업한다.** `colab exec -f`는 `.py`·`.ipynb`를 모두 받으므로 `.py`로 옮길 필요가 없다.

- **`.ipynb` 실행**: `colab exec -s <name> -f nb.ipynb` → 각 코드 셀을 순서대로 실행하고 **결과가 채워진 `nb_output.ipynb`를 로컬 원본 옆에 저장**한다(수동 업로드 불필요). 셀 첫 줄 `# @title 제목`으로 진행 로그에 셀 라벨이 붙는다. 플롯/이미지는 자동 인터셉트(`--output-image <path>`로 저장 위치 지정).
- **`.py`가 유리한 경우**: one-shot `colab run script.py [args...]`(= `new`+`exec`+`stop` 한 방, VM 자동 회수) — `sys.argv`·`__name__=="__main__"`가 네이티브 `python`처럼 설정되고 shebang(`#!/usr/bin/env -S colab run --gpu L4`)도 가능. **인자로 파라미터화된 배치 잡 + 자동 teardown**이 필요할 때만. 노트북엔 이 세만틱이 없으므로 세션 방식(new/exec/stop)으로 돌린다.

## 헤드리스 워크플로 (노트북 세션)

`colab new` → `colab exec -f nb.ipynb`(반복) → 산출물 회수 → `colab stop`. 노트북/스크립트는 데이터·코드를 받아 훈련을 돌리고 산출물(체크포인트·메트릭)을 `/content/out`에 쓴다.

1. **세션 시작**: `colab new -s <name> --gpu L4`. ⚠️ `-s <name>`을 항상 지정(생략 시 랜덤 hex = 추적 불가).
2. **실행**: `colab exec -s <name> -f nb.ipynb`(커널 state 지속 → 셀/노트북 나눠 점진 실행 가능). **끝나면 반드시 `colab stop -s <name>`.**
3. **의존성**: `colab install -s <name> transformers datasets accelerate ...` 또는 노트북 상단 `!pip`. torch가 잡은 GPU 확인(`torch.cuda.get_device_name(0)` → L4=sm_89).
   - 로컬 환경은 **Python 3.12 + `uv.lock`**(현재 `torch 2.12.1+cu130`, `transformers 5.12.1`)으로 고정 — Colab과 버전을 맞추기 위함. VM에서도 같은 버전을 설치해 재현성 확보.
   - ⚠️ **Colab의 실제 Python/torch를 확인**(`import sys; print(sys.version)`, `torch.__version__`)해 3.12 가정과 cu13 드라이버 호환을 검증할 것. 어긋나면 `pyproject.toml`의 `requires-python`을 Colab 버전에 맞춘다.
4. **VM측 시크릿**: 노트북/스크립트 안에서 `os.environ`에 키 설정(로컬 `.env` 값을 상수/인자로). Colab Secrets `userdata`는 UI 전용이라 헤드리스엔 부적합.
5. **데이터 반입 = HF Hub pull(streaming)**. VM은 원격·휘발이라 로컬 `data/`를 못 본다 → 노트북 안에서 `load_dataset("<user>/patent-tokenized", split=..., streaming=True)`로 받는다. **로컬 data를 `colab upload`하지 않는다**(419MB를 매 세션 재업로드해야 하므로 비효율). `colab drivemount`는 인터랙티브라 헤드리스 금지. 설계: [`data-pipeline.md`](../data/data-pipeline.md).
6. **경로는 glob으로 탐색**, 하드코딩 금지. 로컬 출력 디렉터리를 `/content/out/*`로 재지정.
7. **fast-fail**: full 훈련 전 소규모(few steps / 1 epoch·작은 subset)로 GPU·메모리·파이프라인 검증 후 본 실행(쿼터 보호).

## 실행·데이터·산출물

- **실행**: `colab exec -s <name> -f nb.ipynb`(또는 `-f script.py`) — 로컬 파일을 원격 커널로 전송, 수동 업로드 불필요. 파이프도 가능: `cat cell.py | colab exec -s <name>`.
- **노트북 출력물**: `.ipynb` 실행 시 `nb_output.ipynb`가 로컬에 저장되므로 셀 출력·표·그림이 그대로 남는다(실험 기록으로 유용).
- **파일 회수(VM→로컬)**: `colab download -s <name> /content/out/model/ ./outputs/`. one-shot `colab run`은 stop 전에 회수해야 하므로, 산출물을 스크립트 말미에서 외부 스토리지로 push하거나 `--keep` 후 download → 수동 stop.
- **세션 기록**: `colab log -s <name> -o run.ipynb`로 세션 전체를 노트북/`.md`/`.jsonl`로 내보내기.
- ⚠️ **인터랙티브 명령은 에이전트 실행 금지**(TTY 요구·행): `colab repl`, `colab console`, `colab auth`, `colab drivemount`. (`repl`/`console`은 파이프 stdin은 받고 EOF에 종료.)

## 런타임·OOM·재현

- L4(sm_89, 24GB)·T4(sm_75, 16GB). P100/구형 sm_60은 FlashAttention 계열 미지원 — ModernBERT엔 부적합.
- **OOM 시**: `batch_size`↓ / `max_length`(토큰 길이)↓ / gradient accumulation로 유효 배치 유지 / gradient checkpointing / fp16·bf16 mixed precision / 입력 필드 축소(상세설명 제외 등). config로 override.
- 코드 변경 시 최신 스크립트를 재전송(`colab exec -f`는 매번 로컬 파일을 읽으므로 자동 최신).

## Safety·복구

- ⚠️ **작업 끝나면 항상 `colab stop -s <name>`** — idle VM은 컴퓨트를 소모. `colab run`(--keep 없이)은 self-clean.
- **"Session not found"/404/401 on exec** = 백엔드가 VM prune → `colab sessions` 후 `colab new` 재생성(`exec`/`repl`는 로컬 state 자동 정리).
- **멈춘/타임아웃 커널** = `colab restart-kernel -s <name>`(VM 유지, 커널만 리셋) 또는 `colab stop` 후 `colab new`.
- **keep-alive 데몬 사망**(`colab log`에 `keep_alive_stopped reason=consecutive_4xx_errors`) = 거의 항상 `colaboratory` 스코프 누락 → 위 인증 재실행.
- **잘못된 `--gpu` 값은 조용히 A100으로 fallback**(다음 단계에서 실패) → 지원값(`T4`,`L4`,`A100`,`H100`)만 사용. accelerator에 `400` = 해당 계정에 쿼터/권한 없음.
- **병렬/에이전트 격리**: 전역 `--config <path>`로 세션 state를 분리(예: `colab --config /tmp/job.json new -s job`). 데몬이 `--auth`·`--config`를 상속.

## 명령 요약

`colab sessions`(할당 목록) · `colab status -s <name>`(하드웨어·IDLE/BUSY) · `colab log -s <name>`(구조화 이벤트, 실패 진단) · `colab url -s <name>`(기존 세션에 웹 UI 부착) · `colab help [<cmd>]`.
