# Google Colab — 헤드리스 GPU 훈련 (`colab` CLI)

> **L4 24GB** GPU에서 훈련을 돌리는 경로. 노트북 UI 업로드가 아니라 **`colab` CLI**로 원격 VM에서 로컬 `.ipynb`/`.py`를 실행한다(`colab exec -f`는 둘 다 지원) = 에이전트로 자동화 가능. 이 프로젝트는 **`.ipynb` 중심**.
> 대안 경로 = [Lightning Job](./lightning-jobs.md).

## 언제 Colab을 쓰나

- 단일 GPU로 충분한 인코더 훈련의 **기본 경로**. 장문 인코더(`skt/A.X-Encoder-base`, ModernBERT 계열, 최대 16,384 토큰)는 512 토큰 대비 메모리를 많이 써 L4 24GB가 안전.
- ⚠️ accelerator는 tier-gated — L4 쿼터가 없으면 `--gpu T4`(16GB) fallback 또는 Lightning L4 Job으로.

## 설치·인증 (1회)

- **설치**: `uv tool install google-colab-cli`로 설치 완료(`colab.exe`, v0.6.0).
  - ⚠️ **Windows 패치 필요**: 0.6.0은 `colab_cli/console.py` 최상단에서 Unix 전용 `termios`/`tty`를 무조건 import해 **모든 명령이 `ModuleNotFoundError: termios`로 죽는다**(대화형 `repl`/`console` 전용 코드). 해당 import를 `try/except ImportError`로 가드하면 비대화형 명령(`new`/`exec`/`stop`/`url`/`log`/`sessions`)이 정상 동작한다 — 이미 적용함. **`uv tool upgrade`/재설치 시 패치가 지워지므로 재적용** 필요(파일: `%APPDATA%\uv\tools\google-colab-cli\Lib\site-packages\colab_cli\console.py`).
- **인증 = ADC**(Application Default Credentials, 헤드리스 기본). Colab 백엔드는 **4개 스코프 전부** 필요 — 빠지면 401/403:
  ```bash
  gcloud auth application-default login \
    --scopes=openid,\
  https://www.googleapis.com/auth/cloud-platform,\
  https://www.googleapis.com/auth/userinfo.email,\
  https://www.googleapis.com/auth/colaboratory
  ```
  각 스코프 이유: `userinfo.email`(세션 백엔드 `colab.research.google.com` — assign/unassign/sessions **및 keep-alive**, 없으면 401) · `colaboratory`(2026-06-15 이후 keep-alive는 이 스코프·`colab.pa.googleapis.com` RPC를 쓰지 않음 — forward-compat·기타 Colab 기능용으로 유지) · `openid`+`cloud-platform`(gcloud가 요구).
  - ⚠️ 이 `gcloud ...`는 **브라우저 동의가 필요** → 에이전트가 아니라 **사용자가 직접**(세션에서 `! gcloud ...`) 실행한다.
- **인증 검증(1샷)**: `colab whoami`(활성 이메일·스코프·만료 출력, hidden 디버그 명령) 또는 `colab sessions`. assign/keep-alive가 `colab.research.google.com`에 401이면 거의 항상 `userinfo.email` 스코프 누락.
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
   - 로컬 환경은 **Python 3.12 + `uv.lock`**(현재 `torch 2.12.1+cu130`)으로 고정 — Colab과 버전을 맞추기 위함. VM에서도 같은 버전을 설치해 재현성 확보.
   - **transformers 실측(2026-07-13)**: **Colab = `5.12.1`**, 로컬 `.venv` = `5.13.0`(둘 다 5.x 계열이라 동작 regime은 동일 — 아래 「ModernBERT + FA2」 참조). 훈련 런은 **버전을 pin**해 regime을 고정할 것.
   - ⚠️ **Colab의 실제 Python/torch를 확인**(`import sys; print(sys.version)`, `torch.__version__`)해 3.12 가정과 cu13 드라이버 호환을 검증할 것. 어긋나면 `pyproject.toml`의 `requires-python`을 Colab 버전에 맞춘다.
4. **VM측 시크릿(HF 토큰 등)**: `google.colab.userdata.get()`는 **헤드리스에서 못 쓴다** — 시크릿의 *notebook access* 토글이 UI 전용이고, `colab exec` 세션엔 권한을 부여할 노트북 ID가 없어 `NotebookAccessError`가 난다(웹 확인: colabtools #4220 / colab-vscode #215). **대안 = 커널 env 주입**: 커널 state가 `exec` 간 유지되므로, 노트북 실행 **전에** 토큰을 커널에 심는다(노트북엔 하드코딩·userdata 셀 금지).
   ```bash
   echo "import os; os.environ['HF_TOKEN']='<hf_token>'" | colab exec -s <name>
   ```
   이후 `load_dataset`/`huggingface_hub`가 `HF_TOKEN`을 자동 사용. 토큰 문자열은 로컬 `.env`에서 읽어 넘기고 **커밋하지 않는다**. (`colab drivemount`는 인터랙티브라 헤드리스 금지.)
   - ⚠️ **PowerShell 파이프는 UTF-16/BOM을 붙여** `colab exec`에서 `SyntaxError`를 내고, **traceback이 소스 라인(=토큰)을 그대로 출력해 시크릿이 노출된다**(2026-07-08 실제 발생 → 토큰 재발급함). 코드·시크릿을 `colab exec` stdin으로 보낼 땐 **Bash `printf`**(무 BOM)나 UTF-8(no BOM) 임시파일 `-f`를 쓴다. 노출됐으면 **즉시 해당 토큰 폐기·재발급**.
5. **데이터 반입 = HF Hub pull(streaming)**. VM은 원격·휘발이라 로컬 `data/`를 못 본다 → 노트북 안에서 `load_dataset("<user>/patent-tokenized", split=..., streaming=True)`로 받는다. **로컬 data를 `colab upload`하지 않는다**(419MB를 매 세션 재업로드해야 하므로 비효율). `colab drivemount`는 인터랙티브라 헤드리스 금지. 설계: [`data-pipeline.md`](../data/data-pipeline.md).
6. **경로는 glob으로 탐색**, 하드코딩 금지. 로컬 출력 디렉터리를 `/content/out/*`로 재지정.
7. **fast-fail**: full 훈련 전 소규모(few steps / 1 epoch·작은 subset)로 GPU·메모리·파이프라인 검증 후 본 실행(쿼터 보호).

## 실행·데이터·산출물

- **실행**: `colab exec -s <name> -f nb.ipynb`(또는 `-f script.py`) — 로컬 파일을 원격 커널로 전송, 수동 업로드 불필요. 파이프도 가능: `cat cell.py | colab exec -s <name>`.
- **노트북 출력물**: `.ipynb` 실행 시 `nb_output.ipynb`가 로컬에 저장되므로 셀 출력·표·그림이 그대로 남는다(실험 기록으로 유용).
- **파일 회수(VM→로컬)**: `colab download -s <name> /content/out/model/ ./outputs/`. one-shot `colab run`은 stop 전에 회수해야 하므로, 산출물을 스크립트 말미에서 외부 스토리지로 push하거나 `--keep` 후 download → 수동 stop.
- **세션 기록**: `colab log -s <name> -o run.ipynb`로 세션 전체를 노트북/`.md`/`.jsonl`로 내보내기.
- ⚠️ **인터랙티브 명령은 에이전트 실행 금지**(TTY 요구·행): `colab repl`, `colab console`, `colab auth`, `colab drivemount`. (`repl`/`console`은 파이프 stdin은 받고 EOF에 종료.)

## 체크포인트-resume (VM 회수·장시간 런 대비)

Colab VM은 휘발성 — 로컬 keep-alive가 끊겨 VM이 회수되면 `/content`(체크포인트 포함)가 전부 사라진다. 장시간 런(예: KoBERT 12에폭 ≈ 12~17h)이거나 로컬 PC를 계속 못 켜두는 경우, **체크포인트를 HF Hub에 저장**해 새 VM에서 이어받는다.

- **저장 설정**(`TrainingArguments`): `push_to_hub=True` + `hub_model_id="<user>/<repo>"` + `hub_strategy="checkpoint"`. ⚠️ **`push_to_hub=True`가 없으면 `hub_strategy`/`hub_model_id`만으로는 아무것도 push되지 않는다**(체크포인트가 휘발 VM에만 남아 resume 불가). WRITE 토큰을 `trainer.train()` 전에 커널 env로 주입해야 레포 생성 단계에서 안 죽는다.
- **동작**: 매 저장(`save_strategy="epoch"`)마다 최신 체크포인트(모델+옵티마이저+스케줄러+RNG+step)를 레포의 `last-checkpoint/`로 백그라운드 업로드. 체크포인트 1개 ≈ ~1GB(AdamW state 포함). 복원 단위가 에폭이라 중단 시 최대 1에폭(~60~85분) 손실.
- **재개**(새 VM): 모델/트레이너 정의까지 실행 → Hub에서 last-checkpoint 내려받아 `resume_from_checkpoint`:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("<user>/<repo>", local_dir="/content/results", allow_patterns="last-checkpoint/*")
  trainer.train(resume_from_checkpoint="/content/results/last-checkpoint")
  ```
  seed·데이터·config가 고정이면 끊긴 지점부터 연속 재현. **첫 실행은 `trainer.train()`**(빈 output_dir), **재개만** 위 절차.
- **VM이 아직 살아있는 짧은 중단**이면 resume 불필요 — `colab status`/`colab log`로 확인하거나 `colab url`로 재접속(커널이 계속 훈련 중).
- Google Drive는 `colab drivemount`가 인터랙티브라 헤드리스 durable store로 부적합 → **Hub가 유일한 현실적 선택**.

## ⚠️ 헤드리스 GPU 장시간 런의 한계 — L4 ~20분 회수 (실측 2026-07-08)

KoBERT 9.6h 런을 헤드리스 `colab exec`로 **2회** 시도 → 둘 다 **~20분(23분·19분)에 `session_terminated`**. 로그는 매번 `session_created → KEEP started → session_terminated`뿐, **`keep_alive_stopped` 이벤트 없음** = keep-alive 데몬은 죽지 않았고 **Colab이 L4 VM을 서버측에서 회수**한 것(데몬의 다음 ping은 404 → exec "Connection was lost"). 즉 **TFE keep-alive ping이 GPU 회수를 막지 못한다**(idle 타이머는 갱신해도 Colab의 GPU 리스/프리엠션은 별개).
- **결론**: 헤드리스 CLI로 **L4 장시간(수 시간) 무인 훈련은 비현실적**. epoch 저장이면 회수 시 매번 전량 유실.
- **대안**: (a) 활성 브라우저 탭(`colab url`)을 열어두면 프런트엔드 heartbeat로 회수를 늦출 수 있음(**미검증**), (b) `save_steps`(<15분) + 잦은 Hub 체크포인트로 resume 반복(회수 ~20분마다 → 오버헤드 큼), (c) **장시간 런은 [Lightning Job](./lightning-jobs.md) 등 무인 훈련 전용 플랫폼**으로 전환. 짧은 검증·소규모 잡엔 Colab CLI가 여전히 유효.

## 관측(observability) — headless 런에서 진행 보기

⚠️ **headless `colab exec`는 커널 1개를 훈련 내내 독점**한다 → `colab url`로 웹 UI를 붙여도 (a) tqdm 진행바는 exec 채널로만 흐르고, (b) 웹에서 새 셀을 실행하면 훈련 셀 뒤에 **큐잉되어 멈춘다**. 즉 **웹 step-단위 라이브 뷰가 없다**(2026-07-08 KoBERT 런에서 확인). 관측은 아래로 설계한다:

- **W&B(권장, 헤드리스 최적)**: `TrainingArguments(report_to="wandb", run_name=...)`. `WANDB_API_KEY`를 HF_TOKEN과 동일하게 커널 env로 주입(Bash `printf` 파이프). 훈련 프로세스가 **HTTP로 wandb.ai에 직접 푸시**하므로 커널 독점과 무관하게 브라우저 어디서든 loss·P@1/3/5·lr을 실시간 관측. VM에 `wandb` 필요(대개 프리인스톨, 없으면 `colab install -s <name> wandb`).
- **TensorBoard**: HF Trainer가 `logging_dir`에 이벤트를 쓰지만, 헤드리스에선 TB 서버·포트 터널을 커널에서 못 띄워(독점) 실시간 관측이 번거롭다 → **W&B 우선**. 로컬 회수 후 사후 분석용으론 유효.
- **HF Hub 레포**(`push_to_hub=True`): 에폭마다 `last-checkpoint/` 커밋 → **에폭 단위 progress**를 브라우저 Commits 탭에서 병행 확인.
- **CLI 측**: `colab log -s <name>`(구조화 이벤트) / exec 스트림(에이전트가 중계).

## 런타임·OOM·재현

- L4(sm_89, 24GB)·T4(sm_75, 16GB). P100/구형 sm_60은 FlashAttention 계열 미지원 — ModernBERT엔 부적합.
- **실측 처리량**(참고): KoBERT(`monologg/kobert`, 92M) batch 8 · seq 512 · fp16 기준 **L4 ≈ 8.5~8.8 it/s(≈48분/epoch)**, 2080 Ti ≈ 4.67 it/s(가공업체 실측). 12에폭(≈30.3만 step) 풀런 ETA L4 **~9.6h**(early stop로 단축).
- **OOM 시**: `batch_size`↓ / `max_length`(토큰 길이)↓ / gradient accumulation로 유효 배치 유지 / gradient checkpointing / fp16·bf16 mixed precision / 입력 필드 축소(상세설명 제외 등). config로 override.
- 코드 변경 시 최신 스크립트를 재전송(`colab exec -f`는 매번 로컬 파일을 읽으므로 자동 최신).

### ModernBERT + FlashAttention-2 — unpadding 범위는 transformers 버전에 따라 뒤집힌다

장문 배칭 전략(특히 `group_by_length`)의 타당성이 여기에 걸려 있어 실측으로 확인했다(`transformers 5.13.0` 소스 직독).

- **5.x 계열(현재 Colab 5.12.1 / 로컬 5.13.0)**: ModernBERT가 리팩터링돼 **모델 레벨 unpadding이 없다**. `ModernBertModel.forward`는 `hidden_states`를 `[batch, seq, hidden]` **패딩 상태로** embedding → 22개 layer → final_norm까지 흘린다. unpadding은 **FA2 attention 커널 내부에서만** 일어난다(`modeling_flash_attention_utils`의 `unpad_input` → `flash_attn_varlen_func` → `pad_input`). `Wqkv`조차 패딩된 텐서에 적용된다.
  - → **attention만 패딩이 공짜**이고, embedding·projection·**MLP**·norm·residual은 패딩된 채 계산된다. ModernBERT는 `local_attention: 128` + `global_attn_every_n_layers: 3`이라 attention이 원래 싸므로 **FLOPs·활성 메모리는 MLP/projection이 지배** → **패딩 낭비가 실재**한다.
  - → **`group_by_length=True`가 유효하다.** 게다가 패딩 구간 메모리는 `batch × max_len_in_batch`라, random 샘플링도 배치에 장문 하나만 섞이면 같은 peak를 친다 → **group_by_length가 peak를 올리지 않고 평균만 낮춘다.**
- **4.4x~4.5x 계열**: `_unpad_modernbert_input`/`_pad_modernbert_output`이 `forward` 최상단에 있어 **전 forward가 flat unpadded 시퀀스**로 돈다. 이 regime에선 패딩이 전 구간 공짜라 **`group_by_length`는 무의미**하고, all-long 배치가 토큰 합 peak를 올려 오히려 해롭다.
- ⚠️ 즉 **버전을 pin하지 않으면 어느 regime인지 모른 채 장시간 런을 태우게 된다.**
- 부수 확인: `classifier_pooling: "mean"`은 5.x에서 **masked mean**(attention_mask로 나눔)이라 패딩이 pooling을 오염시키지 않는다. `reference_compile`은 5.x에서 **제거된 dead key** → torch.compile 재컴파일 이슈 없음.

## 추론 전용 실행 (로짓 재덤프)

훈련된 모델을 Hub에서 불러 로짓만 다시 덤프하는 경로(예: 순열 로짓 재생성). `src/patent_train`을 그대로 재사용하되 `TrainConfig.for_inference`로 훈련 부속(wandb·early stop·save·train_dataset)을 끈다. 예시 노트북 `notebook/11_03_Redump_Logits.ipynb`는 **Colab 웹에서 직접 실행**하는 형태다(colab-cli 헤드리스 아님).

- **코드 반입 = Drive 마운트 + `copytree`**: VM에 코드를 매번 업로드하면 느리다. `patent_train`을 Drive `MyDrive/patent_disc/src/patent_train`에 올려두고 `drive.mount`한 뒤, **Drive에서 로컬 `/content/src`로 `shutil.copytree`(`__pycache__` 제외)** 하고 `sys.path.insert(0, "/content/src")`로 그 사본을 import한다 — Drive 직접 import는 매 파일 접근이 네트워크라 느리다. `print(patent_train.__file__)`이 `/content/src/...`인지 확인한다.
- **버전 pin**: `transformers`를 로컬 `.venv`·훈련 이미지와 같은 값으로 고정한다(`5.13.0`) — ModernBERT+FA2 regime이 버전에 따라 뒤집히고(위 「ModernBERT + FA2」), 추론 regime이 훈련과 달라지면 안 된다. flash-attn 프리빌트 휠도 함께 설치.
- **시크릿 = `userdata`**: 웹 실행이라 헤드리스 `userdata` 제약이 없다(09_00과 동일). `HF_TOKEN`을 `userdata.get("HUGGINGFACEHUB_API_TOKEN")`으로 주입.
- **모델·데이터셋 = Hub 다운로드**: 팟이 꺼져 있으면 로컬 체크포인트가 없다. `checkpoint`에 push된 repo id를 주면 `build_model`이 그 소스에서 헤드까지 복원한다(Hub id·로컬 디렉터리 동일 해석). Colab의 `HF_HOME`은 휘발이라 매 VM 새로 받는다.
- **산출물 = Drive 저장**(09_00과 동일): `predict_logits(..., out_dir="{DRIVE}/output")`로 로짓을 Drive에 직접 쓴다. 다운로드는 사용자가 Drive에서 한다.
- **split 한정**: `splits=("val","test")`로 train(201k행) 로드를 피한다. 추론 경로는 on-disk prep 캐시를 우회한다(휘발 VM에서 이득 없고, 부분 캐시가 훗날 전체 런에 재사용되는 사고를 원천 차단).
- **행 순서 가드**: `predict_logits`가 덤프 동안 `train_sampling_strategy`를 `"sequential"`로 되돌리고(→ `_get_eval_sampler`가 `SequentialSampler`), 반환 라벨을 데이터셋 라벨과 대조하는 assert를 건다. 함께 나온 predict 지표(`runner.metrics[split]`)의 micro를 훈련 SSOT와 대조해 **모델을 제대로 불렀는지** 확인한다.

```python
from google.colab import drive, userdata; drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/patent_disc"
shutil.copytree(f"{DRIVE}/src/patent_train", "/content/src/patent_train",
                dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
sys.path.insert(0, "/content/src")
os.environ["HF_TOKEN"] = userdata.get("HUGGINGFACEHUB_API_TOKEN")

cfg = TrainConfig.for_inference(
    tag="modernbert-patent-len512-b128", checkpoint="ingyoun/A.X-patent-len512-b128",
    out_path="/content/output/redump", workspace="/content",
    max_len=512, eval_micro_batch=512, splits=("val", "test"))
runner = TrainingRunner(cfg)
runner.load_data(); runner.prepare_data(); runner.load_model()   # 단계별(11_01과 동일) — setup() 축약 대신
runner.build_trainer()
runner.predict_logits("val", out_dir=f"{DRIVE}/output")        # → Drive/output/logits_{tag}_{split}.npy
runner.predict_logits("test", out_dir=f"{DRIVE}/output")
```

## Safety·복구

- ⚠️ **작업 끝나면 항상 `colab stop -s <name>`** — idle VM은 컴퓨트를 소모. `colab run`(--keep 없이)은 self-clean.
- **"Session not found"/404/401 on exec** = 백엔드가 VM prune → `colab sessions` 후 `colab new` 재생성(`exec`/`repl`는 로컬 state 자동 정리).
- **멈춘/타임아웃 커널** = `colab restart-kernel -s <name>`(VM 유지, 커널만 리셋) 또는 `colab stop` 후 `colab new`.
- **keep-alive 데몬 사망**(`colab log`에 `keep_alive_stopped reason=consecutive_4xx_errors`) = keep-alive TFE ping(`colab.research.google.com/tun/m/…/keep-alive/`)이 4xx — 거의 항상 `userinfo.email` 스코프 누락 → 위 인증 재실행. (구 문서의 `colab.pa.googleapis.com` 403·`colaboratory` 누락 진단은 폐기: keep-alive는 2026-06-15부터 그 RPC를 쓰지 않는다.)
- **잘못된 `--gpu` 값은 조용히 A100으로 fallback**(다음 단계에서 실패) → 지원값(`T4`,`L4`,`G4`,`A100`,`H100`)만 사용. accelerator에 `400` = 해당 계정에 쿼터/권한 없음.
- **병렬/에이전트 격리**: 전역 `--config <path>`로 세션 state를 분리(예: `colab --config /tmp/job.json new -s job`). 데몬이 `--auth`·`--config`를 상속.

## 명령 요약

`colab sessions`(할당 목록) · `colab status -s <name>`(하드웨어·IDLE/BUSY) · `colab log -s <name>`(구조화 이벤트, 실패 진단) · `colab url -s <name>`(기존 세션에 웹 UI 부착) · `colab help [<cmd>]`.
