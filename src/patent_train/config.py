"""훈련 설정 — 실험별 레시피 주입 단위.

노트북마다 흩어진 `config = {…}` 딕셔너리를 타입 있는 단일 dataclass로 모은다.
**레시피 축만** 담는다(손실·길이·배치·lr·에폭·임계값·아티팩트 식별자). 인코더·데이터셋·헤드 같은
백본 묶음은 여기 두지 않고 `backbones.BACKBONES`가 관리한다 — `backbone` 필드는 그 레지스트리 키다.
새 인코더 실험은 백본 한 줄 추가 + `backbone` 키 교체로 끝난다(config 코드 불변).

`loss`·`tag`·`repo_final`·`out_path`(실험 정체성)와 최적화 레시피(lr·batch·weight_decay·warmup·early_stop_epochs)는
기본값 없이 노트북에서 명시한다 — 튜닝 대상을 숨은 기본값으로 두면 낡은 레시피가 조용히 도는 것을 막는다.
`seed`만 단일 시드 상수라 기본을 남긴다. `grad_accum`은 배치 두 축에서 유도하고, `epochs`·`evals_per_epoch`는
짧은 탐색 런(`search=True`)에서만 기본을 유도한다(`__post_init__`) — 풀런은 `epochs`를 명시해야 한다.

**단위 규약**: 시간 축 설정은 모두 에폭 단위로 받는다(`epochs`·`evals_per_epoch`·`early_stop_epochs`).
HF Trainer가 요구하는 step·eval 횟수 환산은 `runner.build_trainer`가 맡는다.
"""

from dataclasses import dataclass, field

from .metrics import DEFAULT_TAU


@dataclass(kw_only=True)
class TrainConfig:
    # ── 실험 정체성(매번 명시) ──────────────────────────────────────────────
    loss: str                       # 손실 레지스트리 키: "focal"|"bce"|"asl"|"zlpr"
    tag: str                        # 아티팩트 접두어(metrics·logits 파일명)
    run_name: str                   # wandb 런 이름 — 조합(배치·lr·데이터 버전)이 겹치지 않게 직접 짓는다
    repo_final: str                 # 최종 모델 push 대상 HF repo
    out_path: str                   # 모델·메트릭 저장 디렉터리(/workspace 하위)
    loss_params: dict = field(default_factory=dict)   # 손실 하이퍼파라미터(예: asl의 gamma_neg)
    notebook_name: str | None = None  # 실행 노트북 파일명 → WANDB_NOTEBOOK_NAME(runner가 절대경로로 변환).
                                    # 미지정 시 wandb가 노트북명 탐지에 실패해 code saving이 꺼진다.

    # ── 백본·입력 ───────────────────────────────────────────────────────────
    backbone: str = "axenc"         # backbones.BACKBONES 키 — 인코더+데이터셋+헤드 묶음
    max_len: int = 512              # 입력 길이(레시피 축: 512 vs 8192)

    # ── 추론(체크포인트 로드) ────────────────────────────────────────────────
    checkpoint: str | None = None   # 훈련된 모델 소스(Hub repo id 또는 로컬 디렉터리). None이면 백본에서
                                    # 새 헤드를 얹는다(훈련용). 설정 시 model.build_model이 이 소스에서 로드하고
                                    # runner는 추론 형태로 동작한다(train() 거부·wandb/early stop 없음).
    splits: tuple | None = None     # 로드·절단할 split 부분집합(예: ("val", "test")). None이면 전체.
                                    # val+test 로짓만 필요할 때 train(201k행) 로드를 피한다.

    # ── 실행 모드 ────────────────────────────────────────────────────────────
    search: bool = False            # True=레시피 탐색(짧은 런) / False=최종 풀런
    epochs: int | None = None       # 풀런은 필수 입력. 탐색 런에서만 미지정 시 2로 유도
    evals_per_epoch: int | None = None   # 에폭당 eval·save 횟수. None이면 search에 따라 4/2로 유도

    # ── 최적화(노트북에서 명시 입력 — 실험이 정하는 레시피라 기본값을 숨기지 않는다) ──────────
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    early_stop_epochs: int          # 개선 없이 견디는 **에폭 수**(eval 횟수 아님).
                                    # eval 단위 patience 환산은 runner가 evals_per_epoch를 곱해 처리한다
    eff_batch: int                  # micro_batch × grad_accum
    micro_batch: int
    eval_micro_batch: int
    lr_scheduler_type: str = "linear"     # HF 스케줄러 키("linear"|"cosine"|…)
    seed: int = 42                  # 단일 시드 방법론(ADR-0011) — 실험 상수라 기본 유지

    # ── 평가·저장 ───────────────────────────────────────────────────────────
    tau: float = DEFAULT_TAU        # 멀티라벨 임계값 — 규약의 SSOT는 metrics.DEFAULT_TAU
    save_total_limit: int = 4
    hf_cache: str = "/workspace/hf_cache"
    prep_cache_root: str = "/workspace/prep_cache"
    output_dir: str = "/app/results"      # 체크포인트(휘발 가능)
    logging_dir: str = "/workspace/logs"

    # ── 유도값 ──────────────────────────────────────────────────────────────
    grad_accum: int = field(init=False)

    def __post_init__(self):
        # 나누어떨어지지 않으면 실제 유효 배치가 eff_batch와 조용히 달라진다(steps_per_epoch 계산도 어긋난다).
        if self.micro_batch <= 0 or self.eff_batch % self.micro_batch:
            raise ValueError(
                f"eff_batch({self.eff_batch})는 micro_batch({self.micro_batch})의 양의 배수여야 한다"
            )
        self.grad_accum = self.eff_batch // self.micro_batch
        # 풀런 길이는 실험이 정하는 축이라 유도하지 않는다. 짧은 탐색 런만 관례값(2)을 쓴다.
        if self.epochs is None:
            if not self.search:
                raise ValueError("풀런(search=False)은 epochs를 명시해야 한다")
            self.epochs = 2
        if self.evals_per_epoch is None:
            self.evals_per_epoch = 4 if self.search else 2

    @classmethod
    def for_inference(cls, *, tag: str, checkpoint: str, out_path: str, workspace: str,
                      max_len: int = 512, eval_micro_batch: int = 512,
                      backbone: str = "axenc", tau: float = DEFAULT_TAU,
                      splits: tuple | None = ("val", "test")) -> "TrainConfig":
        """훈련된 체크포인트로 로짓만 덤프하는 추론 전용 config.

        훈련 레시피 필드(lr·batch·warmup·early_stop 등)는 추론에서 무의미하므로 여기서 자리값으로
        채운다 — 풀런 경로는 그대로 명시를 강제하고(낡은 레시피 방지), 자리값 노출은 이 팩토리로 격리한다.
        `workspace` 하나로 캐시·로그 경로를 유도한다(팟 `/workspace`, Colab `/content`). `run_name`·
        `repo_final`은 추론에서 쓰지 않으므로(wandb·push 비활성) tag에서 파생한 표식만 둔다.
        """
        return cls(
            loss="focal", tag=tag, run_name=f"{tag}-infer", repo_final=tag, out_path=out_path,
            checkpoint=checkpoint, splits=splits, backbone=backbone, max_len=max_len,
            eval_micro_batch=eval_micro_batch, micro_batch=eval_micro_batch, eff_batch=eval_micro_batch,
            learning_rate=0.0, weight_decay=0.0, warmup_ratio=0.0, early_stop_epochs=1, epochs=1,
            tau=tau, hf_cache=f"{workspace}/hf_cache", prep_cache_root=f"{workspace}/prep_cache",
            output_dir=f"{workspace}/results", logging_dir=f"{workspace}/logs",
        )
