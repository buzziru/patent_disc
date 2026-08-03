# 다음 세션 인계

> 읽는 순서: `PROJECT.md`(SSOT) → 이 문서. 이 문서는 **다음 세션이 이어서 할 일만** 담는다. 종결된 축의 결론·기제·실측은 `docs/experiments/`(축별 SSOT)와 `docs/adr/`(결정 경위)가 소유하고, 문서 색인은 `docs/README.md`에 있다.

## 지금 상태

**모든 실험 축이 닫혔다.** 남은 작업은 **배포 모델을 만드는 일 하나**다(`PROJECT.md` 「닫힌 갈래」에 축별 결론과 SSOT 링크가 있다).

## 다음 작업 — 배포 런 `16_01` (GPU)

`11_01`에서 `max_len`만 **4096**으로 바꾼 단일 변수 런이다. 결정·근거·실행 후 절차는 [final-run.md](docs/experiments/final-run.md)가 소유한다.

- **노트북은 준비됐다** — `notebook_output/16_01_Model_4096.ipynb`. 레시피 축(eff128 · lr 4.8e-4 · wd 0.01 · warmup 0.1 · linear · 12 epoch · early_stop 2 · seed 42)은 앵커와 동일하고, lr은 LR range test로 안전 판정이 끝났다(4096은 측정한 512·8192 사이).
- **남은 것은 배치 두 값뿐이다** — 팟에서 `probe_batches`로 `micro_batch`·`eval_micro_batch`를 실측해 채운다. `micro_batch`는 **128의 약수**여야 하고(`eff_batch % micro_batch == 0`), 512의 128(=65,536 토큰/배치) 환산으로 4096에서는 **16 부근**이 출발점이다.
  - ⚠️ **`cfg.micro_batch`를 제자리 수정하지 않는다.** `grad_accum`은 `__post_init__`에서만 유도되므로 제자리 수정하면 1로 남아 **유효 배치가 조용히 16이 된다**(lr 4.8e-4는 eff 128 전제). config 셀을 통째로 재실행하고 출력된 `grad_accum`을 확인한다.
- **비교선**: 앵커 `11_01` 0.8588(정리 test) · exp1 0.8683(정리 test 재계산). 4096은 512의 상위집합이라 **0.8588 아래로 착지하면 레시피·데이터를 의심한다.**
- **런 뒤에 할 일**: 로짓 회수 → 행 축 확인 → paired bootstrap 대조 → 길이 bin 델타로 부채 추정 검증 → 헤드라인·`PROJECT.md` 갱신([final-run.md](docs/experiments/final-run.md) 「실행 후 할 일」).

## 남은 선택 항목 (예산이 남을 때만)

1. **RoBERTa·KoBERT 토큰화본 336 필터 미반영** — `ingyoun/patent-clean-text-roberta-tokenized`·`...-kobert-tokenized`가 정리 이전 상태다. 재토큰화 없이 같은 `document_id`로 필터링하면 되고, 아래 대조군이나 KoBERT 재현을 다시 돌릴 때 선행한다([ADR-0010](docs/adr/0010-data-cleaning.md)).
2. **KLUE-RoBERTa 대조군** — 성능이 아니라 주장 방어(크기 confound 제거)가 목적이다([klue-roberta.md](docs/experiments/klue-roberta.md)).

## 함정 — 이 런에서 실제로 걸리는 것

- **팟의 `src/patent_train`을 로컬 최신본으로 교체하고 시작한다.** 볼륨·이미지에 남은 구 사본이 import되면 로컬에서 고친 코드가 반영되지 않은 채 런이 돈다. `13_02`가 이 경로로 순열 로짓을 냈다 — `runner.predict_logits`의 행 순서 방어(순차 샘플러 복원 + 반환 라벨 assert)는 로컬에 이미 있었으나 팟 사본이 구본이었다.
- **`probe_batches`는 GPU에 올린 모델을 받는다.** `load_model()` 직후 모델은 CPU에 있고(`from_pretrained`는 옮기지 않는다) `Trainer`가 `train()` 시작 시 옮기므로, `build_trainer()` 앞에서 프로브를 부르면 CPU 텐서가 flash-attn에 들어가 `NotImplementedError: ... 'flash_attn::_flash_attn_forward' ... 'CPU' backend`로 죽는다. `torch.autocast("cuda", …)`는 캐스팅 정책만 정할 뿐 텐서를 옮기지 않는다. 호출 전에 `runner.model.to("cuda")`를 둔다. 프로브 뒤 상태는 안전하다(`probe.py`가 정리하고 `lr=0.0`이라 가중치 불변).
- **`prep_cache`는 `{backbone}_len{max_len}`으로만 키잉된다.** 데이터셋 버전이 바뀌어도 볼륨에 캐시가 남아 있으면 원본 다운로드를 건너뛰어 **구 데이터로 학습된다.** `[schedule]` 출력의 step/epoch로 행 수를 역산해 확인한다(정리 데이터 = **1,576 step/epoch @ eff128**). 4096은 새 키(`axenc_len4096`)라 이번 런에는 오염 위험이 없다.
- **정리 test(11,244)와 구 test(11,271)는 다른 셋이다.** 서로 다른 test에서 잰 micro를 나란히 놓지 않는다(구 로짓의 정리 test 재계산값은 `output/headline_cleaned_test.json`).
- **Hub에 push된 훈련 repo는 커밋 메시지로 완주를 판단할 수 없다.** `push_to_hub=True`+`hub_strategy="all_checkpoints"`는 저장 시점마다 "Training in progress…"로 커밋하고, 정상 종료해도 "End of training" 커밋이 없을 수 있다. 루트 `model.safetensors`의 sha256을 마지막 `checkpoint-*/model.safetensors`와 대조하고 그 `trainer_state.json`의 `global_step`·`epoch`으로 확인한다.
- **스케줄 길이가 다른 런을 에폭 눈금으로 비교하지 않는다.** `linear`+`warmup_ratio=0.1`은 총 스텝에 비례하므로 짧은 탐색 런과 12 epoch 풀런에서 같은 "1 epoch"이 lr 궤적의 전혀 다른 자리다. 조기 곡선 대조는 **12 epoch 런끼리** 한다(감시 대조점 표는 [training-curves.md](docs/experiments/training-curves.md)).

## 작업 규약

- 노트북 작성은 **지시가 있을 때만** 한다. 훈련 노트북은 `TrainConfig` 주입만 하고 로직은 `src/patent_train`에 둔다.
- 검증은 산출물에서 재현되게 한다 — 판정 기준은 서술 통계가 아니라 **결정 질문에 직접 답하는 양**(오라클·회수율·학습 곡선)으로 세운다.
- **`assert`·verify는 분석 코드의 규약이지 사용자 코드의 규약이 아니다.** Claude가 진단·분석용으로 쓰는 `scripts/*.py`는 자기 결과를 스스로 증명해야 하므로 SSOT 대조 assert(행 축·기존 실측치와의 일치)를 넣는다. **훈련 노트북과 `src/patent_train`에는 넣지 않는다** — 검증 셀·방어 assert를 요구하지 않으며, 리뷰에서도 그 부재를 지적 사항으로 올리지 않는다.
- **닫힌 갈래는 새 근거 없이 다시 제안하지 않는다**(`PROJECT.md` 「닫힌 갈래」).
