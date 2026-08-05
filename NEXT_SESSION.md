# 다음 세션 인계

> 읽는 순서: `PROJECT.md`(SSOT) → 이 문서. 이 문서는 **다음 세션이 이어서 할 일만** 담는다. 종결된 축의 결론·기제·실측은 `docs/experiments/`(축별 SSOT)와 `docs/adr/`(결정 경위)가 소유하고, 문서 색인은 `docs/README.md`에 있다.

## 지금 상태

**모든 실험 축이 닫혔고 배포 모델도 나왔다.** `16_01`(A.X-Encoder 4096, 정리 데이터)이 정리 test micro **0.8660**으로 착지해 기준 런 `11_01`(0.8588)을 +0.72pt 넘었다 — 실측·오류 분석은 [final-run.md](docs/experiments/final-run.md), 모델은 `ingyoun/A.X-patent-len4096-op`.

**필수 작업은 남아 있지 않다.** 아래는 예산이 남을 때만 집행하는 선택 항목이다.

## 남은 선택 항목

1. **RoBERTa·KoBERT 토큰화본 336 필터 미반영** — `ingyoun/patent-clean-text-roberta-tokenized`·`...-kobert-tokenized`가 정리 이전 상태다. 재토큰화 없이 같은 `document_id`로 필터링하면 되고, 아래 대조군이나 KoBERT 재현을 다시 돌릴 때 선행한다([ADR-0010](docs/adr/0010-data-cleaning.md)).
2. **KLUE-RoBERTa 대조군** — 성능이 아니라 주장 방어(크기 confound 제거)가 목적이다([klue-roberta.md](docs/experiments/klue-roberta.md)).

## 작업 규약

- 노트북 작성은 **지시가 있을 때만** 한다. 훈련 노트북은 `TrainConfig` 주입만 하고 로직은 `src/patent_train`에 둔다.
- 검증은 산출물에서 재현되게 한다 — 판정 기준은 서술 통계가 아니라 **결정 질문에 직접 답하는 양**(오라클·회수율·학습 곡선)으로 세운다.
- **`assert`·verify는 분석 코드의 규약이지 사용자 코드의 규약이 아니다.** Claude가 진단·분석용으로 쓰는 `scripts/*.py`는 자기 결과를 스스로 증명해야 하므로 SSOT 대조 assert(행 축·기존 실측치와의 일치)를 넣는다. **훈련 노트북과 `src/patent_train`에는 넣지 않는다** — 검증 셀·방어 assert를 요구하지 않으며, 리뷰에서도 그 부재를 지적 사항으로 올리지 않는다.
- **성능을 보고할 때는 한계를 병기한다** — 라벨 잡음에 의한 하향 편향과 평탄화된 분포(`PROJECT.md`「스코프·한계」, [confident-errors.md](docs/experiments/confident-errors.md)「보고용 요약」).
- **닫힌 갈래는 새 근거 없이 다시 제안하지 않는다**(`PROJECT.md`「닫힌 갈래」).
