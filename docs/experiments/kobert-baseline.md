# KoBERT baseline 재현 — 비교 기준점 수립

> **목적**: 공식 `0.8249`를 그대로 쓰지 않고, **KoBERT baseline을 직접 재현**해 장문 인코더(`skt/A.X-Encoder-base`)와의 공정 비교선을 만든다.  
> 참고 코드(업체 제공): `../../소스코드/{01.data_processing, 02_training_bert, 03_model_test}.ipynb`.

## 원칙

- **동일 test set 원칙**: KoBERT baseline과 이후 모든 실험(A.X-Encoder, KLUE-RoBERTa)은 **같은 고정 test split**에서 평가한다. `ingyoun/patent-clean-text`의 `test`(11,217건)가 그 기준. baseline의 원본 24,525 test는 데이터 부재로 재현 불가하므로 우리 것으로 통일한다.
- **재현 충실도**: 학습 레시피(입력 필드 조합·토크나이저·손실·하이퍼파라미터)를 원본에 최대한 맞춰 baseline의 절대 수치를 정직하게 대변한다. 인프라(apex→torch AMP 등)만 현대화.
- **지표 일치**: baseline `0.8249`는 **top-1 예측 weighted-F1**이다 → 재현도 동일 계산으로 headline을 낸다.

## baseline 프로토콜 (소스코드 실측 추출)


| 항목          | 원본 값                                                                                          | 출처                                                  |
| ----------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| 모델          | `monologg/kobert` (KoBERT, 92M, 512 토큰)                                                       | `02` config `model_name`                            |
| 입력 텍스트      | `ipc_main + " " + invention_title + " " + abstract + " " + claims` (공백 join)                  | `01` cell-15/17/19                                  |
| 최대 길이       | 512 (`[CLS]` + 토큰 + `[SEP]`, post-pad value=1)                                                | `01` `convert`/`text_to_loader`                     |
| 레이블         | 문서별 다중-핫 188 (float, 다중 1 허용)                                                                 | `01` cell-9, `02` labels `[N,188]`                  |
| 분류 헤드       | BERT `pooled_output` → `Dropout(0.5)` → `Linear(768,188)`; forward는 **logits 반환**(sigmoid 없음) | `02`/`03` `BertForMultiLabelSequenceClassification` |
| 손실          | `FocalLoss(alpha=0.25, gamma=2)` on `BCEWithLogits` (smooth=0)                                | `02` cell-10/11                                     |
| 옵티마이저       | `AdamW(lr=3e-5, weight_decay=0.01, eps=1e-8, correct_bias=False)`                             | `02` cell-11                                        |
| 스케줄러        | linear warmup, `num_warmup_steps = total_step/10`                                             | `02` cell-11                                        |
| 배치/에폭       | batch 8, epochs 12, early_stop 5(개선 없으면 중단)                                                   | `02` config                                         |
| AMP         | apex O1 (레거시) → **torch.cuda.amp로 대체**                                                        | `02` cell-11                                        |
| **평가**      | **top-1**(`argmax` 1개) 예측 vs 단일화 정답 → sklearn `f1_score` micro/macro/**weighted**             | `03` cell-8~11                                      |
| baseline 실측 | Micro 0.8261 / Macro 0.8038 / **weighted 0.8249** (원본 24,525 test)                            | `03` cell-11                                        |


> ⚠️ 원본은 학습 중 모니터링에 `P@1/3/5`(`ms_get_p_5`, `topk(logit,5)`)도 계산한다(`02` cell-12). headline은 어디까지나 `03`의 top-1 weighted-F1.

## 노트북 작성 절차

훈련은 **외부 GPU 잡**(로컬은 CPU 전용). KoBERT는 512/소형이라 **Colab L4로 충분** → `[../infra/colab-jobs.md](../infra/colab-jobs.md)`. 데이터는 HF Hub streaming(`[../data/data-pipeline.md](../data/data-pipeline.md)` Layer 2).

1. **로드**: `load_dataset("ingyoun/patent-clean-text", split=...)` (train/validation/test).
2. **입력 조합**: `build_input = ipc_main + title + abstract + claims`(공백 join) — 원본과 동일하게 4필드 모두 사용(충실도). 빈 필드는 skip.
3. **토크나이즈**: KoBERT 토크나이저, `max_length=512`, truncation. 다중-핫 라벨(188, float) 생성 → `labels`.
4. **모델**: 원본 헤드를 재현(`pooled_output → Dropout(0.5) → Linear(188)`), 또는 `AutoModelForSequenceClassification(num_labels=188, problem_type="multi_label_classification", classifier_dropout=0.5)` + **custom FocalLoss**로 대체(편차는 문서에 기록).
5. **학습**: 위 표의 하이퍼파라미터. `Trainer`면 `compute_loss` 오버라이드로 FocalLoss 주입, streaming이면 `max_steps` 명시(`[../data/data-pipeline.md](../data/data-pipeline.md)` Streaming 주의).
6. **평가(고정 test)**: 아래 top-1 weighted-F1으로 headline 산출 + micro/macro·P@1/3/5 병기.

```python
import numpy as np
from sklearn.metrics import f1_score, precision_score

def evaluate_topk(logits, multihot):          # logits/multihot: [N,188]
    pred_top1 = logits.argmax(axis=1)                       # top-1 예측
    gold_top1 = multihot.argmax(axis=1)                     # 정답 단일화(원본 LabelBinarizer.inverse_transform과 등가)
    out = {
        "weighted_f1": f1_score(gold_top1, pred_top1, average="weighted"),  # ← baseline headline과 동일
        "micro_f1":    f1_score(gold_top1, pred_top1, average="micro"),
        "macro_f1":    f1_score(gold_top1, pred_top1, average="macro"),
    }
    order = np.argsort(-logits, axis=1)                     # P@k (멀티레이블 참고 지표)
    for k in (1, 3, 5):
        topk = order[:, :k]
        hit = np.take_along_axis(multihot, topk, axis=1).sum(1)
        out[f"p@{k}"] = float((hit / np.clip(multihot.sum(1), 1, k)).mean())
    return out
```

## 결정·리스크

- **KoBERT 토크나이저 호환성**(가장 큰 리스크): 원본은 `kobert_transformers.get_tokenizer()`/`monologg/kobert`를 썼다. 최신 `transformers`에서 KoBERT 토크나이저(SentencePiece 커스텀)는 로드가 까다로울 수 있다 → **노트북 첫 셀에서 토크나이저 로드·인코딩을 먼저 검증**하고, 안 되면 `skt/kobert-base-v1` 등 대안을 기록. `uv add`로 의존성 고정.
- **입력 필드 충실도**: baseline은 `ipc_main`(IPC 코드)까지 입력에 넣었다. 재현 땐 동일하게 넣되, 우리 본 실험의 필드 조합과는 별개 변수임을 명시.
- **손실/헤드 편차**: HF 기본 경로로 대체하면 dropout·pooling 세부가 원본과 다를 수 있다 → 선택 시 편차를 이 문서에 남긴다.
- **top-1 평가의 의미**: 훈련은 멀티레이블이나 headline은 top-1 단일 정확도 성격. 우리 멀티레이블 목표 지표(micro/macro-F1@threshold)와 구분해 해석.

## 성공 기준

우리 고정 test(11,217) 위에서 **KoBERT weighted-F1(top-1)** 수치를 산출해 이 문서 하단 표에 기록한다. 이 값이 A.X-Encoder·KLUE-RoBERTa 비교의 **기준점**이 된다.


| 실험          | 입력 필드                | max_len | weighted-F1 | micro-F1 | macro-F1 | P@1/3/5 | 비고  |
| ----------- | -------------------- | ------- | ----------- | -------- | -------- | ------- | --- |
| KoBERT (재현) | ipc+title+abs+claims | 512     | *TBD*       |          |          |         | 기준점 |


