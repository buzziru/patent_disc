# KoBERT baseline 재현 — 기준선 수립과 결과 회고

> **목적**: 공식 `0.8249`를 그대로 쓰지 않고 **KoBERT baseline을 직접 재현**해, 장문 인코더(`skt/A.X-Encoder-base`)와의 공정 비교선을 세운다. 이 문서는 재현 **원칙**을 남기고, 실제 실행 **구성·결과**를 기록하며, 원본과의 편차를 **회고·분석**한다.
> 참고 코드(업체 제공): `{01.data_processing, 02_training_bert, 03_model_test}.ipynb`. 재현 노트북: `notebook/03_Baseline_Train_KoBERT.ipynb`, 실행 결과: `notebook_output/03_Baseline_Train_KoBERT_colab_output.ipynb`.

## 재현 원칙

- **동일 test set 원칙**: KoBERT baseline과 이후 모든 실험(A.X-Encoder, KLUE-RoBERTa)은 **같은 고정 test split**에서 평가한다. baseline 원본 분할은 `documentId`가 train·val 양쪽에 존재하는 누수 위험이 있어, **누수 없이 재분할한 split**으로 통일한다 — `document_id` 단위로 **train 201,895 / val 11,162 / test 11,271**. KoBERT baseline은 이 split을 KoBERT 토크나이저로 사전토큰화한 `ingyoun/patent-clean-text-kobert-tokenized`(컬럼 `input_ids`/`attention_mask`/`labels`)를 소비한다.
- **재현 충실도**: 학습 레시피(입력 필드 조합·토크나이저·손실·하이퍼파라미터)를 원본에 최대한 맞춰 baseline의 절대 수치를 정직하게 대변한다. 인프라(apex→torch AMP, 수동 루프→`Trainer`)만 현대화한다.
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
| AMP         | apex O1 (레거시)                                                                                 | `02` cell-11                                        |
| **평가**      | **top-1**(`argmax` 1개) 예측 vs 단일화 정답 → sklearn `f1_score` micro/macro/**weighted**             | `03` cell-8~11                                      |
| baseline 실측 | Micro 0.8261 / Macro 0.8038 / **weighted 0.8249** (원본 24,525 test)                            | `03` cell-11                                        |


> ⚠️ 원본은 학습 중 모니터링에 `P@1/3/5`(`ms_get_p_5`, `topk(logit,5)`)도 계산한다(`02` cell-12). headline은 어디까지나 `03`의 top-1 weighted-F1.

## 재현 구성 (실제 실행)

- **데이터**: `load_dataset("ingyoun/patent-clean-text-kobert-tokenized")` (train 201,895 / val 11,162 / test 11,271). 사전토큰화 완료본을 그대로 소비 — 입력 조합·토크나이즈는 상류 전처리에서 1회 수행하고 학습 노트북은 재실행하지 않는다(토크나이저는 패딩 용도로만 로드).
- **모델**: `AutoModelForSequenceClassification(monologg/kobert, num_labels=188, problem_type="multi_label_classification", classifier_dropout=0.5)`. 원본 커스텀 헤드(`pooled_output → Dropout(0.5) → Linear(188)`)와 등가 경로.
- **손실**: `FocalTrainer.compute_loss`에 커스텀 `FocalLoss(alpha=0.25, gamma=2)` 주입 — 원본과 동일 값.
- **하이퍼파라미터**: lr 3e-5, batch 8, epochs 12, weight_decay 0.01, warmup_ratio 0.1, linear scheduler, fp16, seed 42.
- **학습 관리**: `load_best_model_at_end=True`, `metric_for_best_model="weighted_f1"`, `EarlyStoppingCallback(patience=5)`. 학습은 중간 중단 후 체크포인트에서 재개해 **12에폭 완주**(global_step 302,844).
- **평가 지표** — headline은 top-1 weighted-F1, 참고로 micro/macro·P@1/3/5 병기:

```python
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

### 원본 대비 편차


| 항목                  | 원본                                                         | 재현                                                           | 성격                |
| ------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ | ----------------- |
| 데이터 분할              | 누수 위험(문서 중복) 24,525 test                                   | 누수 제거 재분할 11,271 test                                        | **의도 — 핵심 차별점**   |
| 학습 루프               | 수동 루프                                                      | HF `Trainer`(`FocalTrainer`)                                 | 의도(현대화)           |
| AMP                 | apex O1                                                    | torch fp16                                                   | 의도(현대화)           |
| 모델 선택               | `best_loss < valid_loss or best_p3 < ...` 저장(`02` cell-13) | best `weighted_f1` + EarlyStopping                           | 의도(개선)            |
| 분류 헤드               | 커스텀 `BertForMultiLabel...`                                 | `AutoModelForSequenceClassification(classifier_dropout=0.5)` | 등가                |
| 손실 alpha            | `FocalLoss(alpha=0.25)`                                    | `FocalLoss(alpha=0.25)`                                      | **동일**            |
| Adam `correct_bias` | `False`                                                    | `True`(HF 기본)                                                | 비의도 편차(미세)        |
| 정답 단일화              | `MultiLabelBinarizer.inverse_transform`                    | `multihot.argmax`                                            | 등가(다중라벨 문서서 미세 차) |


## 결과

고정 test(11,271, 누수 제거) 위 실측:


| 실험          | 입력 필드                | max_len | weighted-F1 | micro-F1 | macro-F1 | P@1/3/5               | 비고  |
| ----------- | -------------------- | ------- | ----------- | -------- | -------- | --------------------- | --- |
| KoBERT (재현) | ipc+title+abs+claims | 512     | **0.8148**  | 0.8147   | 0.7870   | 0.894 / 0.962 / 0.979 | 기준점 |


**이 `0.8148`이 A.X-Encoder·KLUE-RoBERTa 비교의 기준점이다.** (참고: 원본 `0.8249`는 누수 위험이 있는 다른 test 24,525건 위 수치 → 직접 비교 대상 아님)

## 회고·분석

- `**0.8148`을 `0.8249`와 직접 비교하지 않는다.** 두 수치는 **서로 다른 test set**에서 나왔다 — 원본은 `documentId`가 train·val에 걸치는 **누수 위험** 24,525건, 재현은 그 누수를 제거한 11,271건. 누수는 점수를 **부풀리는** 방향으로 작용하므로, 누수를 없앤 재현이 다소 낮게 나오는 것은 예상된 결과이자 더 정직한 수치다.
- **모델 선택은 재현이 더 견고하다.** 원본 `02` cell-13의 저장 조건 `best_loss < valid_loss or ...`는 valid loss가 **나빠질 때도** 저장하는 결함성 기준이다. 재현은 best `weighted_f1` + EarlyStopping으로 대체해 선택 노이즈를 줄였다.
- **비의도 편차는 미세하다.** Adam `correct_bias`(원본 False vs HF 기본 True), 다중라벨 문서(~14%)의 정답 단일화 방식 차이(`MultiLabelBinarizer.inverse_transform` vs `argmax`) 정도이며, headline에 미치는 영향은 무시할 수준이다.
- **과적합 신호와 macro 하락.** 12에폭 완주 시 train focal loss가 ~2e-4까지 내려가 사실상 암기 상태였다(dropout 0.5에도). macro-F1이 상대적으로 더 하락(원본 0.8038 대비 재현 0.7870)한 점은 꼬리 클래스에서의 손해로, 개선 여지가 있는 지점이다. 

## 다음 단계·유의

- **A.X-Encoder(04)는 KoBERT와 반드시 같은 test split에서 평가한다.**
- **top-1 평가의 의미**: 학습은 멀티레이블(sigmoid+focal)이나 headline은 top-1 단일 정확도 성격이다. 멀티레이블 목표 지표(micro/macro-F1@threshold)와 구분해 해석하고, 장문 인코더 비교 시 두 관점을 함께 본다.

