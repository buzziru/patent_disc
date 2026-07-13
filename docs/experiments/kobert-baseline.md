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

## 멀티라벨·길이구간 재평가 (03_02_Metric)

top-1 weighted-F1은 벤더 연속성을 위한 **앵커**일 뿐, 멀티라벨(문서당 평균 약 1.2 라벨, 고유 문서의 약 14%가 2개 이상)을 top-1 단일라벨로 접어 성능을 구조적으로 축소한다. 이를 바로잡고 장문 가설(길이구간별 성능)을 측정하기 위해 `notebook/03_02_Metric.ipynb`로 **같은 고정 test(11,271) 위에서** 진짜 멀티라벨 지표를 산출한다. 실측 결과: `output/metrics_kobert-patent-baseline_len512.json`.

- **멀티라벨 F1 (τ=0.5 고정, sigmoid+BCE 확률):** micro **0.8502** / macro **0.8470** / sample **0.8656**. 빈 예측률 1.16% — 빈 예측을 argmax 1개로 강제 보정해도 micro 0.8506 / macro 0.8474로 거의 동일(보정 여지가 작다는 뜻).
- **랭킹 지표:** LRAP **0.9272**, R-Precision **0.8816**.
- **앵커 top-1**(위 「결과」 표와 동일 계산): weighted-F1 0.8148, P@1/3/5 = 0.894 / 0.962 / 0.979. 두 경로가 일치해 지표 구현의 정합성을 확인한다.

### 길이구간별 F1 (bin은 `kobert_len` 기준 고정)

문서를 KoBERT 토큰 길이로 4구간에 고정 배정한다. bin 소속을 KoBERT 길이로 **고정**해야 이후 A.X-Encoder를 **같은 문서집합·같은 축** 위에서 delta로 비교할 수 있다.

| bin | n | micro-F1 | macro-F1 |
| --- | --- | --- | --- |
| B0 (≤512) | 3,197 | 0.8685 | 0.8566 |
| B1 (512–1024) | 5,183 | 0.8531 | 0.8503 |
| B2 (1024–2048) | 2,342 | 0.8261 | 0.8138 |
| B3 (>2048) | 549 | 0.8212 | 0.7244 |

### 해석

- **top-1 프레이밍이 성능을 축소함을 실측 확인.** 앵커 top-1은 0.8148인데 멀티라벨 micro는 0.8502다(분모가 달라 뺄셈 비교는 불가). `P@1`(0.894)이 top-1 accuracy(0.8147)보다 8pt 높은 것도 같은 현상 — 멀티라벨 문서에서 모델이 "유효하지만 argmax-gold가 아닌" 라벨을 상위로 꼽는 경우다.
- **macro ≈ micro (0.847 vs 0.850)는 평탄화 분포의 직접 증거.** 188-class 멀티라벨에서 통상 macro ≪ micro인데 붙어 있는 것은 클래스 균형(중분류당 1,300 ~ 2,600건) 때문 — 지표가 데이터 설계와 정합한다.
- **랭킹(R-Prec 0.882) > 임계 결정(micro 0.850).** τ=0.5가 최적은 아니며 임계 튜닝 여지가 있다. 현재는 "τ=0.5 고정" 결정을 따르되, 후속 실험에서 val 임계 튜닝을 성능 레버로 남겨둔다.
- **길이 단조 하락은 512 truncation 가설의 정황이나 확증은 아니다.** micro가 B0 0.869 → B3 0.821로 완만히 내려간다. 다만 긴 문서가 truncation 때문에 어려운지, 원래 복잡·다라벨이라 어려운지 KoBERT 단독으로는 분리 불가능하다. 이 confound를 깨는 것은 **동일 bin 위 A.X-Encoder 재평가**(장문이 B2/B3를 회복하고 B0는 유지되는지)다.
- **B3 macro 0.724는 소표본 잡음이 크다.** 549문서 × 약 1.2라벨을 188클래스에 흩뿌리면 다수 클래스의 support가 0~3이라 per-class F1이 널뛴다(`zero_division=0`이 결측 클래스를 0으로 끌어내림). **길이 bin은 micro를 주 지표로, macro는 소표본 주의 각주와 함께** 읽는다 — B3 macro를 단독 인용하지 않는다.

## 회고·분석

- `**0.8148`을 `0.8249`와 직접 비교하지 않는다.** 두 수치는 **서로 다른 test set**에서 나왔다 — 원본은 `documentId`가 train·val에 걸치는 **누수 위험** 24,525건, 재현은 그 누수를 제거한 11,271건. 누수는 점수를 **부풀리는** 방향으로 작용하므로, 누수를 없앤 재현이 다소 낮게 나오는 것은 예상된 결과이자 더 정직한 수치다.
- **모델 선택은 재현이 더 견고하다.** 원본 `02` cell-13의 저장 조건 `best_loss < valid_loss or ...`는 valid loss가 **나빠질 때도** 저장하는 결함성 기준이다. 재현은 best `weighted_f1` + EarlyStopping으로 대체해 선택 노이즈를 줄였다.
- **비의도 편차는 미세하다.** Adam `correct_bias`(원본 False vs HF 기본 True), 다중라벨 문서(~14%)의 정답 단일화 방식 차이(`MultiLabelBinarizer.inverse_transform` vs `argmax`) 정도이며, headline에 미치는 영향은 무시할 수준이다.
- **과적합 신호와 macro 하락.** 12에폭 완주 시 train focal loss가 ~2e-4까지 내려가 사실상 암기 상태였다(dropout 0.5에도). macro-F1이 상대적으로 더 하락(원본 0.8038 대비 재현 0.7870)한 점은 꼬리 클래스에서의 손해로, 개선 여지가 있는 지점이다.
- **512 truncation의 실제 손실.** KoBERT 토크나이저 기준 입력 토큰 길이 중앙값이 약 700, 문서의 **약 71%가 512를 초과**한다(`notebook/02_02_KoBERT_Tokenizer.ipynb` 실측 → `../data/data.md` 「입력 토큰 길이 분포」). 즉 baseline `0.8148`은 다수 문서의 뒷부분(주로 `claims`)을 버린 채 얻은 수치 — 장문 인코더(A.X-Encoder)가 회복할 상한을 시사한다.

## 다음 단계·유의

- **A.X-Encoder(04)는 KoBERT와 반드시 같은 test split에서 평가한다.** 나아가 `03_02_Metric`의 **같은 bin·같은 문서집합**으로 재평가해 bin별 delta(A.X-Encoder − KoBERT) 표를 직접 산출한다 — B2/B3 회복 여부가 장문 가설의 결정적 그림이다.
- **두 관점을 함께 본다**: 앵커 top-1 weighted-F1(벤더 연속성)과 멀티라벨 micro/macro-F1@τ(목표 지표)는 계산이 다르므로 뺄셈 비교하지 않고 각각의 축에서 비교한다.
- **`0.8148`(top-1)과 `0.8502`(멀티라벨 micro)를 혼동하지 않는다** — 전자는 벤더 비교용 앵커, 후자는 멀티라벨 목표 지표다.

