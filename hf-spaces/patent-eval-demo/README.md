---
title: Patent Classification — Evaluation Explorer
emoji: 📐
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
short_description: 특허 중분류 분류 — 채점 방식·임계값·문서 길이별 평가 탐색기
---

# 특허 중분류 분류 — 평가 탐색기

특허 문헌을 과학기술표준분류 188개 중분류로 분류하는 프로젝트의 평가를 직접 조작해 보는 정적 데모다.
모델을 싣지 않는다 — 저장된 로짓과 동봉 정답으로 채점만 다시 계산한다.

## 패널

| 패널 | 보이는 것 |
|---|---|
| ① 채점법 | τ를 움직이면 다중 라벨 지표는 반응하지만 배포처 방식(top-1 weighted-F1)은 고정이다. 같은 예측에서 두 지표의 격차는 4.09pt다. |
| ② top-1이 버리는 정답 | test 11,244문서의 정답 라벨 13,534개 중 top-1 채점이 보는 것은 11,244개뿐이다. 2,290개(16.9%)가 채점에서 빠진다. |
| ③ 길이 구간별 이득 | 512 창 대비 이득은 문서가 길수록 커지고(+0.29 → +1.58pt), 8192 대비로는 평평하다. |

## 파일

| 파일 | 설명 |
|---|---|
| `index.html` | 데모 전체 (CSS·JS 인라인, 차트는 CSS div) |
| `payload.js` | 사전 계산 데이터. `scripts/build_demo_payload.py`가 생성하며 직접 편집하지 않는다 |

`payload.js`는 `window.PATENT_EVAL`을 정의한다. fetch 없이 `<script src>`로 읽으므로
데이터를 다시 만들어도 `index.html`을 건드리지 않는다.

## 데이터 범위

τ 스윕은 0.05~0.95를 0.01 간격으로 5개 모델에 대해 미리 돌린 곡선이다. 원 데이터(AI Hub 71531)와
가공 데이터셋은 재배포 제한이 있어 포함하지 않으며, 데모가 담는 것은 집계된 지표뿐이다.

배포처 baseline F1 0.8249는 다른 test(24,525건, 문서 누수 있음)에서 잰 값이라 이 데모의 숫자와
직접 비교할 수 없다. “배포처 방식” 열은 같은 채점법을 누수를 제거한 자체 test 11,244에 적용한
연속성 앵커다.

## 갱신

```bash
uv run python scripts/build_demo_payload.py     # payload.js 재생성
```

## 배포

이 폴더를 HF static space `ingyoun/patent-eval-demo`에 업로드하면 갱신된다.

```python
from huggingface_hub import upload_folder
upload_folder(folder_path=".", repo_id="ingyoun/patent-eval-demo", repo_type="space")
```

서빙 URL: https://ingyoun-patent-eval-demo.static.hf.space
