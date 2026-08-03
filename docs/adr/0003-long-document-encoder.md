# ADR-0003 — 주 모델: 장문 인코더 `skt/A.X-Encoder-base`

- **상태**: 수용
- **용어·런 코드**: [GLOSSARY.md](../GLOSSARY.md)
- **참조**: `PROJECT.md` 「long-document 축」·「모델」, `docs/experiments/modernbert.md`, `docs/experiments/modernbert-comparison.md` 「길이 vs 모델 분해」, `docs/experiments/no-train-analysis.md` C

## 맥락

공식 baseline(KoBERT)은 512 토큰에서 입력을 절단한다. 특허 문헌의 실체는 주로 `claims`이며 길이 분포의 꼬리가 길어(p99≈3,621, max 10,523), 512 창이 본문의 상당 부분을 버린다는 것이 프로젝트의 출발 아이디어였다. 이를 검증·회수할 모델이 필요했다.

## 결정

**주 모델을 `skt/A.X-Encoder-base`로 확정한다** — ModernBERT 아키텍처의 한국어 인코더(컨텍스트 16,384, 149M, apache-2.0). 입력은 `invention_title + ipc_main + abstract + claims` 고정 조합, `max_length`는 소비 시점에 건다.

## 근거 (실측)

- **exp1(8192)이 KoBERT 재현선을 넘었다** — 다중 라벨 micro 0.8502 → **0.8685**(+1.83pt), top-1 weighted 0.8148 → 0.8256.
- **개선 분해(exp2 512 control)**: 컨텍스트 길이에 **+0.84pt**(exp1−exp2, 같은 모델·토크나이저라 길이에 통제 귀속) + 모델 성분 **+0.99pt**(exp2−KoBERT). 창 확장 효과가 길이 bin에서 단조 증가(B0 +0.46 → B3 +2.64pt)해 장문 가설을 지지한다.
- 모델 성분에는 토크나이저 압축이 512 창에 만든 커버리지 우위(절단 문서에서 ~10% 더 많은 본문, `no-train-analysis.md` C 실측)가 섞여 있어 순수 아키텍처 이득으로 읽지 않는다.

## 검토한 대안

- **KoBERT 유지(512)** — 기각. 절단 상한이 헤드룸을 제약한다.
- **KLUE-RoBERTa-base(512)** — 대조군(선택 항목)으로만 유지. 성능이 아니라 크기 교란 요인 방어가 목적이며 예산이 남을 때만 집행.
- **멀티모달(도면 이미지)** — 스코프 밖.

## 결과·영향

- long-document 축은 **실측 완료로 종료** — 추가 검증(입력 필드 실험·대조군 등)에 예산을 쓰지 않는다(`PROJECT.md` 「닫힌 갈래」).
- **비용은 "저비용"이 아니다** — exp1 훈련 ≈29h로 KoBERT 재현 ≈10h의 약 3배. 운영 단일 모델을 8192로 갈지 512로 갈지는 추론 비용 차이 때문에 레시피 확정 후 별도로 결정한다.
- headline은 "장문 + 더 나은 한국어 인코더의 결합 효과"로 서술하되, 주 동력은 512 창이 버리던 본문의 회복이다.
- 공통 함정(dtype fp32 로드 + autocast bf16, 특수토큰 절단 복원, 평가 절단·batch)은 `docs/experiments/modernbert.md` 공통 프로토콜에 있다. 이 결정을 실행하는 코드는 그 함정을 그대로 따른다.
