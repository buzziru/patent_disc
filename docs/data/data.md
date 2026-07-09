# 데이터 — AI Hub 71531 (과학기술표준분류 대응 특허 데이터)

17대분류 / 188중분류, 출처 KIPRIS(한국 공개/등록). 원본 설명서: `data/과학기술표준분류 대응 특허데이터_데이터설명서.pdf`.
`data/`는 `.gitignore` 대상이라 커밋되지 않는다(전체 419MB).

> **건수 주의**: 공식 문서의 `300,240`은 고유 문서 수가 아니라 **(문서, 레이블) 쌍 수**다. 한 특허가 여러 중분류 zip에 중복 수록되므로(→ 다중 레이블) 고유 특허 수는 그보다 적다. **로컬 스냅샷** 실측: (문서,레이블) 쌍 270,216 / **고유 documentId 224,328**(Training 202,860 · Validation 29,290, 단 **7,822건이 양 폴더에 겹침**). 로컬은 업체 `datadam_20230116` 버전과 다른(작은) 스냅샷이다.

## 레이아웃

```
data/
  Training/   Validation/
    Orig/    ← 입력(TS_… .zip)  — 188개 zip = 188개 중분류
    Label/   ← 정답(TL_… .zip)  — 188개 zip = 188개 중분류
```

- zip 파일명은 중분류 단위: `TS_TS1_<대분류코드>_<대분류명>_<중분류코드>_<중분류명>.zip` (라벨은 `TL_TL1_…`).
- 각 zip 안에는 특허 1건당 JSON 1개. **파일명(= `documentId`, 예 `kr20020059117b1`)으로 원천↔라벨을 조인**한다.
- 압축 해제 시 파일 수가 많으므로 **`zipfile`로 zip 내부 JSON을 직접 스트리밍**하는 것을 권장(`unzip -p <zip> <name.json>`으로 단건 확인).

## JSON 스키마

두 파일 모두 최상위 `dataset` 키 아래에 필드가 있다.

**원천데이터**(입력): `invention_title`, `abstract`, `claims`, `applicant_name`, `open_date`/`application_year`/`register_date`, IPC 필드(`ipc_main`, `ipc_section`, `ipc_class`, `ipc_subclass`, `ipc_all`) 등.
→ 분류기 입력 후보 텍스트 필드: **명칭(`invention_title`) · 요약(`abstract`) · 청구항(`claims`)**. 표본 실측 존재율 ~100%(`invention_title` 100%, `abstract`·`claims` 99.9%). **별도 `상세설명` 필드는 없다** — 장문의 실체는 주로 `claims`. 어떤 필드를 쓰는지가 장문 처리 실험의 핵심 축.

**라벨링데이터**(정답): `documentId`, **`Lno`**(대분류 코드, 예 `EA`) / `Ltext`(예 `기계`), **`Mno`**(중분류 코드, 예 `EA01`) / `Mtext`, `ipc_main`, `country_code`, `document_type` 등. 파일이 든 zip명 자체가 그 문서의 `Mno`를 인코딩(`TL_TL1_EG_원자력_EG10_핵융합.zip` → `EG10`).
→ 학습 타깃은 **문서별 `Mno` 다중-핫(188-class, multi-label)**. 한 특허가 여러 `Mno`를 가질 수 있다. `Lno`(대분류)는 **예측된 각** `Mno` → `Lno` 매핑으로 유도(별도 Lno 헤드 불필요 → 계층 일관성 유지).

## 17개 대분류

EA 기계 · EB 재료 · EC 화공 · ED 전기 · EE 정보 · EF 에너지 · EG 원자력 · EH 환경 · EI 건설 · LA 생명과학 · LB 농림수산식품 · LC 보건의료 · NB 물리학 · NC 화학 · ND 지구과학 · OA 뇌과학 · OB 인지.

## 주의 (분포·비교)

- **다중 레이블(multi-label)**: 한 특허가 여러 중분류에 대응한다. 로컬 Training 실측 — 고유 문서 202,860 중 문서당 서로 다른 `Mno` 개수 분포 `{1: 174,275, 2: 22,282, 3: 4,617, 4: 1,128, 5: 396, 6: 134, 7: 20, 8: 7, 10: 1}` → **다중 Mno 문서 28,585건(14.1%)**, 평균 ~1.2개. 학습 타깃은 문서별 다중-핫 벡터(sigmoid+BCE).
- **분할 누수 주의**: 제공된 `Training`/`Validation` 폴더는 **7,822개 documentId가 양쪽에 중복**된다. 문서 단위로 집계 후 **`documentId` 기준 재분할**해야 누수가 없다.
- **제공된 Training/Validation은 라벨별 1/9 층화 분할**: zip(=Mno) 멤버 수 실측 — (문서,라벨) 쌍 Training 240,192 / Validation 30,024. 라벨별 검증 분율 `v/(t+v)`가 **전 188개 중분류에서 상수 0.111(=1/9), sd 0.000**이며 train_share↔val_share 상관 **0.9998**. 즉 제공 분할은 랜덤이 아니라 **각 zip을 9:1로 쪼갠 층화 분할**이다(전체 표 산출 스크립트는 `scratchpad/label_dist.py`). → 재분할할 때도 **라벨 분포를 보존하는 층화 분할을 써야 이 기준선과 정합**한다. 단순 `train_test_split`(비율만, 비층화)은 188-way·희소 클래스(train 최소 406건)에서 검증셋 라벨 분포를 왜곡한다.
- **인위적으로 평탄화된 분포**: 중분류당 대체로 1,300~2,600건. 실제 출원 분포와 다르므로 결과 해석 시 한계로 명시.
- **`0.8249`는 top-1 예측의 weighted-F1**(full test 24,525건 실측 — Micro 0.8261 / Macro 0.8038). baseline 데이터 분할은 **`documentId`가 train·val 양쪽에 존재하는 데이터 누수** 위험이 있어, **누수 없는 데이터셋을 새로 생성**하고 자체 test 고정 + KoBERT 자체 재현으로 비교선을 세운다(`../../PROJECT.md` 평가 절, 재현 절차 `../experiments/kobert-baseline.md`).
