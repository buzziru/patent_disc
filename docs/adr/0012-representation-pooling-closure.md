# ADR-0012 — 표현·풀링 축 종결: 장문 열화는 본질적 난이도

- **상태**: 수용
- **용어·런 코드**: [GLOSSARY.md](../GLOSSARY.md)
- **참조**: [ADR-0003](./0003-long-document-encoder.md)(long-document 축), [ADR-0006](./0006-no-calibration.md)(캘리브레이션 미채택), [ADR-0009](./0009-loss-axis-closure.md)(카디널리티), `../experiments/longdoc-degradation.md`(발단·실측 SSOT), 노트북 `../../notebook/12_01_HiddenState_Dump.ipynb`(hidden-state 덤프)·`../../notebook/12_02_HiddenState_PoolingProbe.ipynb`(동결 풀링 프로브), `../../output/hidden_pooling_probe_test.json`(수치 SSOT)

## 맥락

long-document 축([ADR-0003])의 후속으로 장문 열화 진단(`longdoc-degradation.md`)이 8192에서도 긴 문서 성능이 떨어지는 원인을 **결정층·본질적 난이도**로 귀속했다. 그러나 "표현 붕괴가 아니다"라는 판정은 저장 로짓(풀링 **하류**)으로만 내렸고, 로짓은 풀링 이후라 풀링 단계 자체의 손실을 볼 수 없다. mean 풀링이 8192 토큰을 평균해 소수의 정보성 청구항 토큰을 정형 문구와 함께 희석한다는 가설(→ 항희석·label-aware 풀링으로 회수)이 미확정으로 남았다.

## 결정

**표현·풀링 축(풀링 교체·label-aware attention 헤드)을 성능 레버로 채택하지 않는다.** 장문 열화는 표현·풀링·헤드로 회수되지 않는 본질적 난이도로 종결한다.

## 근거

hidden-state 덤프(`12_01`, test micro 0.8683으로 SSOT 정합해 복원·행순서·`mean==모델 풀링` 검증) 위에서 동결 선형 프로브(`12_02`)로 고정 풀링 4종(mean·max·cls·last)의 test R-Precision을 길이 bin별로 비교한 실측이 근거다(수치 `longdoc-degradation.md`「풀링 실측」).

- 최장 bin(B3)에서 항희석 풀링 max가 mean 대비 **+0.32pt**뿐 — 사전 설정 0.5pt 게이트 아래, ALL에선 −0.07pt.
- concat(mean⊕max)이 mean과 동률 — 상보 신호 부재. mean이 max가 보는 것을 이미 담는다.
- 길이 기울기가 풀링과 무관(mean −4.7 · max −4.2pt) — 저하가 mean 특유가 아니라 표현 전반의 성질.
- mean 선형 프로브 ALL R-Prec 0.8952가 파인튜닝 모델(0.8970)에 근접 — 표현이 이미 모델 수준으로 선형 분리 가능, 갭은 풀링·헤드가 아니라 표현에 있다.

## 검토한 대안

- **학습형 attention 풀러(토큰 정보량 학습 가중)** — 우선순위 하향. 고정 풀링 실측의 세 신호(max 무회수·concat 무상보·풀링 무관 기울기)가 토큰 특징에 회수가능 장문 신호가 없음을 시사해 헤드룸이 얇다. 완전 확정은 토큰 단위 hidden-state 덤프 + attention 프로브를 요구하나 기대 이득이 위 상한 대비 얇아 착수하지 않는다.
- **label-aware attention 헤드를 장문 축에 적용** — 미채택(같은 표현 헤드룸에 걸린다). 단 이를 **카디널리티 축**(k≥2 recall 결손)에서 재평가하는 것은 별개 갈래로 열어 둔다 — KD와 상보적이다.

## 결과·영향

장문 열화 진단이 연 표현/풀링 갈래가 닫혔다. 남은 헤드룸은 k≥2 카디널리티([ADR-0009] 오라클-k +1.63pt)와 이를 훈련 시점에 겨냥하는 KD 축이며, 장문 축 자체는 운영 모델(exp1 8192)에 이미 채택돼 있다. 이 결정은 [ADR-0010](./0010-data-cleaning.md)의 클래스 모호성 확증(장문 열화의 근원이 데이터 오류가 아니라 문제 난이도)과 같은 방향이다.
