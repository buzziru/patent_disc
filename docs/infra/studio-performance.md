# 스튜디오 체감 느림 / "CPU 과부하" 진단

> ⚠️ **과거 Lightning cloudspace 기록 — 현재 로컬 Windows 작업엔 해당 없음.** 작업을 로컬로 옮기기 전(Lightning 스튜디오 인터랙티브 세션 시절)의 진단 노트다. 지금은 스튜디오 인터랙티브 세션을 쓰지 않고, GPU는 로컬에서 Docker 이미지 잡으로 제출한다([lightning-jobs.md](./lightning-jobs.md)). 컨테이너 vs 호스트 CPU 구분 기법은 다른 컨테이너 환경에서 재사용 가능해 보존한다.

Lightning cloudspace(containerd 컨테이너)에서 **"작업도 안 하는데 느리다 / CPU 과부하"** 로 보일 때의 진단 절차와 실제 원인. 2026-07-03 `patent_edu` 스튜디오에서 확인.

## 핵심 결론 (TL;DR)

- **`vmstat`·`/proc/loadavg`·`/proc/stat`·Lightning UI의 CPU 미터는 컨테이너가 아니라 물리 호스트 전체를 반영한다.** containerd는 이 값들을 컨테이너별로 가상화하지 않는다. → 로드·CPU%가 높아도 **내 스튜디오 탓이 아닐 수 있다.**
- 내 컨테이너의 **실제** CPU는 cgroup `cpu.stat`으로 재야 한다. 실측 사례: **컨테이너 9.3% vs 호스트 62.6%** → 나머지는 **같은 물리머신을 쓰는 이웃 테넌트(다른 스튜디오)** 의 부하 = 노이지 네이버.
- CPU 쿼터 제한 없음(`cpu.max = max`), 스로틀 0(`nr_throttled=0`)이라 **호스트가 붐비면** 대화형 프로세스(에디터·터미널)가 타임슬라이스를 늦게 받아 **반응이 끊긴다** — 정작 내 사용률은 낮은데도.
- **해결은 컨테이너 밖.** 프로세스 킬·에디터 설정으로 안 고쳐진다.

## 결정적 측정: 컨테이너 vs 호스트 CPU 비교

```bash
HZ=$(getconf CLK_TCK); CORES=$(nproc)
u1=$(awk '/usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat)
read _ a1 b1 c1 idle1 io1 irq1 sirq1 st1 _ < /proc/stat
sleep 3
u2=$(awk '/usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat)
read _ a2 b2 c2 idle2 io2 irq2 sirq2 st2 _ < /proc/stat
cont_pct=$(awk "BEGIN{printf \"%.1f\", ($u2-$u1)/1000000/3/$CORES*100}")
busy1=$((a1+b1+c1+irq1+sirq1+st1)); busy2=$((a2+b2+c2+irq2+sirq2+st2))
tot1=$((busy1+idle1+io1)); tot2=$((busy2+idle2+io2))
host_pct=$(awk "BEGIN{printf \"%.1f\", ($busy2-$busy1)/($tot2-$tot1)*100}")
echo "컨테이너(cgroup): ${cont_pct}%  /  호스트(/proc/stat): ${host_pct}%  (코어 $CORES)"
```

**호스트는 높은데 컨테이너가 낮으면 = 이웃 테넌트(호스트 경합).** 내 컨테이너가 높으면 그때야 아래 "내부 원인"을 판다.

## 진단 순서 (체크리스트)

1. **좀비**: `ps -eo stat | grep -c Z` — 좀비는 CPU/메모리를 안 먹는다. 거의 항상 원인 아님.
2. **cgroup 스로틀/쿼터**: `cat /sys/fs/cgroup/cpu.stat /sys/fs/cgroup/cpu.max` — `nr_throttled>0`이면 쿼터에 막히는 것.
3. **steal / iowait**: `vmstat 2 5`의 `st`(하이퍼바이저 steal)·`wa`(디스크 대기)·`b`(D-state). 사례에선 st 1~2%, wa 0으로 정상.
4. **컨테이너 vs 호스트 CPU**: 위 스크립트. **가장 중요.**
5. **per-PID CPU는 스냅샷 말고 시간표본으로**: `/proc/PID/stat`의 14+15번째 필드를 N초 간격 델타로. 단, **단명(<150ms) 프로세스는 놓친다** → 집계는 `vmstat`(컨테이너면 cgroup)로 교차검증.

## 주의: 오진하기 쉬운 것들

- **에디터 파일워칭/언어서버 인덱싱**: `.venv`·`data` 같은 대형 트리를 워칭·인덱싱하면 버스트 CPU를 만든다. 실제로 워크스페이스 `.vscode/settings.json`에 `files.watcherExclude`/`search.exclude`/`python.analysis` 제외를 넣어 완화했다(위생상 유지). **단, 이번 과부하의 실제 원인은 아니었다** — 컨테이너 전체가 9.3%였으므로. `files.exclude`는 탐색기 "숨기기"만 할 뿐 워칭/인덱싱을 막지 못한다는 점 주의.
- **code-server(VS Code) 프로세스 킬**: Lightning IDE가 code-server 기반이라 `/commands/detached_commands/start-vscode.sh`가 PID 1 아래에서 **자동 재생성**한다. 죽여도 소용없다.
- **`cpuUsage.sh`가 `sleep`을 반복 spawn**: Cursor/code-server 내장 CPU 모니터(`out/vs/base/node/cpuUsage.sh`). `/proc/stat`을 1초 간격으로 읽어 프로세스 CPU%를 계산하는 정상 동작. 원인 아님.

## 처방 (호스트 경합일 때)

1. **스튜디오 Stop→Start** — 재시작 시 대개 덜 붐비는 다른 물리 호스트로 재배치된다. 노이지 네이버의 1차 처방.
2. 반복되면 **전용/상위 머신 티어**로 (코어 비공유).
3. 스튜디오 복제(duplicate)와는 무관 — 복제본이 붐비는 호스트에 배치됐을 뿐.

## 참고: 적용해둔 에디터 설정 (`.vscode/settings.json`)

`files.watcherExclude`·`search.exclude`(`.venv`·`data`·`__pycache__` 등) + `python.analysis.indexing:false`·`diagnosticMode:openFilesOnly`·`analysis.exclude`. Cursor도 워크스페이스 설정을 읽으므로 두 에디터에 동시 적용. 위생 개선용이며 호스트 경합은 못 고친다.
