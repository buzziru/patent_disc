"""`ipc_all` — `ipc_main` — `Mno` 관계 실측.

현재 모델 입력은 IPC 필드 중 `ipc_main` 하나만 쓴다. 원천 데이터에는 `ipc_all`(복수 코드)이
함께 있으므로, **버려지는 보조 코드가 라벨 정보를 담는가**, 특히 **다중레이블 문서의 추가
`Mno`를 가리키는가**를 네 각도로 잰다.

  (1) 구조 — `ipc_main`이 `ipc_all`의 부분집합인가, 코드 수 분포는 어떤가,
      한 문서가 여러 zip에 수록될 때(다중 `Mno`) **사본 간 IPC 필드가 갈리는가**.
      갈린다면 `documentId` 최초 1회 dedup이 임의의 사본을 고르는 셈이 된다.
  (2) 폭 대 형상 — 보조 코드 수·새 서브클래스·새 섹션이 라벨 형상(k=1 / within-`Lno` /
      cross-`Lno` / 혼합)에 따라 달라지는가.
  (3) 커버리지 — train에서 만든 서브클래스→`Mno` top-1 연관표로 정답 `Mno`를 덮을 때,
      보조 코드가 `ipc_main`이 못 덮은 라벨을 얼마나 회수하는가.
  (4) 상한 — IPC만 쓰는 188-way 다중라벨 프로브(`ipc_main` / 보조만 / 전체)로 정보량을
      직접 재고, 텍스트 모델 로짓과 후기융합해 **텍스트가 이미 갖지 않은 잔여 정보**를
      paired bootstrap으로 검정한다.

구조·폭 통계는 raw zip 전수(224,328문서), 프로브·융합은 정리 split(train 201,616 / val
11,132 / test 11,244)로 낸다.

실행: `uv run python scripts/ipc_field_analysis.py`   (~10분, CPU)
산출: `output/ipc_field_analysis.json`
"""

import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
ROOT = Path(os.environ["DATA_ROOT"])
DATA = ROOT / "data"
OUT = ROOT / "output"

MNO_RE = re.compile(r"[A-Z]{2}[0-9]{2}")
NUM_LABELS = 188
TEXT_LOGITS = "modernbert-patent-len512-b128"     # 현행 최고 체크포인트(정리 축)


# ---------------------------------------------------------------- 수집

def scan_raw():
    """Label zip 명으로 doc→Mno, Orig zip 전수로 사본별 IPC 필드를 모은다(dedup 하지 않음)."""
    doc2mno = defaultdict(set)
    for split in ["Training", "Validation"]:
        for z in (DATA / split / "Label").glob("*.zip"):
            mno = MNO_RE.findall(z.stem)[0]
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.endswith(".json"):
                        doc2mno[Path(n).stem].add(mno)

    recs = defaultdict(list)
    for split in ["Training", "Validation"]:
        for z in sorted((DATA / split / "Orig").glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if not n.endswith(".json"):
                        continue
                    d = json.loads(zf.read(n))["dataset"]
                    recs[Path(n).stem].append((d.get("ipc_main") or "", d.get("ipc_all") or ""))
    return {doc: {"mno": sorted(doc2mno.get(doc, ())), "copies": recs[doc]} for doc in recs}


def splits(P):
    val = json.loads((OUT / "doc_ids_clean_val.json").read_text(encoding="utf-8"))
    test = json.loads((OUT / "doc_ids_clean_test.json").read_text(encoding="utf-8"))
    lc = json.loads((OUT / "label_conflict_docs.json").read_text(encoding="utf-8"))
    removed = set(lc["conflict_docs"]) | set(lc["train_dedup_docs"]) \
        | set(lc["eval_leak_docs"]) | set(lc["valtest_dedup_docs"])
    ev = set(val) | set(test)
    train = [d for d in P if d not in ev and d not in removed]
    return train, val, test


def parts(P, doc):
    """(ipc_main, 보조 코드, 전체 코드) — 첫 사본 기준."""
    main, alls = P[doc]["copies"][0]
    cs = [c for c in alls.split("|") if c]
    return main, [c for c in cs if c != main], cs


def shape_of(mnos):
    lnos = {m[:2] for m in mnos}
    if len(mnos) == 1:
        return "k=1"
    if len(lnos) == 1:
        return "within-Lno"
    return "cross-Lno" if len(lnos) == len(mnos) else "mixed"


# ---------------------------------------------------------------- (1) 구조

def structure(P):
    n_main_empty = n_all_empty = in_all = first = not_in = 0
    ncodes, var_main, var_all, n_multi_copy = Counter(), 0, 0, 0
    for v in P.values():
        main, alls = v["copies"][0]
        cs = [c for c in alls.split("|") if c]
        n_main_empty += not main
        n_all_empty += not cs
        ncodes[len(cs)] += 1
        if main and cs:
            if main in cs:
                in_all += 1
                first += cs[0] == main
            else:
                not_in += 1
        if len(v["copies"]) > 1:
            n_multi_copy += 1
            var_main += len({c[0] for c in v["copies"]}) > 1
            var_all += len({c[1] for c in v["copies"]}) > 1
    tot = sum(ncodes.values())
    codes = Counter(c for v in P.values() for c in v["copies"][0][1].split("|") if c)
    return {
        "n_docs": tot,
        "ipc_main_missing": n_main_empty, "ipc_all_missing": n_all_empty,
        "main_in_all": in_all, "main_is_first_code": first, "main_not_in_all": not_in,
        "n_codes_dist": {int(k): int(ncodes[k]) for k in sorted(ncodes)},
        "mean_codes": sum(k * c for k, c in ncodes.items()) / tot,
        "single_code_share": ncodes[1] / tot,
        "n_docs_multi_copy": n_multi_copy,
        "copies_disagree_ipc_main": var_main, "copies_disagree_ipc_all": var_all,
        "vocab": {"full_code": len(codes), "subclass": len({c[:4] for c in codes})},
    }


# ---------------------------------------------------------------- (2) 폭 대 형상

def breadth_by_shape(P, ids):
    g = defaultdict(list)
    for d in ids:
        main, sec, cs = parts(P, d)
        g[shape_of(P[d]["mno"])].append((
            len(cs), len(sec),
            len({c[:4] for c in sec} - {main[:4]}),
            len({c[:1] for c in sec} - {main[:1]}),
            len({c[:1] for c in cs}) >= 2,
        ))
    keys = ["n_codes", "n_secondary", "secondary_new_subclass", "secondary_new_section",
            "multi_section_rate"]
    return {s: dict(n_docs=len(v), **{k: float(np.mean([x[i] for x in v]))
                                      for i, k in enumerate(keys)})
            for s, v in g.items()}


# ---------------------------------------------------------------- (3) 커버리지

def coverage(P, train, test):
    sub2mno = defaultdict(Counter)
    for d in train:
        for c in parts(P, d)[2]:
            for m in P[d]["mno"]:
                sub2mno[c[:4]][m] += 1
    top1 = {s: c.most_common(1)[0][0] for s, c in sub2mno.items()}
    hit = lambda cs: {top1[c[:4]] for c in cs if c[:4] in top1}

    by_k = defaultdict(lambda: Counter())
    miss_tot = miss_by_sec = 0
    for d in test:
        main, sec, cs = parts(P, d)
        true = set(P[d]["mno"])
        k = len(true)
        cm, csec, ca = hit([main]), hit(sec), hit(cs)
        r = by_k[k]
        r["docs"] += 1; r["labels"] += k
        r["main"] += len(true & cm); r["secondary"] += len(true & csec); r["all"] += len(true & ca)
        r["secondary_only"] += len((true & csec) - cm)
        if k >= 2:
            miss = true - cm
            miss_tot += len(miss)
            miss_by_sec += len(miss & csec)
    agg = Counter()
    for r in by_k.values():
        agg.update(r)
    rate = lambda r: {k: r[k] / r["labels"] for k in ["main", "secondary", "all", "secondary_only"]}
    return {
        "by_k": {int(k): dict(docs=r["docs"], labels=r["labels"], **rate(r))
                 for k, r in sorted(by_k.items())},
        "all": dict(docs=agg["docs"], labels=agg["labels"], **rate(agg)),
        "multi_label_missed_by_main": miss_tot,
        "multi_label_missed_recovered_by_secondary": miss_by_sec,
        "multi_label_recovery_rate": miss_by_sec / miss_tot,
    }


# ---------------------------------------------------------------- (4) 프로브 · 융합

def feats(cs):
    """코드 리스트 → 피처 집합(서브클래스 · 메인그룹 · 전체 코드 3층)."""
    return {f"SUB:{c[:4]}" for c in cs} | {f"GRP:{c.split('/')[0]}" for c in cs} | \
           {f"FUL:{c}" for c in cs}


def probe_and_fuse(P, train, val, test, mnos):
    m2i = {m: i for i, m in enumerate(mnos)}
    cache = {d: parts(P, d) for d in P}

    df = Counter()
    for d in train:
        df.update(feats(cache[d][2]))
    vocab = {f: i for i, f in enumerate(sorted(f for f, c in df.items() if c >= 5))}

    def design(ids, which):
        rows, cols = [], []
        for r, d in enumerate(ids):
            main, sec, allc = cache[d]
            for f in feats({"M": [main], "S": sec, "A": allc}[which]):
                j = vocab.get(f)
                if j is not None:
                    rows.append(r); cols.append(j)
        return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                                 shape=(len(ids), len(vocab)))

    def gold(ids):
        Y = np.zeros((len(ids), NUM_LABELS), np.int8)
        for r, d in enumerate(ids):
            for m in P[d]["mno"]:
                Y[r, m2i[m]] = 1
        return Y

    Ytr, Yva, Yte = gold(train), gold(val), gold(test)

    def fit_one(Xtr, y, Xva, Xte):
        clf = LogisticRegression(C=1.0, max_iter=400, solver="liblinear")
        clf.fit(Xtr, y)
        return clf.decision_function(Xva), clf.decision_function(Xte)

    micro = lambda Y, S, t: f1_score(Y, (S > t).astype(int), average="micro", zero_division=0)
    scores, probe = {}, {}
    for which in ["M", "S", "A"]:
        Xtr, Xva, Xte = design(train, which), design(val, which), design(test, which)
        out = Parallel(n_jobs=8)(delayed(fit_one)(Xtr, Ytr[:, c], Xva, Xte)
                                 for c in range(NUM_LABELS))
        Sva, Ste = np.stack([o[0] for o in out], 1), np.stack([o[1] for o in out], 1)
        probe[which] = (Sva, Ste)
        tau = max(np.arange(-3, 2.01, 0.1), key=lambda t: micro(Yva, Sva, t))
        pred = (Ste > tau).astype(int)
        scores[which] = {
            "tau": float(tau), "val_micro": float(micro(Yva, Sva, tau)),
            "test_micro": float(f1_score(Yte, pred, average="micro", zero_division=0)),
            "test_macro": float(f1_score(Yte, pred, average="macro", zero_division=0)),
        }
        print(f"  IPC-only {which}: test micro {scores[which]['test_micro']:.4f}", flush=True)

    # --- 후기융합 (w·τ는 val에서 선택, test에서 평가)
    Lva = np.load(OUT / f"logits_{TEXT_LOGITS}_val.npy")
    Lte = np.load(OUT / f"logits_{TEXT_LOGITS}_test.npy")
    grid = np.arange(-3, 3.01, 0.05)
    t0 = max(grid, key=lambda t: micro(Yva, Lva, t))
    base_te = micro(Yte, Lte, t0)

    def counts(pred, Y):
        return ((pred & (Y == 1)).sum(1).astype(float),
                (pred & (Y == 0)).sum(1).astype(float),
                ((~pred) & (Y == 1)).sum(1).astype(float))

    def micro_of(c, idx):
        T, F, N = c[0][idx].sum(), c[1][idx].sum(), c[2][idx].sum()
        return 2 * T / (2 * T + F + N)

    rng = np.random.default_rng(42)
    boot = rng.integers(0, len(test), size=(5000, len(test)))
    base_c = counts(Lte > t0, Yte)
    fusion = {"text_only": {"tau": float(t0), "test_micro": float(base_te)}}
    for which in ["M", "S", "A"]:
        Sva, Ste = probe[which]
        best = max(((micro(Yva, Lva + w * Sva, t), w, t)
                    for w in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0] for t in grid))
        vmicro, w, tau = best
        fc = counts((Lte + w * Ste) > tau, Yte)
        d0 = micro_of(fc, slice(None)) - micro_of(base_c, slice(None))
        ds = np.array([micro_of(fc, b) - micro_of(base_c, b) for b in boot])
        k = Yte.sum(1)
        fusion[which] = {
            "w": float(w), "tau": float(tau), "val_micro": float(vmicro),
            "test_micro": float(micro_of(fc, slice(None))),
            "delta_pt": float(d0 * 100),
            "delta_ci95_pt": [float(np.percentile(ds, 2.5) * 100),
                              float(np.percentile(ds, 97.5) * 100)],
            "delta_sd_pt": float(ds.std() * 100), "p_delta_le_0": float(np.mean(ds <= 0)),
            "delta_by_k_pt": {name: float((micro_of(fc, m) - micro_of(base_c, m)) * 100)
                              for name, m in [("k=1", k == 1), ("k=2", k == 2), ("k>=3", k >= 3)]},
        }
        print(f"  fuse {which}: Δmicro {d0*100:+.3f}pt "
              f"CI [{fusion[which]['delta_ci95_pt'][0]:+.3f}, "
              f"{fusion[which]['delta_ci95_pt'][1]:+.3f}]", flush=True)
    return {"ipc_only_probe": scores, "late_fusion": fusion,
            "n_features_total": len(vocab), "text_logits": TEXT_LOGITS}


# ---------------------------------------------------------------- main

def main():
    print("raw zip 스캔…", flush=True)
    P = scan_raw()
    train, val, test = splits(P)
    mnos = sorted({m for v in P.values() for m in v["mno"]})
    assert len(mnos) == NUM_LABELS
    print(f"문서 {len(P):,} / train {len(train):,} · val {len(val):,} · test {len(test):,}")

    st = structure(P)
    print(f"\n[구조] ipc_main ∈ ipc_all {st['main_in_all']:,}/{st['n_docs']:,}"
          f" (첫 코드와 일치 {st['main_is_first_code']:,}) · 밖 {st['main_not_in_all']:,}")
    print(f"  코드 수 평균 {st['mean_codes']:.3f} · 단일 코드 문서 {st['single_code_share']:.2%}")
    print(f"  사본 2개 이상 문서 {st['n_docs_multi_copy']:,} 중 사본 간 불일치 —"
          f" ipc_main {st['copies_disagree_ipc_main']:,} · ipc_all {st['copies_disagree_ipc_all']:,}")

    br = breadth_by_shape(P, test)
    print("\n[폭 대 형상] test — 형상별 평균")
    for s in ["k=1", "within-Lno", "cross-Lno", "mixed"]:
        v = br[s]
        print(f"  {s:<11}{v['n_docs']:>6,}  코드 {v['n_codes']:.2f} · 보조 {v['n_secondary']:.2f}"
              f" · 새 서브클래스 {v['secondary_new_subclass']:.2f}"
              f" · 새 섹션 {v['secondary_new_section']:.2f}")

    cv = coverage(P, train, test)
    print("\n[커버리지] 서브클래스→top-1 Mno 연관표로 정답을 덮는 비율 (test)")
    for k, r in cv["by_k"].items():
        if r["docs"] < 20:
            continue
        print(f"  k={k}  문서 {r['docs']:>6,}  main {r['main']:.3f} · 보조 {r['secondary']:.3f}"
              f" · 전체 {r['all']:.3f} · 보조만 추가 {r['secondary_only']:.3f}")
    print(f"  k>=2에서 main이 못 덮은 라벨 {cv['multi_label_missed_by_main']:,}건 중"
          f" 보조가 회수 {cv['multi_label_missed_recovered_by_secondary']:,}"
          f" ({cv['multi_label_recovery_rate']:.3f})")

    print("\n[프로브·융합]", flush=True)
    pf = probe_and_fuse(P, train, val, test, mnos)

    (OUT / "ipc_field_analysis.json").write_text(json.dumps({
        "n": {"docs": len(P), "train": len(train), "val": len(val), "test": len(test)},
        "structure": st, "breadth_by_shape": br, "coverage": cv, **pf,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT / 'ipc_field_analysis.json'}")


if __name__ == "__main__":
    main()
