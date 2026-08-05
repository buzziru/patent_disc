"""입력 길이 분포와 `max_len`별 훈련 비용 — 배포 런의 창 크기 결정 근거.

배포 모델의 `max_len`을 고르려면 두 축을 함께 봐야 한다 — **무엇을 잃는가**(절단되는 문서·토큰)와
**무엇을 아끼는가**(연산량·메모리). 이 스크립트가 둘을 같은 표에 놓는다(서술은
`docs/experiments/final-run.md`).

주의 — 패딩 낭비는 이미 제거돼 있다. 훈련 경로는 세 겹으로 실토큰만 계산한다:
배치별 동적 패딩(`data.MultiLabelCollator`) · 길이 그룹 배칭(`train_sampling_strategy`) ·
ModernBERT unpadding(flash-attention-2). 따라서 `max_len`을 키워도 비용이 창 크기에 비례하지
않으며(8192가 512의 16배가 아니라 2.02배), **`max_len`이 실제로 제약하는 것은 최악 배치의
메모리 = `micro_batch` 상한**이다.

FLOPs 모델(층당 forward, 문서 길이 L):
  선형    (4·d² + 3·d·d_ff)·L        # QKV·출력 projection + GeGLU FFN
  어텐션  4·d·L·L_eff                # local은 L_eff=window, global은 L_eff=L
상수는 비율만 쓰므로 forward/backward 배수는 생략한다.

실행: `uv run python scripts/length_cost.py`
산출: `output/length_cost.json`

참조하는 HF 데이터셋은 공개 배포하지 않는다 — 재생성 절차는
`docs/data/data-pipeline.md`「가공 데이터셋은 배포하지 않는다 — 재현 경로」.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
ROOT = Path(os.environ["DATA_ROOT"])
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from datasets import load_dataset                      # noqa: E402  (HF_HOME 설정 뒤 import)
from transformers import AutoConfig                    # noqa: E402

OUT = ROOT / "output"
BACKBONE = "skt/A.X-Encoder-base"
REV = "9708f9c404ace91efd25c06fac2d73413616f4ef"       # backbones.BACKBONES["axenc"].rev
TOKENIZED = "ingyoun/patent-clean-text-modernbert-tokenized"
CANDIDATES = (512, 1024, 2048, 4096, 8192)
EFF_BATCH = 128                                        # 길이 그룹 배칭의 단위(= LengthGroupedSampler)


def token_lengths(split="train"):
    """토큰화 데이터셋의 문서별 길이 — Arrow 리스트 오프셋에서 바로 읽는다(행 미실체화)."""
    ds = load_dataset(TOKENIZED, split=split)
    col = ds.data.table.column("input_ids").combine_chunks()
    return np.asarray(pc.list_value_length(col)).astype(np.float64)


def main():
    cfg = AutoConfig.from_pretrained(BACKBONE, revision=REV)
    d, d_ff, n_layer = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    every = getattr(cfg, "global_attn_every_n_layers", 3)
    window = getattr(cfg, "local_attention", 128)
    window = window if isinstance(window, int) else window[0]
    n_global = len([i for i in range(n_layer) if i % every == 0])
    lin_per_token = 4 * d * d + 3 * d * d_ff

    L = token_lengths("train")
    n = len(L)
    dist = {f"p{q}": float(np.percentile(L, q)) for q in (25, 50, 75, 90, 95, 99)}
    print(f"[분포] train {n:,}문서 · 평균 {L.mean():.0f} · 중앙 {np.median(L):.0f}"
          f" · p90 {dist['p90']:.0f} · p99 {dist['p99']:.0f} · 최대 {L.max():,.0f} 토큰")

    rows, cost = {}, {}
    for m in CANDIDATES:
        Lc = np.minimum(L, m)
        lin = lin_per_token * Lc.sum() * n_layer
        att = 4 * d * ((Lc * window).sum() * (n_layer - n_global) + (Lc ** 2).sum() * n_global)
        cost[m] = lin + att
        rows[m] = {
            "truncated_docs": float((L > m).mean()),
            "lost_tokens": float(1 - Lc.sum() / L.sum()),
            "total_tokens": int(Lc.sum()),
            "attention_share": float(att / (lin + att)),
            "worst_seq": int(Lc.max()),
        }
    for m in CANDIDATES:
        rows[m]["cost_vs_512"] = round(cost[m] / cost[CANDIDATES[0]], 3)
        rows[m]["cost_vs_8192"] = round(cost[m] / cost[CANDIDATES[-1]], 3)
        # micro_batch는 최악 배치의 메모리가 정한다 — 창을 줄이면 그만큼 여유가 생긴다
        rows[m]["micro_batch_headroom_vs_8192"] = round(CANDIDATES[-1] / m, 2)

    print(f"\n{'max_len':>8}{'절단문서':>9}{'유실토큰':>9}{'총토큰(M)':>11}"
          f"{'어텐션%':>8}{'비용/8192':>10}{'micro_batch 여유':>16}")
    for m in CANDIDATES:
        r = rows[m]
        print(f"{m:>8}{r['truncated_docs']:>8.1%}{r['lost_tokens']:>9.2%}"
              f"{r['total_tokens']/1e6:>11.1f}{r['attention_share']:>8.1%}"
              f"{r['cost_vs_8192']:>10.2f}x{r['micro_batch_headroom_vs_8192']:>15.0f}x")

    # 길이 그룹 배칭이 실제로 패딩을 얼마나 없애는가 — 정렬 후 eff_batch 묶음의 채움률
    fill = {}
    for m in (2048, 4096, 8192):
        s = np.sort(np.minimum(L, m))[: (n // EFF_BATCH) * EFF_BATCH].reshape(-1, EFF_BATCH)
        fill[m] = {
            "mean_over_max_in_group": float((s.mean(1) / s.max(1)).mean()),
            "median_group_max": float(np.median(s.max(1))),
            "median_group_max_over_window": float(np.median(s.max(1)) / m),
        }
    print("\n[길이 그룹 배칭] 정렬 후 128문서 묶음")
    for m, f in fill.items():
        print(f"  max_len {m:>5}: 묶음 내 평균/최장 {f['mean_over_max_in_group']:.1%}"
              f" · 묶음 최장의 중앙 {f['median_group_max']:.0f} 토큰"
              f"(창의 {f['median_group_max_over_window']:.1%})")

    payload = {
        "question": "배포 런의 max_len을 무엇으로 둘 것인가 — 잃는 것(절단)과 아끼는 것(비용·메모리).",
        "backbone": {"model": BACKBONE, "rev": REV, "hidden": d, "ffn": d_ff,
                     "layers": n_layer, "global_every": every, "n_global": n_global,
                     "local_window": window, "max_position_embeddings": cfg.max_position_embeddings},
        "dataset": {"id": TOKENIZED, "split": "train", "n_docs": n},
        "length": {"mean": float(L.mean()), "max": float(L.max()), **dist},
        "by_max_len": {str(m): rows[m] for m in CANDIDATES},
        "length_grouped_batching": {str(m): f for m, f in fill.items()},
        "eff_batch": EFF_BATCH,
        "script": "scripts/length_cost.py",
        "note": "비용은 FLOPs 비율이며 실측 벽시계가 아니다. 패딩은 동적 패딩·길이 그룹 배칭·"
                "ModernBERT unpadding으로 이미 제거돼 있어 창 크기에 비례하지 않는다 — "
                "max_len이 실제로 제약하는 것은 최악 배치 메모리 = micro_batch 상한이다.",
    }
    path = OUT / "length_cost.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
