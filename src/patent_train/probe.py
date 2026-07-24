"""OOM 탐침 — worst-case 배치로 실제 스텝을 돌려 micro_batch 상한을 튜닝.

새 모델·길이·GPU로 실험을 옮길 때 안전한 micro_batch를 실측한다.
worst case = 모든 시퀀스가 seq_len이고 mask가 전부 1(패딩 0) — group_by_length가 초반에 만드는 배치.
lr=0으로 AdamW state만 상주시켜 정상상태 peak를 재므로 사전학습 가중치는 훼손하지 않는다.
"""

import torch
import torch.nn.functional as F


def probe_batches(model, vocab_size, seq_len, train_mb=(16, 32, 64, 96, 128, 160),
                  eval_mb=(32, 64, 128, 256, 512), num_labels: int = 188):
    """train·eval 각 micro_batch 후보의 peak 메모리를 출력하고, OOM 후보부터 중단한다."""
    dev = model.device
    opt = torch.optim.AdamW(model.parameters(), lr=0.0)   # lr=0 → state만 할당, 가중치 불변

    def _mk(mb):
        ids = torch.randint(5, vocab_size, (mb, seq_len), device=dev)
        return ids, torch.ones_like(ids), torch.zeros(mb, num_labels, device=dev)

    model.train()
    for mb in train_mb:
        try:
            ids, mask, labels = _mk(mb)
            for step in (0, 1):                       # step0: AdamW state 할당 / step1: 정상상태 peak
                if step == 1:
                    torch.cuda.reset_peak_memory_stats()
                with torch.autocast("cuda", dtype=torch.bfloat16):   # Trainer(bf16=True)와 동일 경로
                    out = model(input_ids=ids, attention_mask=mask)
                    loss = F.binary_cross_entropy_with_logits(out.logits.float(), labels)
                loss.backward()                                      # backward는 autocast 밖
                opt.step()
                opt.zero_grad(set_to_none=True)
            print(f"train micro={mb:>3}: peak {torch.cuda.max_memory_allocated()/1e9:5.1f} GB  OK")
        except torch.cuda.OutOfMemoryError:
            print(f"train micro={mb:>3}: OOM")
            opt.zero_grad(set_to_none=True)
            break                                     # 이후 후보는 자명하게 OOM
        finally:
            torch.cuda.empty_cache()

    model.eval()
    for mb in eval_mb:                                # 옵티마이저 state 상주 조건에서 측정
        try:
            torch.cuda.reset_peak_memory_stats()
            ids, mask, _ = _mk(mb)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=ids, attention_mask=mask)
            print(f"eval  micro={mb:>3}: peak {torch.cuda.max_memory_allocated()/1e9:5.1f} GB  OK")
        except torch.cuda.OutOfMemoryError:
            print(f"eval  micro={mb:>3}: OOM")
            break
        finally:
            torch.cuda.empty_cache()

    del opt                                           # Trainer가 자체 옵티마이저를 새로 만들도록 정리
    model.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()              # train() peak를 깨끗하게 재측정하기 위함
