"""COIG-CQIA → SFT JSONL with ChatML formatting.

Subset weights are hand-picked: prefer instruction-rich, knowledge-grounded
subsets (ruozhiba, zhihu, exam, wiki, human_value) over noisy social ones.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

SUBSET_QUOTAS = {
    "ruozhiba": 800,
    "zhihu": 1500,
    "exam": 1000,
    "wiki": 800,
    "human_value": 600,
    "logi_qa": 500,
    "wikihow": 800,
    "douban": 400,
    "coig_pc": 1200,
    "segmentfault": 600,
    "finance": 400,
}

SYSTEM_PROMPT = "你是一个有用、诚实、无害的助手。"


def to_chatml(instruction: str, inp: str, output: str) -> str:
    user = f"{instruction}\n\n{inp}".strip() if inp else instruction
    # No trailing <|im_end|>: SFTTrainer appends tokenizer.eos_token (which we
    # override to <|im_end|>) so duplicating it here would teach a 2x EOS pattern.
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}"
    )


def is_clean(row: dict, min_out: int, max_out: int) -> bool:
    inst = (row.get("instruction") or "").strip()
    out = (row.get("output") or "").strip()
    if not inst or not out:
        return False
    if len(out) < min_out or len(out) > max_out:
        return False
    if out.count("\n") > 60:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--total", type=int, default=8000)
    ap.add_argument("--eval_frac", type=float, default=0.05)
    ap.add_argument("--min_out_chars", type=int, default=20)
    ap.add_argument("--max_out_chars", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = args.total / sum(SUBSET_QUOTAS.values())
    quotas = {k: max(1, int(v * scale)) for k, v in SUBSET_QUOTAS.items()}

    pool: list[str] = []
    for subset, n in quotas.items():
        try:
            ds = load_dataset("m-a-p/COIG-CQIA", subset, split="train")
        except Exception as e:
            print(f"[skip] {subset}: {e}")
            continue
        rows = [r for r in ds if is_clean(r, args.min_out_chars, args.max_out_chars)]
        random.shuffle(rows)
        picked = rows[:n]
        for r in picked:
            pool.append(to_chatml(r["instruction"], r.get("input", ""), r["output"]))
        print(f"[ok]   {subset:<14s} kept {len(picked):>5d} / cleaned {len(rows):>5d}")

    random.shuffle(pool)
    n_eval = max(50, int(len(pool) * args.eval_frac))
    eval_set, train_set = pool[:n_eval], pool[n_eval:]

    for name, items in [("sft_train.jsonl", train_set), ("sft_eval.jsonl", eval_set)]:
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for s in items:
                f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
        print(f"wrote {len(items):>5d} → {out_dir / name}")


if __name__ == "__main__":
    main()
