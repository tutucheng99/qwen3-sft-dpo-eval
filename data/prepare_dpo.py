"""UltraFeedback-chinese → DPO JSONL (prompt / chosen / rejected) in ChatML."""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

SYSTEM_PROMPT = "你是一个有用、诚实、无害的助手。"


def fmt_prompt(user_msg: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def fmt_response(text: str) -> str:
    # No trailing <|im_end|>: DPOTrainer appends tokenizer.eos_token (which we
    # override to <|im_end|> in dpo_train.py).
    return text


def extract_pair(row: dict) -> tuple[str, str, str] | None:
    instr = (row.get("instruction") or row.get("prompt") or "").strip()
    if not instr:
        return None
    completions = row.get("completions") or row.get("responses") or []
    if completions and isinstance(completions, list) and isinstance(completions[0], dict):
        scored = [(c.get("overall_score") or c.get("score") or 0, c.get("response") or c.get("text") or "")
                  for c in completions if (c.get("response") or c.get("text"))]
        scored = [(s, t) for s, t in scored if t.strip()]
        if len(scored) < 2:
            return None
        scored.sort(key=lambda x: x[0])
        rejected = scored[0][1].strip()
        chosen = scored[-1][1].strip()
        if scored[-1][0] - scored[0][0] < 1:
            return None
    else:
        chosen = (row.get("chosen") or "").strip()
        rejected = (row.get("rejected") or "").strip()
        if not chosen or not rejected or chosen == rejected:
            return None
    return instr, chosen, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--total", type=int, default=8000)
    ap.add_argument("--eval_frac", type=float, default=0.05)
    ap.add_argument("--min_chars", type=int, default=15)
    ap.add_argument("--max_chars", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("opencsg/UltraFeedback-chinese", split="train")
    print(f"loaded {len(ds)} raw rows")

    samples: list[dict] = []
    for row in ds:
        pair = extract_pair(row)
        if not pair:
            continue
        prompt, chosen, rejected = pair
        if not (args.min_chars <= len(chosen) <= args.max_chars):
            continue
        if not (args.min_chars <= len(rejected) <= args.max_chars):
            continue
        samples.append({
            "prompt": fmt_prompt(prompt),
            "chosen": fmt_response(chosen),
            "rejected": fmt_response(rejected),
        })

    print(f"kept {len(samples)} valid pairs")
    random.shuffle(samples)
    samples = samples[: args.total]

    n_eval = max(50, int(len(samples) * args.eval_frac))
    eval_set, train_set = samples[:n_eval], samples[n_eval:]

    for name, items in [("dpo_train.jsonl", train_set), ("dpo_eval.jsonl", eval_set)]:
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for s in items:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"wrote {len(items):>5d} → {out_dir / name}")


if __name__ == "__main__":
    main()
