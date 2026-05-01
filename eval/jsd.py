"""Jensen-Shannon Divergence between BASE / SFT / DPO output token distributions.

For each prompt, get N=8 stochastic samples from each model (temperature=0.7),
build a token-frequency distribution per model, then compute pairwise JSD.

The methodology is inherited from the user's prior bridge-game policy work,
where JSD on action distributions revealed how policy refinements concentrate
or spread probability mass in non-obvious ways.

Output:
  - eval/results/samples.jsonl  — N samples per (prompt, model)
  - eval/results/jsd.json       — pairwise JSD scores + per-prompt and overall
"""
import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PROMPTS

BASE_MODEL = "Qwen/Qwen3-8B-Base"
SFT_PATH = "checkpoints/sft"
DPO_PATH = "checkpoints/dpo"
SYSTEM = "你是一个有用、诚实、无害的助手。"


def fmt(prompt: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_eos(tok):
    return [
        tok.convert_tokens_to_ids("<|im_end|>"),
        tok.convert_tokens_to_ids("<|endoftext|>"),
    ]


def load_tok():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )


def load_sft():
    base = load_base()
    return PeftModel.from_pretrained(base, SFT_PATH).eval()


def load_dpo():
    base = load_base()
    base = PeftModel.from_pretrained(base, SFT_PATH).merge_and_unload()
    return PeftModel.from_pretrained(base, DPO_PATH).eval()


def free(m):
    del m
    gc.collect()
    torch.cuda.empty_cache()


def sample_n(model, tok, prompt: str, eos_ids, n: int = 8, max_new: int = 150):
    text = fmt(prompt)
    ids = tok(text, return_tensors="pt").to(model.device)
    samples = []
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=eos_ids[1],
                eos_token_id=eos_ids,
            )
            decoded = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            samples.append(decoded)
    return samples


def kl(p: np.ndarray, q: np.ndarray, eps=1e-12) -> float:
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * (kl(p, m) + kl(q, m))


def token_dist(samples: list[str], tok, vocab_size: int) -> np.ndarray:
    counter = Counter()
    for s in samples:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        counter.update(ids)
    arr = np.zeros(vocab_size, dtype=np.float64)
    for tid, c in counter.items():
        if 0 <= tid < vocab_size:
            arr[tid] = c
    total = arr.sum()
    if total > 0:
        arr /= total
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_out", default="eval/results/samples.jsonl")
    ap.add_argument("--jsd_out", default="eval/results/jsd.json")
    ap.add_argument("--n_samples", type=int, default=8)
    args = ap.parse_args()

    tok = load_tok()
    eos_ids = build_eos(tok)
    vocab_size = tok.vocab_size + 1000  # padding for special tokens

    Path(args.samples_out).parent.mkdir(parents=True, exist_ok=True)
    samples_f = open(args.samples_out, "w", encoding="utf-8")

    by_model_prompt: dict = {"base": {}, "sft": {}, "dpo": {}}

    loaders = {"base": load_base, "sft": load_sft, "dpo": load_dpo}
    for name in ["base", "sft", "dpo"]:
        print(f"\n=== Sampling from {name.upper()} ({args.n_samples} per prompt) ===")
        model = loaders[name]()
        for row in PROMPTS:
            samples = sample_n(model, tok, row["prompt"], eos_ids, n=args.n_samples)
            by_model_prompt[name][row["id"]] = samples
            for s in samples:
                samples_f.write(json.dumps({
                    "id": row["id"], "model": name, "sample": s,
                }, ensure_ascii=False) + "\n")
            samples_f.flush()
            print(f"  [{row['id']:>2d}/{len(PROMPTS)}] {name}")
        free(model)

    samples_f.close()

    # Compute pairwise JSD per prompt + overall
    print("\n=== Computing JSD ===")
    pairs = [("base", "sft"), ("sft", "dpo"), ("base", "dpo")]
    per_prompt: dict = {f"{a}_{b}": {} for a, b in pairs}
    overall: dict = {}

    for a, b in pairs:
        scores = []
        for row in PROMPTS:
            pid = row["id"]
            p = token_dist(by_model_prompt[a][pid], tok, vocab_size)
            q = token_dist(by_model_prompt[b][pid], tok, vocab_size)
            j = jsd(p, q)
            per_prompt[f"{a}_{b}"][pid] = j
            scores.append(j)
        overall[f"{a}_{b}"] = {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
        }
        print(f"  JSD({a}, {b}): mean={np.mean(scores):.4f}  median={np.median(scores):.4f}  std={np.std(scores):.4f}")

    out = {"overall": overall, "per_prompt": per_prompt}
    with open(args.jsd_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.jsd_out}")


if __name__ == "__main__":
    main()
