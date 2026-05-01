"""Generate responses from BASE / SFT / DPO on the curated prompt set.

Saves to eval/results/generations.jsonl with one row per (prompt, model)
so downstream judge.py / jsd.py / dimension.py all read from the same file.

Each model is loaded once, run on all prompts, then released. Generation is
greedy + max_new_tokens=200 to keep outputs comparable. EOS handling: stop
at <|im_end|> (151645) or <|endoftext|> (151643); strip both from decoded
text and post-truncate at first repeating substring of length ≥ 30.
"""
import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PROMPTS

BASE_MODEL = "Qwen/Qwen3-8B-Base"
SFT_PATH = "checkpoints/sft"
DPO_PATH = "checkpoints/dpo"
SYSTEM = "你是一个有用、诚实、无害的助手。"
MAX_NEW = 200


def fmt(prompt: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def truncate_repeats(text: str, min_len: int = 30) -> str:
    """If a substring of length ≥ min_len repeats consecutively, cut at first repeat.

    Handles the trl 1.x EOS-learning failure: the model produces good content
    then loops a phrase. We keep the first instance and drop everything after.
    """
    for L in range(min_len, len(text) // 2 + 1):
        for i in range(len(text) - 2 * L + 1):
            if text[i : i + L] == text[i + L : i + 2 * L]:
                return text[: i + L].rstrip()
    return text.rstrip()


def load_tokenizer():
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


def generate(model, tok, prompt: str, eos_ids: list[int]) -> str:
    text = fmt(prompt)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=eos_ids[1],
            eos_token_id=eos_ids,
        )
    decoded = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=False)
    raw = decoded.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    return truncate_repeats(raw)


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/generations.jsonl")
    ap.add_argument("--models", nargs="+", default=["base", "sft", "dpo"],
                    choices=["base", "sft", "dpo"])
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w", encoding="utf-8")

    tok = load_tokenizer()
    eos_ids = [
        tok.convert_tokens_to_ids("<|im_end|>"),
        tok.convert_tokens_to_ids("<|endoftext|>"),
    ]
    print(f"eos_ids = {eos_ids}")

    loaders = {"base": load_base, "sft": load_sft, "dpo": load_dpo}
    for name in args.models:
        print(f"\n=== Loading {name.upper()} model ===")
        model = loaders[name]()
        for row in PROMPTS:
            response = generate(model, tok, row["prompt"], eos_ids)
            rec = {
                "id": row["id"],
                "category": row["category"],
                "model": name,
                "prompt": row["prompt"],
                "response": response,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"  [{row['id']:>2d}/{len(PROMPTS)}] {name} {row['category']:<10s} {response[:60]!r}")
        free(model)

    out_f.close()
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
