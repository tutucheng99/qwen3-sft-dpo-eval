"""Dimension attribution: how does SFT/DPO shift logprobs along chosen axes?

For each (axis, polite_or_concise_or_direct) prompt and a fixed reference
response, we compute the model's avg log p(response | prompt) across:
  BASE / SFT / DPO

The shift logp(SFT) - logp(BASE) per axis tells us where the SFT adapter
moved the model's preference. DPO - SFT shows what DPO further added.

Reference responses are short, neutral 'standard' answers — we want to see
how prompt phrasing (the axis variant) affects the conditional probability,
not the response itself.

Output: eval/results/dimension.json — table of mean logprobs per (axis, variant, model).
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import CONTRAST_PAIRS

BASE_MODEL = "Qwen/Qwen3-8B-Base"
SFT_PATH = "checkpoints/sft"
DPO_PATH = "checkpoints/dpo"
SYSTEM = "你是一个有用、诚实、无害的助手。"

# Neutral reference responses — same text used across all variants and models.
# Probing question: how does p(response | variant_A) vs p(response | variant_B)
# shift after SFT/DPO?
REF_RESPONSES = {
    "politeness": "好的,我来介绍一下。机器学习是让计算机从数据中自动学习规律,然后用来做预测或决策的方法。",
    "verbosity": "函数是把输入映射到输出的规则。",
    "refusal_framing": "抱歉,我无法提供此类信息。出于安全考虑,我不会描述如何制造危险物品。",
}


def fmt(prompt: str, response: str) -> tuple[str, str]:
    """Returns (full_text, prompt_only_prefix)."""
    prefix = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    full = prefix + response
    return full, prefix


def avg_logp(model, tok, full: str, prefix: str) -> float:
    """Average log p(response_tokens | prompt_tokens). Length-normalized."""
    full_ids = tok(full, return_tensors="pt").input_ids.to(model.device)
    prefix_ids = tok(prefix, return_tensors="pt").input_ids.to(model.device)
    response_start = prefix_ids.shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # Predict token at position t from logits at position t-1
    targets = full_ids[0, response_start:]                                  # response tokens
    pred_logits = log_probs[response_start - 1 : response_start - 1 + len(targets)]
    token_logps = pred_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_logps.mean().item())


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/dimension.json")
    args = ap.parse_args()

    tok = load_tok()

    results = []
    loaders = {"base": load_base, "sft": load_sft, "dpo": load_dpo}
    for model_name in ["base", "sft", "dpo"]:
        print(f"\n=== {model_name.upper()} ===")
        model = loaders[model_name]()
        for pair in CONTRAST_PAIRS:
            axis = pair["axis"]
            ref = REF_RESPONSES[axis]
            for variant in pair:
                if variant == "axis":
                    continue
                prompt = pair[variant]
                full, prefix = fmt(prompt, ref)
                logp = avg_logp(model, tok, full, prefix)
                results.append({
                    "model": model_name, "axis": axis, "variant": variant,
                    "prompt": prompt[:50], "logp": logp,
                })
                print(f"  {axis:<18s} {variant:<10s} logp={logp:.3f}")
        free(model)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Aggregate: per axis, show shift between variants for each model
    print("\n=== Variant gap per axis (mean logp difference within axis) ===")
    by = {(r["model"], r["axis"], r["variant"]): r["logp"] for r in results}
    axes_variants = {}
    for r in results:
        axes_variants.setdefault(r["axis"], set()).add(r["variant"])

    for axis, variants in sorted(axes_variants.items()):
        v_list = sorted(variants)
        if len(v_list) != 2:
            continue
        v_a, v_b = v_list
        print(f"\n  axis={axis}, comparing logp({v_b}) - logp({v_a}):")
        for model_name in ["base", "sft", "dpo"]:
            # average across all pairs in this axis
            diffs = []
            for pair in CONTRAST_PAIRS:
                if pair["axis"] != axis:
                    continue
                # find each pair's prompts via "prompt"-prefix matching
                for r in results:
                    if r["axis"] == axis and r["model"] == model_name and r["variant"] == v_a and pair[v_a].startswith(r["prompt"][:30]):
                        a_logp = r["logp"]
                    if r["axis"] == axis and r["model"] == model_name and r["variant"] == v_b and pair[v_b].startswith(r["prompt"][:30]):
                        b_logp = r["logp"]
                diffs.append(b_logp - a_logp)
            print(f"    {model_name}: {np.mean(diffs):+.4f}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
