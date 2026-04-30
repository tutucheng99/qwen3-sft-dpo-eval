"""Merge LoRA adapter back into base for vLLM serving."""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stack", choices=["sft", "sft+dpo"], default="sft")
    ap.add_argument("--sft_adapter", help="required if --stack sft+dpo")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    if args.stack == "sft+dpo":
        assert args.sft_adapter, "--sft_adapter required for sft+dpo"
        model = PeftModel.from_pretrained(model, args.sft_adapter).merge_and_unload()

    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tok.save_pretrained(args.out)
    print(f"merged → {args.out}")


if __name__ == "__main__":
    main()
