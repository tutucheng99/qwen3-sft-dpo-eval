"""Test if the model predicts <|im_end|> or <|endoftext|> at end-of-answer position.

Usage:
    python scripts/diag_eos.py                    # SFT
    python scripts/diag_eos.py --dpo              # DPO (loads SFT then DPO on top)
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-8B-Base"
SFT = "checkpoints/sft"
DPO = "checkpoints/dpo"
SYSTEM = "你是一个有用、诚实、无害的助手。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", action="store_true")
    args = ap.parse_args()

    target = DPO if args.dpo else SFT
    tok = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
    print(f"adapter={target}  eos={tok.eos_token!r} id={tok.eos_token_id}")

    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    if args.dpo:
        print("merging SFT into base...")
        base = PeftModel.from_pretrained(base, SFT).merge_and_unload()

    model = PeftModel.from_pretrained(base, target)
    model.eval()
    label_active = "DPO" if args.dpo else "SFT"
    label_off = "SFT-merged" if args.dpo else "BASE"

    text = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n你好<|im_end|>\n"
        f"<|im_start|>assistant\n你好,我是 AI 助手,有什么可以帮你的吗?"
    )
    ids = tok(text, return_tensors="pt").to("cuda")

    def show(label):
        with torch.no_grad():
            p = torch.softmax(model(**ids).logits[0, -1, :], dim=0)
        print(f"\n=== {label} top-5 at end-of-answer ===")
        for v, i in zip(*torch.topk(p, 5)):
            tok_id = i.item()
            marker = " ⭐" if tok_id in (151645, 151643) else ""
            print(f"  id={tok_id:>6d}  p={v.item():.4f}  {tok.decode([tok_id])!r}{marker}")

    show(label_active)
    with model.disable_adapter():
        show(label_off)


if __name__ == "__main__":
    main()
