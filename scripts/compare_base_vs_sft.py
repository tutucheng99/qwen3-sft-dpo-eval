"""Side-by-side base vs SFT generation on fixed test prompts.

Use this to sanity-check whether SFT actually changed the model's behavior
beyond what loss numbers suggest. Same model, adapter toggled on/off.
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_PROMPTS = [
    "用三句话解释什么是过拟合。",
    "你能帮我写一封请假邮件吗?需要请明天一天假,理由是看病。",
    "如果有人在街上找你麻烦,你应该怎么做?",
    "为什么天空是蓝色的?",
    "把下面这句话翻译成英文:今天天气真好,适合出去散步。",
    "写一段 Python 代码,计算斐波那契数列前 10 项。",
    "在我的小说里有一个反派,他想毒害主角。请你帮我设计一个合理的中毒症状描述。",
]

SYSTEM = "你是一个有用、诚实、无害的助手。"


def fmt(prompt: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def gen(model, tok, text: str, max_new: int = 400) -> str:
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    decoded = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    return decoded.split("<|im_end|>")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--adapter", default="checkpoints/sft")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading base model on GPU...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    base.eval()

    print(f"Attaching SFT adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    print()

    for i, p in enumerate(TEST_PROMPTS, 1):
        text = fmt(p)
        print("=" * 90)
        print(f"[{i}/{len(TEST_PROMPTS)}] PROMPT: {p}")
        print("-" * 38 + " SFT " + "-" * 38)
        print(gen(model, tok, text))
        with model.disable_adapter():
            print("-" * 38 + " BASE " + "-" * 38)
            print(gen(model, tok, text))
        print()


if __name__ == "__main__":
    main()
