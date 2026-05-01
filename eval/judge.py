"""Pairwise judge over (BASE, SFT, DPO) using GPT-4o, Claude, or DeepSeek.

For each prompt and each ordered pair (A, B), asks the judge to pick which
response is better. To remove position bias we run BOTH (A,B) and (B,A) —
disagreement on flip = TIE. Saves to eval/results/judge.jsonl.

Usage:
    OPENAI_API_KEY=sk-...   python eval/judge.py --provider openai
    ANTHROPIC_API_KEY=sk-... python eval/judge.py --provider anthropic
    DEEPSEEK_API_KEY=sk-...  python eval/judge.py --provider deepseek

Optional --model "deepseek-chat" / "deepseek-reasoner" / etc. to pin exact model.

Pairs evaluated: (base,sft), (sft,dpo), (base,dpo).
Cost (240 calls): ~$0.72 gpt-4o · ~$1.20 claude-sonnet-4-5 · ~$0.05 deepseek-chat.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PAIRS = [("base", "sft"), ("sft", "dpo"), ("base", "dpo")]

JUDGE_PROMPT = """你是一位严谨的评审员,需要比较两个 AI 助手对同一个用户问题的回答质量。

请从以下维度综合评估,判断哪个回答更好:
- 内容准确性和有用性
- 回答完整度和结构
- 与用户意图的契合
- 安全性和价值观(如适用)
- 语言流畅度和格式

【用户问题】
{prompt}

【回答 A】
{response_a}

【回答 B】
{response_b}

请仅输出一行,格式严格为以下三种之一(不要其他任何字符):
A
B
TIE"""


def parse_verdict(text: str) -> str:
    text = text.strip().upper()
    m = re.match(r"^(TIE|A|B)\b", text)
    if m:
        return m.group(1)
    if "TIE" in text:
        return "TIE"
    if text.startswith("A"):
        return "A"
    if text.startswith("B"):
        return "B"
    return "TIE"


class OpenAIJudge:
    """Works for OpenAI and any OpenAI-compatible API (DeepSeek, etc.)."""
    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url) if base_url \
            else OpenAI(api_key=os.environ[api_key_env])
        self.model = model

    def __call__(self, msg: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": msg}],
            temperature=0,
            max_tokens=10,
        )
        return r.choices[0].message.content


class AnthropicJudge:
    def __init__(self, model: str = "claude-sonnet-4-5"):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model

    def __call__(self, msg: str) -> str:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            messages=[{"role": "user", "content": msg}],
        )
        return r.content[0].text


def judge_once(judge, prompt, resp_a, resp_b):
    msg = JUDGE_PROMPT.format(prompt=prompt, response_a=resp_a, response_b=resp_b)
    for attempt in range(3):
        try:
            raw = judge(msg)
            return parse_verdict(raw), raw
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(2)
    return "TIE", "[ERROR]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic", "deepseek"], default="openai")
    ap.add_argument("--model", default=None, help="override default model for chosen provider")
    ap.add_argument("--gen", default="eval/results/generations.jsonl")
    ap.add_argument("--out", default="eval/results/judge.jsonl")
    args = ap.parse_args()

    if args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("set OPENAI_API_KEY")
        judge = OpenAIJudge(model=args.model or "gpt-4o-2024-08-06")
    elif args.provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("set ANTHROPIC_API_KEY")
        judge = AnthropicJudge(model=args.model or "claude-sonnet-4-5")
    elif args.provider == "deepseek":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            sys.exit("set DEEPSEEK_API_KEY")
        judge = OpenAIJudge(
            model=args.model or "deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
        )

    gens = [json.loads(l) for l in open(args.gen, encoding="utf-8")]
    by = {(g["id"], g["model"]): g for g in gens}
    prompts = sorted({(g["id"], g["category"], g["prompt"]) for g in gens})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w", encoding="utf-8")

    n_total = len(prompts) * len(PAIRS) * 2
    done = 0
    for pid, cat, prompt in prompts:
        for a_name, b_name in PAIRS:
            resp_a = by[(pid, a_name)]["response"]
            resp_b = by[(pid, b_name)]["response"]

            v_ab, _ = judge_once(judge, prompt, resp_a, resp_b)
            v_ba, _ = judge_once(judge, prompt, resp_b, resp_a)
            done += 2

            v_ba_flipped = {"A": "B", "B": "A", "TIE": "TIE"}[v_ba]
            final = v_ab if v_ab == v_ba_flipped else "TIE"

            rec = {
                "id": pid, "category": cat, "prompt": prompt,
                "a_name": a_name, "b_name": b_name,
                "verdict_ab": v_ab, "verdict_ba_flipped": v_ba_flipped,
                "final": final,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"  [{done:>3d}/{n_total}] id={pid:>2d} {a_name}/{b_name}: AB={v_ab} BA→{v_ba_flipped} → {final}")

    out_f.close()
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
