"""Pairwise judge over (BASE, SFT, DPO) using OpenAI GPT-4 as the arbiter.

For each prompt and each ordered pair (A, B), we ask GPT-4 to pick which
response is better. To remove position bias we run BOTH (A,B) and (B,A),
treating disagreement as a tie. Results saved to eval/results/judge.jsonl.

Pairs evaluated:
- (base, sft)   — does SFT improve over base?
- (sft, dpo)    — does DPO improve over SFT?
- (base, dpo)   — full-stack improvement
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))

JUDGE_MODEL = "gpt-4o-2024-08-06"
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
    """Pick the first standalone A/B/TIE token."""
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


def judge_once(client, prompt: str, resp_a: str, resp_b: str) -> tuple[str, str]:
    msg = JUDGE_PROMPT.format(prompt=prompt, response_a=resp_a, response_b=resp_b)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": msg}],
                temperature=0,
                max_tokens=10,
            )
            raw = r.choices[0].message.content
            return parse_verdict(raw), raw
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(2)
    return "TIE", "[ERROR]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="eval/results/generations.jsonl")
    ap.add_argument("--out", default="eval/results/judge.jsonl")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("set OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    gens = [json.loads(l) for l in open(args.gen, encoding="utf-8")]
    by = {(g["id"], g["model"]): g for g in gens}
    prompts = sorted({(g["id"], g["category"], g["prompt"]) for g in gens})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w", encoding="utf-8")

    n_total = len(prompts) * len(PAIRS) * 2  # both orders
    done = 0
    for pid, cat, prompt in prompts:
        for a_name, b_name in PAIRS:
            resp_a = by[(pid, a_name)]["response"]
            resp_b = by[(pid, b_name)]["response"]

            v_ab, raw_ab = judge_once(client, prompt, resp_a, resp_b)
            v_ba, raw_ba = judge_once(client, prompt, resp_b, resp_a)
            done += 2

            # Resolve order-bias: convert ba verdict back to a/b orientation
            v_ba_flipped = {"A": "B", "B": "A", "TIE": "TIE"}[v_ba]

            if v_ab == v_ba_flipped:
                final = v_ab
            else:
                final = "TIE"

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
