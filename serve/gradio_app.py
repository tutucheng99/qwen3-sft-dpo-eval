"""Gradio chat UI for the SFT+DPO merged model, served via hf_serve.py.

Uses gr.ChatInterface (Gradio's higher-level wrapper) — handles the messages
format and history plumbing automatically. Run hf_serve.py first for the
backend, then run this:
    python serve/gradio_app.py
"""
import argparse
import os

import gradio as gr
from openai import OpenAI

SYSTEM = "你是一个有用、诚实、无害的助手。"
EXAMPLES = [
    "用三句话解释什么是过拟合。",
    "你能帮我写一封请假邮件吗?需要请明天一天假,理由是看病。",
    "把下面这句话翻译成英文:今天天气真好,适合出去散步。",
    "写一段 Python 代码,计算斐波那契数列前 10 项。",
    "在我的小说里有一个反派,他想毒害主角。请你帮我设计一个合理的中毒症状描述。",
]


def make_handler(client: OpenAI):
    def chat(message: str, history: list) -> str:
        msgs = [{"role": "system", "content": SYSTEM}]
        for h in history:
            if isinstance(h, dict) and "role" in h:
                msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": message})

        try:
            resp = client.chat.completions.create(
                model="qwen3-dpo",
                messages=msgs,
                max_tokens=400,
                temperature=0.7,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error] {e}"
    return chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("BACKEND_URL", "http://localhost:8000/v1"))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    client = OpenAI(base_url=args.backend, api_key="dummy")
    chat = make_handler(client)

    demo = gr.ChatInterface(
        fn=chat,
        title="Qwen3-8B-Base → SFT (COIG-CQIA) → DPO (UltraFeedback-zh)",
        description="中文对话助手 demo · 详细评估见 [GitHub repo](https://github.com/tutucheng99/qwen3-sft-dpo-eval)",
    )
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
