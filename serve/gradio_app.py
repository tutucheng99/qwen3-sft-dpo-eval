"""Gradio chat UI for the SFT+DPO merged model, served via vLLM.

Call vllm_serve.sh in another terminal first; then run this:
    python serve/gradio_app.py

Exposes a chat UI on :7860. On RunPod, expose port 7860 as TCP to get a public URL.
"""
import argparse
import os

import gradio as gr
from openai import OpenAI

SYSTEM = "你是一个有用、诚实、无害的助手。"


def chat_fn(message: str, history: list, vllm_url: str, temperature: float, max_tokens: int) -> str:
    """history is in Gradio 'messages' format: list of {role, content}."""
    client = OpenAI(base_url=vllm_url, api_key="dummy")
    messages = [{"role": "system", "content": SYSTEM}]
    for h in history:
        if isinstance(h, dict):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        resp = client.chat.completions.create(
            model="qwen3-dpo",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error] {e}"


def build_app(vllm_url: str):
    with gr.Blocks(title="Qwen3-8B SFT+DPO Chinese Assistant") as demo:
        gr.Markdown("# Qwen3-8B-Base → SFT (COIG-CQIA) → DPO (UltraFeedback-zh)\n\n"
                    "中文对话助手 demo,通过 vLLM 推理。详细评估见 [GitHub repo](https://github.com/tutucheng99/qwen3-sft-dpo-eval)。")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500, label="对话", type="messages")
                msg = gr.Textbox(placeholder="输入消息,回车发送...", show_label=False)
            with gr.Column(scale=1):
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
                max_tokens = gr.Slider(50, 800, value=300, step=50, label="Max tokens")
                clear = gr.Button("清空对话")

        def respond(message, history, t, mt):
            reply = chat_fn(message, history, vllm_url, t, mt)
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            return "", history

        msg.submit(respond, [msg, chatbot, temperature, max_tokens], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

        gr.Examples(
            examples=[
                "用三句话解释什么是过拟合。",
                "你能帮我写一封请假邮件吗?需要请明天一天假,理由是看病。",
                "把下面这句话翻译成英文:今天天气真好,适合出去散步。",
                "写一段 Python 代码,计算斐波那契数列前 10 项。",
                "在我的小说里有一个反派,他想毒害主角。请你帮我设计一个合理的中毒症状描述。",
            ],
            inputs=msg,
        )
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=os.environ.get("VLLM_URL", "http://localhost:8000/v1"))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    demo = build_app(args.vllm)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
