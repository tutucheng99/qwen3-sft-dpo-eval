"""Lightweight FastAPI server for the merged SFT+DPO model.

Uses plain transformers (no vLLM dependency). Throughput is lower (~5-15 tok/s
on B200 single-stream) but adequate for a Gradio chat demo. Avoids the
torch-version-pin headache that vLLM brings.

Endpoints:
  POST /v1/chat/completions  — OpenAI-compatible (so gradio_app.py works unchanged)
  GET  /healthz              — liveness check

Run from the training venv:
  python serve/hf_serve.py --model /workspace/merged/dpo --port 8000
"""
import argparse
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = None
TOK = None
EOS_IDS: list[int] = []
MODEL_NAME = "qwen3-dpo"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message]
    max_tokens: int = 300
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[list[str]] = None


def messages_to_chatml(messages: list[Message]) -> str:
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def setup(model_path: str):
    global MODEL, TOK, EOS_IDS
    print(f"Loading {model_path}...")
    TOK = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if TOK.pad_token is None:
        TOK.pad_token = TOK.eos_token
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    MODEL.eval()
    EOS_IDS = [
        TOK.convert_tokens_to_ids("<|im_end|>"),
        TOK.convert_tokens_to_ids("<|endoftext|>"),
    ]
    print("Ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    text = messages_to_chatml(req.messages)
    ids = TOK(text, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        out = MODEL.generate(
            **ids,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=max(req.temperature, 1e-3),
            top_p=req.top_p,
            pad_token_id=EOS_IDS[1],
            eos_token_id=EOS_IDS,
        )
    decoded = TOK.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=False)
    response = decoded.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response},
            "finish_reason": "stop",
        }],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/merged/dpo")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    setup(args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
