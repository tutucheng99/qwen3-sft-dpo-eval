"""SFT with LoRA on Qwen3-8B-Base, ChatML-formatted JSONL."""
import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    Path(cfg["train"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg["model"]["name_or_path"])

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name_or_path"],
        dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
        task_type="CAUSAL_LM",
    )

    train_ds = load_dataset("json", data_files=cfg["data"]["train_file"], split="train")
    eval_ds = load_dataset("json", data_files=cfg["data"]["eval_file"], split="train")

    sft_cfg = SFTConfig(
        max_length=cfg["data"]["max_seq_length"],
        packing=False,
        **cfg["train"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(cfg["train"]["output_dir"])
    tokenizer.save_pretrained(cfg["train"]["output_dir"])
    print(f"saved adapter → {cfg['train']['output_dir']}")


if __name__ == "__main__":
    main()
