"""DPO with LoRA on top of SFT adapter. Reference policy = SFT-frozen."""
import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    tok.eos_token = "<|im_end|>"
    tok.pad_token = "<|endoftext|>"
    tok.padding_side = "right"
    return tok


def align_model_special_tokens(model, tok):
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.pad_token_id = tok.pad_token_id


def load_policy_with_sft(base_id: str, sft_adapter: str, dtype, attn):
    base = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=dtype, attn_implementation=attn, trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, sft_adapter).merge_and_unload()
    merged.config.use_cache = False
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    Path(cfg["train"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, cfg["model"]["dtype"])
    attn = cfg["model"].get("attn_implementation", "sdpa")

    tokenizer = build_tokenizer(cfg["model"]["name_or_path"])

    policy = load_policy_with_sft(
        cfg["model"]["name_or_path"], cfg["model"]["sft_adapter_path"], dtype, attn,
    )
    align_model_special_tokens(policy, tokenizer)

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

    train_kwargs = dict(cfg["train"])
    beta = train_kwargs.pop("beta")
    loss_type = train_kwargs.pop("loss_type")

    dpo_cfg = DPOConfig(
        beta=beta,
        loss_type=loss_type,
        max_prompt_length=cfg["data"]["max_prompt_length"],
        max_length=cfg["data"]["max_length"],
        **train_kwargs,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(cfg["train"]["output_dir"])
    tokenizer.save_pretrained(cfg["train"]["output_dir"])
    print(f"saved DPO adapter → {cfg['train']['output_dir']}")


if __name__ == "__main__":
    main()
