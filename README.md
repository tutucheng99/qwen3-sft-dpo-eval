# Chinese LLM Alignment: SFT + DPO with Statistical Evaluation

A 10-day portfolio project: fine-tune **Qwen3-8B-Base** with LoRA SFT (COIG-CQIA) and DPO (UltraFeedback-Chinese), then evaluate base / SFT / DPO with bootstrap CI on GPT-4-judge win rates, JSD on output distributions, and lightweight preference-dimension attribution.

## Project status

| Phase | State |
|---|---|
| 1. Env + tooling | ✅ Done |
| 2. Data prep (COIG-CQIA, UltraFeedback-zh) | ✅ Done |
| 3. SFT training (r=32 LoRA on Qwen3-8B-Base) | ✅ Done |
| 4. DPO training | 🔄 In progress |
| 5. Statistical evaluation (bootstrap CI / JSD / dimension attribution) | ⏳ Pending |
| 6. vLLM deployment + Gradio demo | ⏳ Pending |

## Stack

| Stage | Choice |
|---|---|
| Base model | `Qwen/Qwen3-8B-Base` (Apache 2.0) |
| SFT data | `m-a-p/COIG-CQIA`, 11 hand-picked subsets, score-filtered to 7000 train + 350 eval |
| DPO data | `opencsg/UltraFeedback-chinese` (binarized variant), 7600 train + 400 eval pairs |
| Adapter | LoRA r=32, alpha=64 (~80M trainable params, 1% of 8B) |
| Hardware | RunPod B200 (192 GB HBM3e), single GPU |
| Software | PyTorch 2.8 + CUDA 12.8, transformers 5.7, trl 1.3, peft 0.19 |
| Serving | vLLM + Gradio (TBD) |

## Layout

```
configs/      sft.yaml, dpo.yaml         — all hyperparams
data/         prepare_sft.py, prepare_dpo.py
              processed/                 — generated JSONL (gitignored)
scripts/      sft_train.py, dpo_train.py, merge_lora.py, compare_base_vs_sft.py
notebooks/    diagnostics.ipynb          — interactive position/generation inspection
eval/         (TBD — Day 7-8)
serve/        (TBD — Day 9)
```

## Setup

RunPod template: **PyTorch 2.8.0 (CUDA 12.8)**. Recommended: `Container Disk ≥ 50 GB`, `Volume Disk ≥ 100 GB`. Use a venv on `/workspace` so it survives pod restarts:

```bash
python -m venv /workspace/.venv && source /workspace/.venv/bin/activate
export TMPDIR=/workspace/tmp PIP_CACHE_DIR=/workspace/pip_cache HF_HOME=/workspace/hf_cache
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.lock.txt        # exact frozen versions from B200 pod
```

`vllm` is intentionally excluded from `requirements.txt` — its torch pin can downgrade the training env. Install in a separate venv at deployment time.

## Run

```bash
# SFT
python data/prepare_sft.py --total 8000
python scripts/sft_train.py --config configs/sft.yaml          # ~36 min on B200, r=32

# DPO (uses SFT as policy init)
python data/prepare_dpo.py --total 8000
python scripts/dpo_train.py --config configs/dpo.yaml          # ~45 min on B200

# Merge for serving (after both stages)
python scripts/merge_lora.py --adapter checkpoints/sft --out merged/sft --stack sft
python scripts/merge_lora.py --adapter checkpoints/dpo --sft_adapter checkpoints/sft \
    --out merged/dpo --stack sft+dpo
```

## Implementation notes

- **ChatML EOS handling**: training data emits explicit `<|im_end|>` at the end of each assistant turn; trl auto-appends `<|endoftext|>` after, giving a `content<|im_end|><|endoftext|>` dual-EOS pattern. Inference passes both as `eos_token_id=[151645, 151643]` to `model.generate`. This was non-obvious to get right with trl 1.x — see `notebooks/diagnostics.ipynb` for the position-by-position EOS verification.
- **LoRA r=32**: doubled from typical r=16 because r=16 left the model with near-uniform output at end-of-answer position. r=32 fits better on the EOS distribution.
- **trl 1.x API**: SFTTrainer/DPOTrainer use `processing_class=` not `tokenizer=`; `dataset_text_field` is auto-detected and removed; `DPOConfig` no longer has `max_prompt_length`.

## Evaluation differentiator

Most LLM eval reports show win-rate point estimates without uncertainty. This project reports **bootstrap 95% CIs** on every comparison, **JSD between output distributions** over a fixed prompt set (a methodology reused from a prior bridge-game-policy analysis), and **logprob-shift attribution** on hand-paired contrastive prompts (politeness / verbosity / refusal-tendency dimensions).
