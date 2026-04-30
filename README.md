# Chinese LLM Alignment: SFT + DPO with Statistical Evaluation

A 10-day portfolio project: fine-tune **Qwen3-8B-Base** with LoRA SFT (COIG-CQIA) and DPO (UltraFeedback-Chinese), then evaluate base / SFT / DPO with bootstrap CI on GPT-4-judge win rates, JSD on output distributions, and lightweight preference-dimension attribution.

## Stack

| Stage | Choice |
|---|---|
| Base model | `Qwen/Qwen3-8B-Base` (Apache 2.0) |
| SFT data | `m-a-p/COIG-CQIA`, score-filtered ~8k |
| DPO data | `opencsg/UltraFeedback-chinese`, ~8k pairs |
| Hardware | RunPod H100 single-card |
| Adapter | LoRA r=16, alpha=32 |
| Serving | vLLM + Gradio |

## Layout

```
configs/      sft.yaml, dpo.yaml — single source of truth for hyperparams
data/         prepare_sft.py, prepare_dpo.py — HF → JSONL with ChatML
scripts/      sft_train.py, dpo_train.py, merge_lora.py
eval/         (Day 7-8: MT-Bench, bootstrap CI, JSD, dimension attribution)
serve/        vLLM + Gradio (Day 9)
```

## Setup

RunPod template: **PyTorch 2.8.0 (CUDA 12.8)**. Don't reinstall torch — the template's build is CUDA-matched.

```bash
pip install -r requirements.txt                        # vllm intentionally excluded
pip install flash-attn --no-build-isolation
pip freeze > requirements.lock.txt                     # commit this
```

vLLM is installed only at Day 9 (deployment), often in a separate venv to avoid its strict torch pin downgrading the training env:

```bash
python -m venv .venv-serve && source .venv-serve/bin/activate
pip install vllm gradio fastapi uvicorn
pip freeze > requirements.serve.lock.txt
```

For reproduction: `pip install -r requirements.lock.txt`.

## Run

```bash
python data/prepare_sft.py --total 8000
python data/prepare_dpo.py --total 8000

python scripts/sft_train.py --config configs/sft.yaml
python scripts/dpo_train.py --config configs/dpo.yaml

python scripts/merge_lora.py --adapter checkpoints/sft --out merged/sft --stack sft
python scripts/merge_lora.py --adapter checkpoints/dpo --sft_adapter checkpoints/sft \
    --out merged/dpo --stack sft+dpo
```

## Evaluation differentiator

Most LLM eval reports show win-rate point estimates without uncertainty. This project reports **bootstrap 95% CIs** on every comparison, plus **JSD between output distributions** over a fixed prompt set, plus **logprob-shift attribution** on hand-paired contrastive prompts (politeness / verbosity / refusal-tendency dimensions).
