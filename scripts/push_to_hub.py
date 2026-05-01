"""Push SFT and DPO LoRA adapters to HuggingFace Hub.

Reads HF_TOKEN from /workspace/.env (or environment). Creates two repos under
the user's namespace, generates a model card describing each adapter and how
to load it.

Usage:
    HF_TOKEN=hf_... python scripts/push_to_hub.py --user JeffCheng12138
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

REPO_BASE = "Qwen/Qwen3-8B-Base"
GITHUB_REPO = "https://github.com/tutucheng99/qwen3-sft-dpo-eval"


SFT_CARD = """---
library_name: peft
base_model: Qwen/Qwen3-8B-Base
license: apache-2.0
language:
- zh
tags:
- chinese
- instruction-tuning
- lora
- qwen3
datasets:
- m-a-p/COIG-CQIA
---

# Qwen3-8B-Base SFT LoRA — COIG-CQIA

LoRA adapter (`r=32, α=64`) over **`Qwen/Qwen3-8B-Base`**, instruction-tuned on
**`m-a-p/COIG-CQIA`** (11 hand-picked subsets, 6,654 train samples, ChatML format).
Trained on RunPod B200 in 36 minutes.

This is the **SFT-only** adapter. For the full SFT+DPO stack, see
[`{user}/qwen3-8b-dpo-ultrafeedback-zh`](https://huggingface.co/{user}/qwen3-8b-dpo-ultrafeedback-zh).

Project repo: {github}

## Quick load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "{base}", dtype="bfloat16", device_map="cuda", trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "{user}/qwen3-8b-sft-coig-cqia")
tok = AutoTokenizer.from_pretrained("{base}", trust_remote_code=True)
```

## Caveats

- **Statistical evaluation found this SFT regressed below BASE** on a 40-prompt
  pairwise judge (BASE wins 70%, 95% CI [0.60, 0.80]). The COIG-CQIA filter
  steered toward terser responses. See {github}/blob/main/docs/REPORT.md §4.1.
- DPO on top recovers the regression — the DPO adapter is the recommended
  artifact for actual use.
- EOS not strongly learned; deployments should post-process trailing
  low-frequency tokens. See `serve/hf_serve.py:clean_trailing` in the repo.

## Training config

- LoRA: `r=32, α=64, dropout=0.05`, target all attention + MLP linear layers
- Optimizer: AdamW fused, lr `2e-4`, cosine schedule, warmup 0.03
- 2 epochs, effective batch 16 (per-device 4 × accum 4), bf16, sdpa attention
- Hardware: 1× B200 (192 GB HBM3e)
"""


DPO_CARD = """---
library_name: peft
base_model: Qwen/Qwen3-8B-Base
license: apache-2.0
language:
- zh
tags:
- chinese
- preference-learning
- dpo
- lora
- qwen3
datasets:
- opencsg/UltraFeedback-chinese
---

# Qwen3-8B-Base SFT+DPO LoRA — UltraFeedback-zh

LoRA adapter (`r=32, α=64`) trained with **DPO** (β=0.1, sigmoid loss) on top of
the SFT-merged base. Reference policy = SFT-merged base (trl's default with
`peft_config + ref_model=None`).

Preference pairs from **`opencsg/UltraFeedback-chinese`** binarized variant
(7,600 train + 400 eval). Trained on RunPod B200 in 46 minutes.

⚠️ This adapter is calibrated for **the SFT-merged base**, not raw Qwen3-Base.
Apply SFT first, merge, then load this DPO adapter on top — see loading example.

Project repo: {github} · Full eval writeup: {github}/blob/main/docs/REPORT.md

## Quick load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "{base}", dtype="bfloat16", device_map="cuda", trust_remote_code=True,
)
# 1) merge SFT into base
base = PeftModel.from_pretrained(base, "{user}/qwen3-8b-sft-coig-cqia").merge_and_unload()
# 2) apply DPO LoRA on top
model = PeftModel.from_pretrained(base, "{user}/qwen3-8b-dpo-ultrafeedback-zh")
tok = AutoTokenizer.from_pretrained("{base}", trust_remote_code=True)
```

## Eval highlights

Evaluated against BASE and SFT on 40 hand-curated Chinese prompts (DeepSeek-chat
judge with order-bias control + bootstrap 95% CI):

| Pair | DPO win rate | 95% CI |
|---|---|---|
| DPO vs SFT | **0.625** | [0.537, 0.713] (significant) |
| DPO vs BASE | 0.463 | [0.350, 0.562] (statistical tie) |

DPO **recovered the SFT regression** and reached parity with BASE on the judge.
Reward margins climbed from 0.02 to 1.0 nats over training.

## Known regression

Dimension attribution found DPO **weakened refusal robustness under jailbreak
framing** by 0.27 nats — under fictional / research framing the model assigns
substantially less probability to refusal text. Mitigation strategies for
deployment are out of scope here, but flag as a known side-effect.

## Training config

- LoRA: `r=32, α=64, dropout=0.05`, all attention + MLP linear layers
- DPO: β=0.1, loss_type=sigmoid, max_length=2048
- Optimizer: AdamW fused, lr `5e-6`, cosine schedule, warmup 0.1
- 1 epoch, effective batch 16 (per-device 2 × accum 8), bf16, sdpa attention
- Reference model: SFT-merged base (peft + ref_model=None default)
"""


def push_one(api: HfApi, local_dir: Path, repo_id: str, card: str, token: str):
    print(f"\n=== {repo_id} ===")
    create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True, private=False)

    # write README.md (model card)
    card_path = local_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")

    upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        token=token,
        commit_message="Initial upload of LoRA adapter + model card",
    )
    print(f"  → https://huggingface.co/{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="HF username, e.g. JeffCheng12138")
    ap.add_argument("--sft_dir", default="checkpoints/sft")
    ap.add_argument("--dpo_dir", default="checkpoints/dpo")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set in env")

    api = HfApi(token=token)

    sft_repo = f"{args.user}/qwen3-8b-sft-coig-cqia"
    dpo_repo = f"{args.user}/qwen3-8b-dpo-ultrafeedback-zh"

    sft_card = SFT_CARD.format(user=args.user, base=REPO_BASE, github=GITHUB_REPO)
    dpo_card = DPO_CARD.format(user=args.user, base=REPO_BASE, github=GITHUB_REPO)

    push_one(api, Path(args.sft_dir), sft_repo, sft_card, token)
    push_one(api, Path(args.dpo_dir), dpo_repo, dpo_card, token)

    print(f"\nDone. Both adapters live at:")
    print(f"  https://huggingface.co/{sft_repo}")
    print(f"  https://huggingface.co/{dpo_repo}")


if __name__ == "__main__":
    main()
