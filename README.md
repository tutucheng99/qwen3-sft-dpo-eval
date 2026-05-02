# Qwen3-8B SFT + DPO + Statistical Evaluation

**Aligning Qwen3-8B-Base for Chinese conversation, with a layered eval that goes beyond single-number win rate.**
**面向中文对话的 Qwen3-8B 对齐项目,统计评估超越单一胜率指标。**

10-day portfolio project on RunPod B200. Full technical writeup: **[docs/REPORT.md](docs/REPORT.md)**.

**Trained adapters on HuggingFace Hub:**
- [`JeffCheng12138/qwen3-8b-sft-coig-cqia`](https://huggingface.co/JeffCheng12138/qwen3-8b-sft-coig-cqia) — SFT LoRA
- [`JeffCheng12138/qwen3-8b-dpo-ultrafeedback-zh`](https://huggingface.co/JeffCheng12138/qwen3-8b-dpo-ultrafeedback-zh) — DPO LoRA (recommended)

---

## TL;DR — three findings most reports would miss

### 1. SFT actually regressed below BASE; DPO recovered.

![Win rates with 95% CI](docs/figures/fig2_winrates_ci.png)

40 hand-curated Chinese prompts judged pairwise by DeepSeek-chat (with order-bias control) + bootstrap 95% CI:

| Pair | Win rate of A | 95% CI | Verdict |
|---|---|---|---|
| BASE vs SFT | **0.700** | [0.600, 0.800] | BASE wins significantly |
| SFT vs DPO  | **0.375** | [0.287, 0.463] | DPO wins significantly |
| BASE vs DPO | 0.537 | [0.438, 0.650] | Statistical tie |

Per-category breakdown points to **shorter responses** as SFT's regression mechanism: BASE wins 100% on `explain` / `format`, 83% on `creative`. The COIG-CQIA filter steered toward terser answers that the judge consistently rated worse on open-ended categories. **DPO on UltraFeedback-zh restored response quality to BASE's level.**

### 2. DPO weakened refusal robustness under jailbreak framing.

![Dimension attribution](docs/figures/fig3_dimension_shift.png)

Per-axis Δ log-prob of a fixed reference response across paired contrastive prompts:

- **Politeness gap shrinks** through alignment: BASE +0.169 → DPO −0.065. Models become less polite-discriminating.
- **Verbosity matching strengthens**: −0.261 → −0.594. DPO learned to mirror prompt's verbosity expectation.
- **Refusal robustness regresses under DPO**: −0.098 → **−0.271**. Under research/fiction framing the model assigns 0.27 nats less probability to refusal text — a concrete safety side-effect that win-rate eval would not surface.

### 3. DPO moves orthogonally to SFT, not further along the same axis.

![JSD distribution](docs/figures/fig4_jsd_distribution.png)

JSD(BASE, SFT) = 0.220, JSD(SFT, DPO) = 0.193, **JSD(BASE, DPO) = 0.189** (≈ JSD to SFT, not larger). Consistent with DPO's design as a small KL-regularized adjustment around the SFT policy rather than a continuation of it.

---

## Stack

| Stage | Choice |
|---|---|
| Base | `Qwen/Qwen3-8B-Base` (Apache 2.0) |
| SFT data | `m-a-p/COIG-CQIA`, 11 hand-picked subsets, 6,654 train + 350 eval |
| DPO data | `opencsg/UltraFeedback-chinese` binarized, 7,600 train + 400 eval pairs |
| Adapter | LoRA `r=32, α=64`, all attention + MLP linear layers (~80M trainable, 1% of 8B) |
| Hardware | RunPod B200 192 GB HBM3e, single GPU |
| Stack | torch 2.8 / cu128 · transformers 5.7 · trl 1.3 · peft 0.19 |
| Judge | DeepSeek-chat (10× cheaper than GPT-4o, similarly strong on Chinese) |
| Serve | transformers + FastAPI backend → Gradio chat UI |

## Layout

```
configs/      sft.yaml, dpo.yaml         all hyperparams
data/         prepare_sft.py             COIG-CQIA → ChatML JSONL
              prepare_dpo.py             UltraFeedback-zh binarized → DPO triples
scripts/      sft_train.py / dpo_train.py / merge_lora.py / compare_base_vs_sft.py
              diag_eos.py                position-by-position EOS probe
notebooks/    diagnostics.ipynb          interactive SFT/DPO inspection
eval/         prompts.py                 40 Chinese prompts + 6 contrast pairs
              generate.py                BASE/SFT/DPO outputs
              judge.py                   GPT-4o / Claude / DeepSeek pairwise judge
              bootstrap.py               win rate + 95% CI
              jsd.py                     output distribution divergence
              dimension.py / dim_summary.py
              plot_results.py            generates docs/figures/*.png
serve/        hf_serve.py                FastAPI OpenAI-compat backend
              gradio_app.py              chat UI on :7860
docs/         REPORT.md, figures/
```

## Run

```bash
# RunPod template: PyTorch 2.8 (CUDA 12.8), container disk ≥50GB, volume ≥100GB
python -m venv /workspace/.venv && source /workspace/.venv/bin/activate
export TMPDIR=/workspace/tmp PIP_CACHE_DIR=/workspace/pip_cache HF_HOME=/workspace/hf_cache
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.lock.txt

# SFT (~36 min on B200)
python data/prepare_sft.py --total 8000
python scripts/sft_train.py --config configs/sft.yaml

# DPO (~46 min)
python data/prepare_dpo.py --total 8000
python scripts/dpo_train.py --config configs/dpo.yaml

# Eval
python eval/generate.py                                          # ~12 min
DEEPSEEK_API_KEY=... python eval/judge.py --provider deepseek    # ~5 min, ~$0.05
python eval/bootstrap.py
python eval/jsd.py                                               # ~15 min
python eval/dimension.py && python eval/dim_summary.py
python eval/plot_results.py

# Demo
python scripts/merge_lora.py --adapter checkpoints/dpo --sft_adapter checkpoints/sft \
    --out merged/dpo --stack sft+dpo
python serve/hf_serve.py --model merged/dpo --port 8000 &
python serve/gradio_app.py --backend http://localhost:8000/v1 --port 7860
```

Reproduction wall-clock ≈ **2.5 hours of GPU + $0.05 judge API**.

## Implementation notes

**ChatML EOS handling.** Training data emits explicit `<|im_end|>` at end of assistant turn; trl auto-appends `<|endoftext|>` after, giving a `content<|im_end|><|endoftext|>` dual-EOS pattern. Inference passes both as `eos_token_id=[151645, 151643]` to `model.generate`. Despite this, the model's top-1 prediction at end-of-answer never converges (top-1 prob ≈ 0.001) — suspected trl 1.x data-collator behavior masking the final position from loss. Mitigated at inference by truncating trailing low-frequency tokens. See `docs/REPORT.md §5.2` for full discussion.

**LoRA r=32.** Doubled from typical r=16 because a tighter rank left the same EOS regression. r=32 = ~80M trainable parameters, well within the 8B model.

**trl 1.x API.** Both SFTTrainer and DPOTrainer use `processing_class=` not `tokenizer=`; `dataset_text_field` was removed and is auto-detected; `DPOConfig` no longer has `max_prompt_length`. Code is calibrated for trl 1.3 / transformers 5.7.

**RunPod operations.** Container disk default 5 GB is too small — set ≥50 GB at pod creation. Persist the venv on `/workspace` (the network volume) so pod restarts don't trigger 10-min reinstalls. Don't `Stop` pods mid-project: GPU is released and may not be available on resume; `Pause` keeps the lease.

## License

Code is MIT. Model artifacts inherit Qwen3-8B-Base's Apache 2.0.
