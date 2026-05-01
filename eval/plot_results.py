"""Generate publication-quality figures from eval/results/ + training logs.

Output: docs/figures/*.png

Figures:
  fig1_dpo_training.png     — DPO loss + reward margins over steps
  fig2_winrates_ci.png      — pairwise win rates with 95% bootstrap CI
  fig3_dimension_shift.png  — per-axis logp shift across BASE/SFT/DPO
  fig4_jsd_distribution.png — JSD per prompt + summary statistics
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})


def parse_dpo_log(log_path: Path) -> dict:
    """Pull (step, loss, margins, accuracies) tuples from training log."""
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    rows = []
    for m in re.finditer(
        r"\{'loss': '([^']+)', 'grad_norm': '[^']+', 'learning_rate': '[^']+', "
        r"[^}]*?'rewards/accuracies': '([^']+)', 'rewards/margins': '([^']+)'[^}]*?'epoch': '([^']+)'\}",
        text,
    ):
        rows.append({
            "loss": float(m.group(1)),
            "accuracy": float(m.group(2)),
            "margins": float(m.group(3)),
            "epoch": float(m.group(4)),
        })
    return rows


def fig_dpo_training(rows, out: Path):
    if not rows:
        print("  (skip fig1: no training log rows)")
        return
    epochs = [r["epoch"] for r in rows]
    losses = [r["loss"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    margins = [r["margins"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(epochs, losses, lw=1.5, color="#444", label="DPO loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DPO training loss")
    ax.set_ylim(0, max(losses) * 1.1)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, margins, lw=1.5, color="#1f77b4", label="rewards/margins")
    ax.plot(epochs, accs, lw=1.5, color="#2ca02c", label="rewards/accuracies")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5, lw=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("DPO reward margins & accuracy")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def fig_winrates_ci(winrates_path: Path, out: Path):
    data = json.loads(winrates_path.read_text())["overall"]
    pairs = [
        ("base_vs_sft", "BASE > SFT", "SFT"),
        ("sft_vs_dpo", "SFT > DPO", "DPO"),
        ("base_vs_dpo", "BASE > DPO", "DPO"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(pairs))
    means = [data[k]["winrate_A"] for k, _, _ in pairs]
    lows = [data[k]["ci_low"] for k, _, _ in pairs]
    highs = [data[k]["ci_high"] for k, _, _ in pairs]
    err_low = [m - l for m, l in zip(means, lows)]
    err_high = [h - m for m, h in zip(means, highs)]

    colors = ["#d62728" if l > 0.5 or h < 0.5 else "#7f7f7f"
              for l, h in zip(lows, highs)]
    ax.barh(y_pos, means, xerr=[err_low, err_high], color=colors,
            edgecolor="black", lw=0.5, capsize=4)
    ax.axvline(0.5, ls="--", color="black", alpha=0.6, lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[1] for p in pairs])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Win rate of model A (95% bootstrap CI, n=40 each)")
    ax.set_title("Pairwise win rates judged by DeepSeek-chat")
    for i, (m, l, h) in enumerate(zip(means, lows, highs)):
        ax.text(m + 0.02, i, f"{m:.3f} [{l:.3f}, {h:.3f}]",
                va="center", fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def fig_dimension_shift(dim_path: Path, out: Path):
    results = json.loads(dim_path.read_text())
    from collections import defaultdict
    by = defaultdict(list)
    for r in results:
        by[(r["axis"], r["variant"], r["model"])].append(r["logp"])
    mean_lp = {k: float(np.mean(v)) for k, v in by.items()}

    axes_variants = defaultdict(set)
    for r in results:
        axes_variants[r["axis"]].add(r["variant"])

    axes_list = sorted(axes_variants.keys())
    diffs = {}
    labels = {}
    for axis in axes_list:
        vs = sorted(axes_variants[axis])
        if len(vs) != 2:
            continue
        va, vb = vs
        diffs[axis] = [mean_lp[(axis, vb, m)] - mean_lp[(axis, va, m)] for m in ("base", "sft", "dpo")]
        labels[axis] = f"logp({vb})\n−logp({va})"

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(axes_list))
    w = 0.27
    colors = {"base": "#7f7f7f", "sft": "#1f77b4", "dpo": "#d62728"}
    for i, m in enumerate(("base", "sft", "dpo")):
        vals = [diffs[a][i] for a in axes_list]
        ax.bar(x + (i - 1) * w, vals, w, label=m.upper(), color=colors[m],
               edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[a] for a in axes_list])
    ax.set_ylabel("Δ avg log-probability of reference response")
    ax.set_title("Dimension attribution: how SFT/DPO shift conditional preference")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def fig_jsd(jsd_path: Path, out: Path):
    data = json.loads(jsd_path.read_text())
    pairs = list(data["overall"].keys())
    overall_means = [data["overall"][p]["mean"] for p in pairs]
    overall_stds = [data["overall"][p]["std"] for p in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    pretty = [p.replace("_", " ↔ ").upper() for p in pairs]
    ax.bar(pretty, overall_means, yerr=overall_stds, color="#1f77b4",
           edgecolor="black", lw=0.5, capsize=5)
    ax.set_ylabel("JSD (mean ± std across prompts)")
    ax.set_title("Output distribution divergence (pairwise)")
    for i, (m, s) in enumerate(zip(overall_means, overall_stds)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for p, c in zip(pairs, ("#7f7f7f", "#1f77b4", "#d62728")):
        per = data["per_prompt"][p]
        vals = sorted(per.values())
        ax.plot(np.arange(len(vals)) / len(vals), vals,
                label=p.replace("_", " ↔ ").upper(), lw=1.5, color=c)
    ax.set_xlabel("Prompt percentile")
    ax.set_ylabel("JSD")
    ax.set_title("JSD distribution across 40 prompts")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def main():
    print("Generating figures...")
    rows = parse_dpo_log(Path("/workspace/dpo_training.log"))
    fig_dpo_training(rows, OUT / "fig1_dpo_training.png")
    fig_winrates_ci(RESULTS / "winrates.json", OUT / "fig2_winrates_ci.png")
    fig_dimension_shift(RESULTS / "dimension.json", OUT / "fig3_dimension_shift.png")
    fig_jsd(RESULTS / "jsd.json", OUT / "fig4_jsd_distribution.png")
    print(f"\n4 figures in {OUT}")


if __name__ == "__main__":
    main()
