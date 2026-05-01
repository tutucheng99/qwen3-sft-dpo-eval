"""Aggregate dimension.json into a clean per-axis shift table.

Prints logp difference between variant pairs for each model. Run after
dimension.py. The bug in dimension.py's inline aggregation is harmless —
this script just does it correctly.
"""
import json
from collections import defaultdict

results = json.load(open("eval/results/dimension.json"))

by = defaultdict(list)
for r in results:
    by[(r["axis"], r["variant"], r["model"])].append(r["logp"])
mean_lp = {k: sum(v) / len(v) for k, v in by.items()}

axes_variants = defaultdict(set)
for r in results:
    axes_variants[r["axis"]].add(r["variant"])

print(f"{'axis':<18s}  {'comparison':<35s}  {'BASE':>8s}  {'SFT':>8s}  {'DPO':>8s}")
print("-" * 82)
for axis in sorted(axes_variants):
    vs = sorted(axes_variants[axis])
    if len(vs) != 2:
        continue
    va, vb = vs
    row = [mean_lp[(axis, vb, m)] - mean_lp[(axis, va, m)] for m in ("base", "sft", "dpo")]
    label = f"logp({vb}) - logp({va})"
    print(f"{axis:<18s}  {label:<35s}  {row[0]:>+8.3f}  {row[1]:>+8.3f}  {row[2]:>+8.3f}")

print()
print("Interpretation:")
print("  positive value = model assigns HIGHER probability to reference response under variant_b prompt")
print("  shift across BASE→SFT→DPO shows how alignment moved the conditional preference")
