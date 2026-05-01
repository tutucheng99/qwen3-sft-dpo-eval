"""Bootstrap 95% confidence intervals on pairwise win rates.

Reads eval/results/judge.jsonl (output of judge.py) and computes:
  - point win rate of A over B (treating TIE as 0.5)
  - 95% bootstrap CI from 10,000 resamples
  - per-category breakdown

Output: eval/results/winrates.json + a printed table.

This is the project's main differentiator: most reports show point estimates,
we ship 95% CIs so reviewers can tell whether differences are significant.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def winrate(verdicts: list[str]) -> float:
    """A wins = 1, TIE = 0.5, B wins = 0. Mean."""
    if not verdicts:
        return float("nan")
    score = {"A": 1.0, "TIE": 0.5, "B": 0.0}
    return float(np.mean([score[v] for v in verdicts]))


def bootstrap_ci(verdicts: list[str], n_resamples: int = 10000, seed: int = 0):
    if not verdicts:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    score_map = {"A": 1.0, "TIE": 0.5, "B": 0.0}
    scores = np.array([score_map[v] for v in verdicts])
    n = len(scores)
    boot_means = scores[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    point = float(scores.mean())
    lo, hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="eval/results/judge.jsonl")
    ap.add_argument("--out", default="eval/results/winrates.json")
    ap.add_argument("--n_resamples", type=int, default=10000)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.judge, encoding="utf-8")]

    by_pair = defaultdict(list)
    by_pair_cat = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["a_name"], r["b_name"])
        by_pair[key].append(r["final"])
        by_pair_cat[key][r["category"]].append(r["final"])

    out: dict = {"overall": {}, "by_category": {}}

    print(f"\n{'pair':<12s}  {'n':>4s}  {'win_A':>6s}  {'95% CI':>17s}")
    print("-" * 50)
    for (a, b), verdicts in sorted(by_pair.items()):
        point, lo, hi = bootstrap_ci(verdicts, args.n_resamples)
        ci_str = f"[{lo:.3f}, {hi:.3f}]"
        print(f"{a:>4s} vs {b:<4s}  {len(verdicts):>4d}  {point:>6.3f}  {ci_str:>17s}")
        out["overall"][f"{a}_vs_{b}"] = {
            "n": len(verdicts), "winrate_A": point, "ci_low": lo, "ci_high": hi,
            "n_A_wins": sum(1 for v in verdicts if v == "A"),
            "n_B_wins": sum(1 for v in verdicts if v == "B"),
            "n_ties": sum(1 for v in verdicts if v == "TIE"),
        }

    print(f"\n{'pair':<12s}  {'category':<10s}  {'n':>4s}  {'win_A':>6s}  {'95% CI':>17s}")
    print("-" * 60)
    for (a, b), cat_dict in sorted(by_pair_cat.items()):
        out["by_category"][f"{a}_vs_{b}"] = {}
        for cat, verdicts in sorted(cat_dict.items()):
            point, lo, hi = bootstrap_ci(verdicts, args.n_resamples)
            ci_str = f"[{lo:.3f}, {hi:.3f}]"
            print(f"{a:>4s} vs {b:<4s}  {cat:<10s}  {len(verdicts):>4d}  {point:>6.3f}  {ci_str:>17s}")
            out["by_category"][f"{a}_vs_{b}"][cat] = {
                "n": len(verdicts), "winrate_A": point, "ci_low": lo, "ci_high": hi,
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
