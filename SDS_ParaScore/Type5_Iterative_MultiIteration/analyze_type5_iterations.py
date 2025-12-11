"""
TYPE 2 → TYPE 5 ANALYSIS (Multi-Iteration Paraphrasing)
Complete script for iterative paraphrase drift analysis

This script analyzes multi-step LLM paraphrasing:
- LLM-generated (Type 2) → Type 5 (1st) → Type 5 (3rd)
- Hop-by-hop drift measurement
- Cumulative drift tracking

Outputs:
- type5_iteration_analysis.png
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
import json
import os

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Configuration - use script's directory for outputs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = "/Users/zaidshaikh/GitHub/padben_eda_notebook/data.json"
OUTPUT_DIR = SCRIPT_DIR

# ParaScore threshold
PARASCORE_THRESHOLD = 0.35

print("=" * 60)
print("TYPE 2 → TYPE 5 ANALYSIS")
print("Multi-Iteration Paraphrasing with Hop-by-Hop Tracking")
print("=" * 60)

# ===== LOAD DATA =====
print("\n📂 Loading data...")
with open(DATA_FILE, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Map to standard names
df["Type_2"] = df["llm_generated_text(type2)"]
df["Type_5_iter1"] = df["llm_paraphrased_generated_text(type5)-1st"]
df["Type_5_iter3"] = df["llm_paraphrased_generated_text(type5)-3rd"]

print(f"✅ Loaded {len(df)} samples")
print(f"✅ Type 5 (1st): {df['Type_5_iter1'].notna().sum()} samples")
print(f"✅ Type 5 (3rd): {df['Type_5_iter3'].notna().sum()} samples")


# ===== CALCULATE SIMPLE SDS =====
def simple_sds(
    text1_series: pd.Series[Any], text2_series: pd.Series[Any]
) -> pd.Series[Any]:
    """Quick SDS calculation using Jaccard + Edit Distance"""
    from Levenshtein import distance

    scores: list[float] = []
    for t1, t2 in zip(text1_series, text2_series):
        if pd.isna(t1) or pd.isna(t2):
            scores.append(np.nan)
            continue

        # Jaccard divergence
        tokens1 = set(str(t1).lower().split())
        tokens2 = set(str(t2).lower().split())
        jaccard = (
            len(tokens1 & tokens2) / len(tokens1 | tokens2)
            if len(tokens1 | tokens2) > 0
            else 0
        )
        jaccard_div = 1 - jaccard

        # Normalized edit distance
        max_len = max(len(str(t1)), len(str(t2)))
        edit_dist = distance(str(t1), str(t2)) / max_len if max_len > 0 else 0

        # Weighted SDS
        sds = 0.5 * jaccard_div + 0.5 * edit_dist
        scores.append(sds)

    return pd.Series(scores)


# ===== HOP-BY-HOP DRIFT CALCULATION =====
print("\n" + "=" * 60)
print("CALCULATING HOP-BY-HOP DRIFT")
print("=" * 60)

print("\n📊 Step A: Type 2 → Type 5 (1st iteration)...")
df["SDS_Step_A"] = simple_sds(df["Type_2"], df["Type_5_iter1"])

print("📊 Step B: Type 5 (1st) → Type 5 (3rd iteration)...")
df["SDS_Step_B"] = simple_sds(df["Type_5_iter1"], df["Type_5_iter3"])

# Statistics
step_a_mean = df["SDS_Step_A"].mean()
step_b_mean = df["SDS_Step_B"].mean()

print("\n📈 RESULTS:")
print(f"   Step A mean: {step_a_mean:.3f}")
print(f"   Step B mean: {step_b_mean:.3f}")

if step_b_mean > step_a_mean:
    print(
        f"   🔥 DRIFT ACCELERATING! Step B > Step A by {step_b_mean - step_a_mean:.3f}"
    )
else:
    print("   ✅ Drift stable (Step B ≤ Step A)")

# Threshold crossing
step_a_above = (df["SDS_Step_A"] > PARASCORE_THRESHOLD).sum()
step_b_above = (df["SDS_Step_B"] > PARASCORE_THRESHOLD).sum()

print("\n📊 ParaScore Threshold Crossing (>0.35):")
print(
    f"   Step A: {step_a_above}/{len(df)} samples ({step_a_above / len(df) * 100:.1f}%)"
)
print(
    f"   Step B: {step_b_above}/{len(df)} samples ({step_b_above / len(df) * 100:.1f}%)"
)

# ===== CUMULATIVE DRIFT CALCULATION =====
print("\n" + "=" * 60)
print("CALCULATING CUMULATIVE DRIFT")
print("=" * 60)

# Cumulative SDS across iterations
cumulative_sds = {
    "Type_2": 0.0,  # Baseline
    "5_1st": step_a_mean,
    "5_3rd": step_a_mean + step_b_mean,  # Cumulative
}

print("\n📈 Cumulative Drift Progression:")
print(f"   Type 2 (baseline):    {cumulative_sds['Type_2']:.3f}")
print(f"   Type 5 (1st iter):    {cumulative_sds['5_1st']:.3f}")
print(f"   Type 5 (3rd iter):    {cumulative_sds['5_3rd']:.3f}")
print(f"\n🔥 Total drift accumulated: {cumulative_sds['5_3rd']:.3f}")
print(
    f"   That's {cumulative_sds['5_3rd'] / PARASCORE_THRESHOLD:.1f}x ParaScore threshold!"
)

# ===== VISUALIZATION - FIGURE 3 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Hop-by-Hop Comparison
axes[0].hist(
    df["SDS_Step_A"].dropna(),
    bins=30,
    alpha=0.6,
    label="Step A (2→5_1st)",
    color="blue",
    edgecolor="black",
)
axes[0].hist(
    df["SDS_Step_B"].dropna(),
    bins=30,
    alpha=0.6,
    label="Step B (5_1st→5_3rd)",
    color="red",
    edgecolor="black",
)
axes[0].axvline(
    PARASCORE_THRESHOLD,
    color="green",
    linestyle="--",
    linewidth=2,
    label="ParaScore (0.35)",
)
axes[0].set_xlabel("SDS", fontsize=11)
axes[0].set_ylabel("Frequency", fontsize=11)
axes[0].set_title("Hop-by-Hop Drift Comparison", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Plot 2: Cumulative Progression
iterations = ["Type 2", "5 (1st)", "5 (3rd)"]
cumulative_values = [
    cumulative_sds["Type_2"],
    cumulative_sds["5_1st"],
    cumulative_sds["5_3rd"],
]

axes[1].plot(
    iterations,
    cumulative_values,
    marker="o",
    linewidth=3,
    markersize=12,
    color="darkred",
)
axes[1].axhline(
    PARASCORE_THRESHOLD,
    color="green",
    linestyle="--",
    linewidth=2,
    label="ParaScore Threshold",
)
axes[1].set_ylabel("Cumulative Mean SDS", fontsize=11)
axes[1].set_title(
    "Drift Accumulation Across Iterations", fontsize=12, fontweight="bold"
)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

# Add annotations
axes[1].annotate(
    f"{cumulative_sds['5_1st']:.2f}",
    xy=(1, cumulative_sds["5_1st"]),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
)
axes[1].annotate(
    f"{cumulative_sds['5_3rd']:.2f}",
    xy=(2, cumulative_sds["5_3rd"]),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "type5_iteration_analysis.png"),
    dpi=300,
    bbox_inches="tight",
)
print("\n✅ Figure 3 saved: type5_iteration_analysis.png")
plt.close()

# ===== SAVE RESULTS =====
results = {
    "step_a_mean": float(step_a_mean),
    "step_b_mean": float(step_b_mean),
    "cumulative_type2": float(cumulative_sds["Type_2"]),
    "cumulative_5_1st": float(cumulative_sds["5_1st"]),
    "cumulative_5_3rd": float(cumulative_sds["5_3rd"]),
    "vs_parascore_ratio": float(cumulative_sds["5_3rd"] / PARASCORE_THRESHOLD),
    "step_a_above_threshold_pct": float(step_a_above / len(df) * 100),
    "step_b_above_threshold_pct": float(step_b_above / len(df) * 100),
}

with open(os.path.join(OUTPUT_DIR, "iteration_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n💾 Results saved: iteration_results.json")

# ===== FINAL SUMMARY =====
print("\n" + "=" * 60)
print("📊 FINAL SUMMARY")
print("=" * 60)

print("\n✅ Hop-by-Hop Analysis:")
print(f"   - Step A (2→5_1st): {step_a_mean:.3f}")
print(f"   - Step B (5_1st→5_3rd): {step_b_mean:.3f}")

print("\n✅ Cumulative Drift:")
print(f"   - By 1st iteration: {cumulative_sds['5_1st']:.3f}")
print(f"   - By 3rd iteration: {cumulative_sds['5_3rd']:.3f}")
print(f"   - That's {results['vs_parascore_ratio']:.1f}x ParaScore threshold!")

print("\n✅ Threshold Crossing:")
print(f"   - Step A: {results['step_a_above_threshold_pct']:.1f}% above 0.35")
print(f"   - Step B: {results['step_b_above_threshold_pct']:.1f}% above 0.35")

print("\n🎉 Type 2 → Type 5 analysis complete!")
print("=" * 60)
