"""
Analysis of false memory experiment results.
Computes statistics, generates visualizations, and tests hypotheses.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})


def load_results(filename):
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


def analyze_experiment1():
    """Analyze Experiment 1: Emoji existence probing."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Experiment 1 - Emoji Existence Probing")
    print("=" * 70)

    data = load_results("experiment1_existence.json")
    df = pd.DataFrame(data)

    # Overall accuracy by model
    print("\n--- Overall Accuracy ---")
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        acc = mdf["correct"].mean()
        print(f"  {model}: {acc:.3f} ({mdf['correct'].sum()}/{len(mdf)})")

    # FALSE POSITIVE RATES by plausibility and model
    print("\n--- False Positive Rates (claiming nonexistent items exist) ---")
    fp_results = {}
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        fp_results[model] = {}
        for plaus in ["real", "plausible", "implausible"]:
            subset = mdf[mdf["plausibility"] == plaus]
            if plaus == "real":
                # True positive rate
                tpr = subset["assertion"].mean()
                fp_results[model]["TPR"] = tpr
                print(f"  {model} | Real items TPR: {tpr:.3f} ({subset['assertion'].sum()}/{len(subset)})")
            else:
                # False positive rate
                fpr = subset["assertion"].mean()
                fp_results[model][plaus] = fpr
                print(f"  {model} | {plaus.capitalize()} nonexistent FPR: {fpr:.3f} ({subset['assertion'].sum()}/{len(subset)})")

    # H1 Test: FPR(plausible) > FPR(implausible)?
    print("\n--- H1: Plausible > Implausible FPR ---")
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        plaus_assertions = mdf[mdf["plausibility"] == "plausible"]["assertion"].values
        implaus_assertions = mdf[mdf["plausibility"] == "implausible"]["assertion"].values

        # Fisher's exact test (2x2 contingency table)
        plaus_fp = plaus_assertions.sum()
        plaus_tn = len(plaus_assertions) - plaus_fp
        implaus_fp = implaus_assertions.sum()
        implaus_tn = len(implaus_assertions) - implaus_fp

        table = [[plaus_fp, plaus_tn], [implaus_fp, implaus_tn]]
        odds_ratio, p_value = stats.fisher_exact(table, alternative='greater')
        print(f"  {model}: FPR_plaus={plaus_fp}/{len(plaus_assertions)}, FPR_implaus={implaus_fp}/{len(implaus_assertions)}")
        print(f"    Fisher's exact test (one-sided): OR={odds_ratio:.2f}, p={p_value:.4f}")
        print(f"    H1 {'SUPPORTED' if p_value < 0.05 else 'NOT SUPPORTED'} (α=0.05)")

    # Per-category FPR analysis
    print("\n--- Per-Category False Positive Rates ---")
    cat_fpr = {}
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        print(f"\n  Model: {model}")
        for cat in sorted(mdf["category"].unique()):
            cdf = mdf[(mdf["category"] == cat) & (mdf["plausibility"] != "real")]
            fpr = cdf["assertion"].mean()
            n_false = cdf["assertion"].sum()
            cat_fpr.setdefault(model, {})[cat] = fpr
            real_count = len(mdf[(mdf["category"] == cat) & (mdf["plausibility"] == "real")])
            print(f"    {cat}: FPR={fpr:.3f} ({n_false}/{len(cdf)}), real_count={real_count}")

    # Visualization: FPR by plausibility and model
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model in enumerate(df["model"].unique()):
        mdf = df[df["model"] == model]
        plaus_groups = mdf.groupby("plausibility")["assertion"].agg(["mean", "sem"]).reset_index()
        plaus_groups = plaus_groups.set_index("plausibility").loc[["real", "plausible", "implausible"]]

        colors = ["#2ecc71", "#e74c3c", "#3498db"]
        bars = axes[idx].bar(
            range(3), plaus_groups["mean"],
            yerr=plaus_groups["sem"] * 1.96,
            color=colors, edgecolor="black", alpha=0.8, capsize=5,
        )
        axes[idx].set_xticks(range(3))
        axes[idx].set_xticklabels(["Real\n(True Positive)", "Plausible\nNonexistent", "Implausible\nNonexistent"])
        axes[idx].set_ylabel("Assertion Rate (claims item exists)")
        axes[idx].set_title(f"{model}")
        axes[idx].set_ylim(0, 1.1)
        axes[idx].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

        # Add value labels
        for bar, val in zip(bars, plaus_groups["mean"]):
            axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                          f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    fig.suptitle("Experiment 1: Existence Assertion Rates by Item Plausibility", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_fpr_by_plausibility.png")
    plt.close()
    print(f"\n  Saved: {PLOTS_DIR / 'exp1_fpr_by_plausibility.png'}")

    # Visualization: Per-category FPR heatmap
    fig, ax = plt.subplots(figsize=(12, 5))
    cat_data = []
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        for cat in sorted(mdf["category"].unique()):
            for plaus in ["plausible", "implausible"]:
                subset = mdf[(mdf["category"] == cat) & (mdf["plausibility"] == plaus)]
                if len(subset) > 0:
                    cat_data.append({
                        "model": model,
                        "category": cat.replace("_emojis", "").replace("_", " ").title(),
                        "plausibility": plaus,
                        "FPR": subset["assertion"].mean(),
                    })

    cdf = pd.DataFrame(cat_data)
    pivot = cdf.pivot_table(index=["model", "plausibility"], columns="category", values="FPR")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1, ax=ax)
    ax.set_title("False Positive Rates by Category, Model, and Plausibility")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_category_heatmap.png")
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'exp1_category_heatmap.png'}")

    # Detailed item-level analysis for most interesting false positives
    print("\n--- Most Interesting False Positives ---")
    false_positives = df[(df["ground_truth"] == False) & (df["assertion"] == True)]
    for _, row in false_positives.iterrows():
        print(f"  [{row['model']}] {row['category']}/{row['item']}: \"{row['raw_response'][:100]}...\"")

    return df, fp_results


def analyze_experiment2(exp1_df):
    """Analyze Experiment 2: Category density effect using Exp 1 data."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Experiment 2 - Category Density Effect")
    print("=" * 70)

    df = exp1_df[exp1_df["plausibility"] != "real"].copy()

    # Category size (number of real members) vs FPR
    from probe_dataset import CATEGORIES
    category_sizes = {}
    for cat_name, cat_data in CATEGORIES.items():
        if "emoji" in cat_name:
            category_sizes[cat_name] = len(cat_data["real"])

    density_data = []
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        for cat in mdf["category"].unique():
            if cat in category_sizes:
                cdf = mdf[mdf["category"] == cat]
                fpr = cdf["assertion"].mean()
                density_data.append({
                    "model": model,
                    "category": cat,
                    "category_size": category_sizes[cat],
                    "FPR": fpr,
                    "n_probes": len(cdf),
                })

    ddf = pd.DataFrame(density_data)

    print("\n--- Category Size vs FPR ---")
    for model in ddf["model"].unique():
        mdf = ddf[ddf["model"] == model]
        print(f"\n  Model: {model}")
        for _, row in mdf.iterrows():
            print(f"    {row['category']}: size={row['category_size']}, FPR={row['FPR']:.3f}")

        # Spearman correlation
        if len(mdf) >= 3:
            rho, p = stats.spearmanr(mdf["category_size"], mdf["FPR"])
            print(f"    Spearman rho={rho:.3f}, p={p:.4f}")
            print(f"    H2 {'SUPPORTED' if p < 0.05 and rho > 0 else 'NOT SUPPORTED'} (positive correlation)")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in ddf["model"].unique():
        mdf = ddf[ddf["model"] == model]
        ax.scatter(mdf["category_size"], mdf["FPR"], s=100, label=model, alpha=0.8)
        for _, row in mdf.iterrows():
            ax.annotate(row["category"].replace("_emojis", ""),
                       (row["category_size"], row["FPR"]),
                       textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Number of Real Members in Category")
    ax.set_ylabel("False Positive Rate (nonexistent items)")
    ax.set_title("Experiment 2: Category Density vs False Positive Rate")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp2_density_vs_fpr.png")
    plt.close()
    print(f"\n  Saved: {PLOTS_DIR / 'exp2_density_vs_fpr.png'}")

    return ddf


def analyze_experiment3():
    """Analyze Experiment 3: Prompt framing comparison."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Experiment 3 - Prompt Framing")
    print("=" * 70)

    data = load_results("experiment3_framing.json")
    df = pd.DataFrame(data)

    # FPR by framing for nonexistent items
    nonexist = df[df["ground_truth"] == False]
    exist = df[df["ground_truth"] == True]

    print("\n--- True Positive Rate by Framing ---")
    for framing in exist["framing"].unique():
        subset = exist[exist["framing"] == framing]
        tpr = subset["assertion"].mean()
        print(f"  {framing}: TPR={tpr:.3f} ({subset['assertion'].sum()}/{len(subset)})")

    print("\n--- False Positive Rate by Framing ---")
    framing_fpr = {}
    for framing in nonexist["framing"].unique():
        subset = nonexist[nonexist["framing"] == framing]
        fpr = subset["assertion"].mean()
        framing_fpr[framing] = fpr
        n_fp = subset["assertion"].sum()
        print(f"  {framing}: FPR={fpr:.3f} ({n_fp}/{len(subset)})")

    # Chi-square test across framings
    contingency = []
    for framing in sorted(nonexist["framing"].unique()):
        subset = nonexist[nonexist["framing"] == framing]
        fp = subset["assertion"].sum()
        tn = len(subset) - fp
        contingency.append([fp, tn])

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n  Chi-square test across framings: χ²={chi2:.2f}, df={dof}, p={p:.4f}")
    print(f"  H3 {'SUPPORTED' if p < 0.05 else 'NOT SUPPORTED'} (framing affects FPR)")

    # Confidence analysis
    print("\n--- Self-Reported Confidence (confidence framing only) ---")
    conf_data = df[df["framing"] == "confidence"]
    for gt in [True, False]:
        subset = conf_data[conf_data["ground_truth"] == gt]
        label = "Real" if gt else "Nonexistent"
        mean_conf = subset["confidence"].mean()
        print(f"  {label} items: mean confidence={mean_conf:.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # FPR by framing
    framings = sorted(framing_fpr.keys())
    fprs = [framing_fpr[f] for f in framings]
    colors = plt.cm.Set2(np.linspace(0, 1, len(framings)))
    bars = axes[0].bar(range(len(framings)), fprs, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_xticks(range(len(framings)))
    axes[0].set_xticklabels(framings, rotation=15)
    axes[0].set_ylabel("False Positive Rate")
    axes[0].set_title("FPR by Prompt Framing (Nonexistent Items)")
    axes[0].set_ylim(0, 1.1)
    for bar, val in zip(bars, fprs):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Per-item comparison across framings
    item_data = nonexist.pivot_table(index="item", columns="framing", values="assertion")
    sns.heatmap(item_data, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=axes[1],
                vmin=0, vmax=1, cbar_kws={"label": "Claims Exists (1=Yes)"})
    axes[1].set_title("Per-Item Assertions Across Framings")

    fig.suptitle("Experiment 3: Effect of Prompt Framing on False Memories", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp3_framing_comparison.png")
    plt.close()
    print(f"\n  Saved: {PLOTS_DIR / 'exp3_framing_comparison.png'}")

    # Show detailed adversarial responses for interesting cases
    print("\n--- Adversarial Framing: Detailed False Positives ---")
    adv_fp = df[(df["framing"] == "adversarial") & (df["ground_truth"] == False) & (df["assertion"] == True)]
    for _, row in adv_fp.iterrows():
        print(f"  {row['item']}: \"{row['raw_response'][:120]}...\"")

    return df


def analyze_experiment4():
    """Analyze Experiment 4: Cross-domain generalization."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Experiment 4 - Cross-Domain Generalization")
    print("=" * 70)

    data = load_results("experiment4_cross_domain.json")
    df = pd.DataFrame(data)

    for cat in df["category"].unique():
        cdf = df[df["category"] == cat]
        print(f"\n--- {cat} ---")
        for plaus in cdf["plausibility"].unique():
            subset = cdf[cdf["plausibility"] == plaus]
            if plaus == "real":
                rate = subset["assertion"].mean()
                print(f"  TPR ({plaus}): {rate:.3f} ({subset['assertion'].sum()}/{len(subset)})")
            else:
                rate = subset["assertion"].mean()
                print(f"  FPR ({plaus}): {rate:.3f} ({subset['assertion'].sum()}/{len(subset)})")

        # Show false positives
        fps = cdf[(cdf["ground_truth"] == False) & (cdf["assertion"] == True)]
        if len(fps) > 0:
            print(f"  False positives:")
            for _, row in fps.iterrows():
                print(f"    {row['item']}: \"{row['raw_response'][:100]}...\"")

    return df


def analyze_experiment5():
    """Analyze Experiment 5: Category listing."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Experiment 5 - Category Listing")
    print("=" * 70)

    data = load_results("experiment5_listing.json")

    for entry in data:
        cat = entry["category_description"]
        print(f"\n--- {cat} emojis ---")
        print(f"  Real emojis found: {entry['real_found']}/{entry['real_count']}")
        print(f"  False inclusions: {entry['false_inclusion_count']}/{entry['plausible_total']}")
        if entry["false_inclusions"]:
            print(f"  False items: {entry['false_inclusions']}")

    return data


def create_summary_visualization(exp1_df, exp3_df):
    """Create a comprehensive summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Overall FPR by plausibility (Exp 1)
    ax = axes[0, 0]
    for idx, model in enumerate(exp1_df["model"].unique()):
        mdf = exp1_df[exp1_df["model"] == model]
        plaus_rates = {}
        for plaus in ["plausible", "implausible"]:
            subset = mdf[mdf["plausibility"] == plaus]
            plaus_rates[plaus] = subset["assertion"].mean()

        x = np.array([0, 1]) + idx * 0.35
        bars = ax.bar(x, [plaus_rates["plausible"], plaus_rates["implausible"]],
                      width=0.3, label=model, alpha=0.8, edgecolor="black")
        for bar, val in zip(bars, [plaus_rates["plausible"], plaus_rates["implausible"]]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks([0.175, 1.175])
    ax.set_xticklabels(["Plausible\nNonexistent", "Implausible\nNonexistent"])
    ax.set_ylabel("False Positive Rate")
    ax.set_title("A. FPR: Plausible vs Implausible (H1)")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Panel B: Per-category FPR (Exp 1, gpt-4.1 only)
    ax = axes[0, 1]
    mdf = exp1_df[(exp1_df["model"] == "gpt-4.1") & (exp1_df["plausibility"] != "real")]
    cat_fpr = mdf.groupby("category")["assertion"].mean().sort_values(ascending=False)
    colors_b = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.2 else "#2ecc71" for v in cat_fpr.values]
    bars = ax.barh(range(len(cat_fpr)), cat_fpr.values, color=colors_b, edgecolor="black", alpha=0.8)
    ax.set_yticks(range(len(cat_fpr)))
    ax.set_yticklabels([c.replace("_emojis", "").replace("_", " ") for c in cat_fpr.index])
    ax.set_xlabel("False Positive Rate")
    ax.set_title("B. FPR by Category (gpt-4.1)")
    for bar, val in zip(bars, cat_fpr.values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2.,
               f'{val:.2f}', ha='left', va='center', fontweight='bold')

    # Panel C: Prompt framing effect (Exp 3)
    ax = axes[1, 0]
    nonexist = exp3_df[exp3_df["ground_truth"] == False]
    framing_fpr = nonexist.groupby("framing")["assertion"].mean().sort_values(ascending=False)
    colors_c = plt.cm.Set2(np.linspace(0, 0.8, len(framing_fpr)))
    bars = ax.bar(range(len(framing_fpr)), framing_fpr.values, color=colors_c, edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(framing_fpr)))
    ax.set_xticklabels(framing_fpr.index, rotation=15)
    ax.set_ylabel("False Positive Rate")
    ax.set_title("C. FPR by Prompt Framing (H3)")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, framing_fpr.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Panel D: Item-level heatmap for key false positives
    ax = axes[1, 1]
    # Show per-item FPR across models for plausible nonexistent in marine category
    marine_plaus = exp1_df[(exp1_df["category"] == "marine_animal_emojis") & (exp1_df["plausibility"] == "plausible")]
    if len(marine_plaus) > 0:
        pivot = marine_plaus.pivot_table(index="item", columns="model", values="assertion")
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax,
                    vmin=0, vmax=1, cbar_kws={"label": "Claims Exists"})
        ax.set_title("D. Marine Animal Emoji False Memories")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("D. Marine Animal Emoji False Memories")

    fig.suptitle("Understanding False Memories in LLMs: Key Results", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_figure.png")
    plt.close()
    print(f"\n  Saved: {PLOTS_DIR / 'summary_figure.png'}")


def compute_signal_detection(exp1_df):
    """Compute d-prime (signal detection sensitivity) for each model."""
    print("\n" + "=" * 70)
    print("SIGNAL DETECTION ANALYSIS")
    print("=" * 70)

    for model in exp1_df["model"].unique():
        mdf = exp1_df[exp1_df["model"] == model]

        # Hit rate (correctly identifying real items)
        real = mdf[mdf["plausibility"] == "real"]
        hit_rate = real["assertion"].mean()

        # False alarm rate (claiming nonexistent items exist)
        nonexist = mdf[mdf["plausibility"] != "real"]
        fa_rate = nonexist["assertion"].mean()

        # Adjust for edge cases (0 or 1)
        n_real = len(real)
        n_nonexist = len(nonexist)
        hit_adj = np.clip(hit_rate, 0.5/n_real, 1 - 0.5/n_real)
        fa_adj = np.clip(fa_rate, 0.5/n_nonexist, 1 - 0.5/n_nonexist)

        # d-prime
        d_prime = stats.norm.ppf(hit_adj) - stats.norm.ppf(fa_adj)

        # Criterion (bias)
        criterion = -0.5 * (stats.norm.ppf(hit_adj) + stats.norm.ppf(fa_adj))

        print(f"\n  {model}:")
        print(f"    Hit rate: {hit_rate:.3f}")
        print(f"    False alarm rate: {fa_rate:.3f}")
        print(f"    d': {d_prime:.3f}")
        print(f"    Criterion (c): {criterion:.3f}")
        print(f"    Interpretation: {'Liberal bias (tends to say Yes)' if criterion < 0 else 'Conservative bias'}")


def run_full_analysis():
    """Run all analyses."""
    print("=" * 70)
    print("FULL ANALYSIS OF FALSE MEMORY EXPERIMENTS")
    print("=" * 70)

    exp1_df, fp_results = analyze_experiment1()
    analyze_experiment2(exp1_df)
    exp3_df = analyze_experiment3()
    analyze_experiment4()
    analyze_experiment5()
    compute_signal_detection(exp1_df)
    create_summary_visualization(exp1_df, exp3_df)

    # Save analysis summary
    summary = {
        "exp1_overall_accuracy": {
            model: float(exp1_df[exp1_df["model"] == model]["correct"].mean())
            for model in exp1_df["model"].unique()
        },
        "exp1_fpr_plausible": {
            model: float(exp1_df[(exp1_df["model"] == model) & (exp1_df["plausibility"] == "plausible")]["assertion"].mean())
            for model in exp1_df["model"].unique()
        },
        "exp1_fpr_implausible": {
            model: float(exp1_df[(exp1_df["model"] == model) & (exp1_df["plausibility"] == "implausible")]["assertion"].mean())
            for model in exp1_df["model"].unique()
        },
    }
    with open(RESULTS_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\nANALYSIS COMPLETE")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    run_full_analysis()
