import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_auc_score, roc_curve

from src.dataloader import ChromosomeDataLoader
from src.metrics import jaccard, kl_divergence_symmetric, kl_similarity_symmetric


def make_quantile_norm(dist_matrix, threshold, n_bins=10, power=0.4):
    """
    Build a threshold-centered diverging color mapping.

    Distances below threshold are shown in green shades (more similar), while
    values above threshold are shown in pink/magenta shades (less similar).
    """
    vals = dist_matrix.flatten()
    below = vals[vals <= threshold]
    above = vals[vals >= threshold]

    q_below = np.quantile(below, np.linspace(0, 1, n_bins + 1)[:-1]) if len(below) > 1 else np.array([vals.min()])
    q_above = np.quantile(above, np.linspace(0, 1, n_bins + 1)[1:]) if len(above) > 1 else np.array([vals.max()])

    boundaries = np.unique(np.concatenate([q_below, [threshold], q_above]))
    boundaries = np.sort(boundaries)
    boundaries = boundaries[np.concatenate([[True], np.diff(boundaries) > 1e-12])]

    n_colors = len(boundaries) - 1
    n_left = int(np.searchsorted(boundaries, threshold, side="right")) - 1
    n_right = n_colors - n_left

    base = plt.get_cmap("PiYG")
    greens = [base(0.5 + 0.5 * ((n_left - 1 - i) / max(n_left - 1, 1)) ** power) for i in range(n_left)]
    pinks = [base(0.5 - 0.25 * (i / max(n_right - 1, 1)) ** power) for i in range(n_right)]
    cmap = ListedColormap(greens + pinks)
    norm = BoundaryNorm(boundaries, ncolors=n_colors)
    return norm, cmap


def compute_matrices(hap1_data, hap2_data, sim_func, dist_func):
    n = len(hap1_data)
    sim_matrix = np.zeros((n, n))
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = sim_func(hap1_data[i], hap2_data[j])
            d = dist_func(hap1_data[i], hap2_data[j])
            if isinstance(s, torch.Tensor):
                s = s.item()
            if isinstance(d, torch.Tensor):
                d = d.item()
            sim_matrix[i, j] = s
            dist_matrix[i, j] = d
    return sim_matrix, dist_matrix


def euclidean_dist(p, q):
    return torch.norm(p - q).item()


def euclidean_sim(p, q):
    return 1.0 / (1.0 + euclidean_dist(p, q))


def plot_separability(all_results, output_dir):
    """Plot pooled diagonal vs off-diagonal distance distributions per metric."""
    log_scale_metrics = {"KL_Symmetric"}
    fontsize = 21

    metric_names = list(all_results.keys())
    fig, axes = plt.subplots(1, len(metric_names), figsize=(7 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]

    legend_handles = None
    legend_labels = None

    for ax, name in zip(axes, metric_names):
        auroc, _, threshold, diag_vals, off_vals = all_results[name]
        use_log = name in log_scale_metrics

        if use_log:
            eps = 1e-6
            diag_plot = np.clip(diag_vals, eps, None)
            off_plot = np.clip(off_vals, eps, None)
            sns.kdeplot(diag_plot, ax=ax, label="Same chromosome", fill=True, alpha=0.4, color="steelblue", log_scale=True)
            sns.kdeplot(off_plot, ax=ax, label="Different chromosome", fill=True, alpha=0.4, color="tomato", log_scale=True)
            ax.set_xlabel("Distance (log scale)", fontsize=fontsize)
        else:
            sns.kdeplot(diag_vals, ax=ax, label="Same chromosome", fill=True, alpha=0.4, color="steelblue")
            sns.kdeplot(off_vals, ax=ax, label="Different chromosome", fill=True, alpha=0.4, color="tomato")
            ax.set_xlabel("Distance", fontsize=fontsize)

        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5)
        ylim = ax.get_ylim()
        ax.text(
            threshold,
            ylim[1] * 0.97,
            f"{threshold:.3g}",
            ha="center",
            va="top",
            fontsize=fontsize - 1,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.7},
            rotation=90,
        )

        ax.figure.canvas.draw()
        thresh_ax_x = ax.transData.transform((threshold, 0))[0]
        thresh_ax_frac = (thresh_ax_x - ax.bbox.x0) / ax.bbox.width
        if thresh_ax_frac > 0.5:
            auroc_x, auroc_ha = 0.03, "left"
        else:
            auroc_x, auroc_ha = 0.97, "right"

        ax.text(
            auroc_x,
            0.95,
            f"AUROC = {auroc:.4f}",
            transform=ax.transAxes,
            ha=auroc_ha,
            va="top",
            fontsize=fontsize,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "grey", "alpha": 0.8},
        )

        ax.set_title(name, fontsize=fontsize + 2, fontweight="bold")
        ax.set_ylabel("Density", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize - 1)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlim(left=1e-3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.get_legend():
            ax.get_legend().remove()

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=2,
            fontsize=fontsize,
            frameon=True,
            bbox_to_anchor=(0.5, -0.2),
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "separability_analysis.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved separability plot -> {out_path}")


def main():
    data_dir = "data/T2T"
    loader = ChromosomeDataLoader(data_dir=data_dir)
    individuals = loader.available_individuals
    chr_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

    metrics = {
        "Jaccard": (jaccard, lambda p, q: 1.0 - jaccard(p, q)),
        "KL_Symmetric": (kl_similarity_symmetric, kl_divergence_symmetric),
        "Euclidean": (euclidean_sim, euclidean_dist),
    }

    output_dir = "outputs/analysis/heatmaps_hybrid"
    os.makedirs(output_dir, exist_ok=True)

    pooled = {name: {"diag": [], "off": []} for name in metrics}
    stored = []

    for individual in individuals:
        print(f"Pass 1 - {individual}...")
        hap1_data, hap2_data, valid_chrs = [], [], []
        for chrom in chr_order:
            try:
                h1 = loader.get_chromosome(individual, chrom, "hap1", as_distribution=True, max_val=5000)
                h2 = loader.get_chromosome(individual, chrom, "hap2", as_distribution=True, max_val=5000)
                hap1_data.append(h1)
                hap2_data.append(h2)
                valid_chrs.append(chrom)
            except (KeyError, ValueError):
                continue
        if not hap1_data:
            continue

        for name, (sim_func, dist_func) in metrics.items():
            sim_matrix, dist_matrix = compute_matrices(hap1_data, hap2_data, sim_func, dist_func)
            stored.append(
                {
                    "individual": individual,
                    "name": name,
                    "sim_matrix": sim_matrix,
                    "dist_matrix": dist_matrix,
                    "valid_chrs": list(valid_chrs),
                }
            )
            n = dist_matrix.shape[0]
            mask_diag = np.eye(n, dtype=bool)
            pooled[name]["diag"].append(dist_matrix[mask_diag])
            pooled[name]["off"].append(dist_matrix[~mask_diag])

    print("\n=== Separability Analysis (pooled across all individuals) ===")
    print(f"{'Metric':<15} {'AUROC':>8} {'Cohen\'s d':>10} {'Threshold':>12}")
    print("-" * 50)

    sep_results = {}
    thresholds_by_metric = {}
    summary_rows = []
    for name in metrics:
        diag_all = np.concatenate(pooled[name]["diag"])
        off_all = np.concatenate(pooled[name]["off"])
        y_true = np.concatenate([np.ones(len(diag_all)), np.zeros(len(off_all))])
        y_scores = np.concatenate([-diag_all, -off_all])

        auroc = roc_auc_score(y_true, y_scores)
        pooled_std = np.sqrt((diag_all.std() ** 2 + off_all.std() ** 2) / 2.0)
        cohens_d = (off_all.mean() - diag_all.mean()) / (pooled_std + 1e-12)

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        best_idx = np.argmax(tpr - fpr)
        threshold = float(-roc_thresholds[best_idx])

        thresholds_by_metric[name] = threshold
        sep_results[name] = (auroc, cohens_d, threshold, diag_all, off_all)

        print(f"{name:<15} {auroc:>8.4f} {cohens_d:>10.3f} {threshold:>12.4f}")
        summary_rows.append({"Metric": name, "AUROC": auroc, "Cohen's d": cohens_d, "Threshold": threshold})

    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "separability_summary.csv"), index=False)
    plot_separability(sep_results, output_dir)

    print("\nPass 2 - plotting heatmaps in restored hybrid style...")
    for entry in stored:
        individual = entry["individual"]
        name = entry["name"]
        sim_matrix = entry["sim_matrix"]
        dist_matrix = entry["dist_matrix"]
        valid_chrs = entry["valid_chrs"]
        _ = thresholds_by_metric[name]

        vmin = 0.4 if name == "Euclidean" else 0.0

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            sim_matrix,
            xticklabels=valid_chrs,
            yticklabels=valid_chrs,
            cmap="magma",
            annot=dist_matrix,
            fmt=".2f",
            square=True,
            vmin=vmin,
            vmax=1,
            cbar_kws={"label": "Distance (d)"},
        )

        cbar = ax.collections[0].colorbar
        if cbar is not None:
            ticks = cbar.get_ticks()
            dist_labels = []
            for tick in ticks:
                if name == "Jaccard":
                    dist_value = 1.0 - tick
                elif "KL" in name:
                    dist_value = -np.log(max(tick, 1e-9))
                elif name == "Euclidean":
                    dist_value = (1.0 / max(tick, 1e-9)) - 1.0
                else:
                    dist_value = 0
                dist_labels.append(f"{dist_value:.1f}")
            cbar.set_ticks(ticks.tolist())
            cbar.set_ticklabels(dist_labels)

        plt.title(f"{individual} - {name}", fontsize=16)
        plt.xlabel("Hap2")
        plt.ylabel("Hap1")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        filename = f"{output_dir}/{individual.replace('.', '_')}_{name.lower()}.png"
        plt.savefig(filename, dpi=100)
        plt.close()


if __name__ == "__main__":
    main()
