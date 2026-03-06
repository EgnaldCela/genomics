from matplotlib.colors import LinearSegmentedColormap, LogNorm
import collections
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def main():
    np.random.seed(42)
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "T2T"
    output_dir = repo_root / "outputs" / "analysis" / "multiple_genomes"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("chm13.pt", "CHM13", [""]),
        ("HG002v1.1.pt", "HG002", ["hap1", "hap2"]),
        ("RPE1v1.1.pt", "RPE1", ["hap1", "hap2"]),
        ("YAOv2.0.pt", "YAO", ["hap1", "hap2"]),
    ]
    male_genomes = {"HG002", "YAO"}

    loaded_data = {}
    for filename, label, _ in datasets:
        path = data_dir / filename
        if not path.exists():
            print(f"Error: {path} does not exist.")
            continue

        print(f"Loading {path}...")
        loaded_data[label] = torch.load(path)

    chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

    selected_lengths = []
    selected_lengths.extend(range(148, 156))
    selected_lengths.extend(range(318, 326))
    selected_lengths.extend([489, 491, 492, 493, 494, 495, 496, 497])
    selected_lengths.append(511)
    selected_lengths.extend(range(658, 668))
    selected_lengths.append(678)
    selected_lengths.extend([830, 832, 833, 834, 837])
    selected_lengths.extend([1001, 1003, 1172, 1345])

    heatmap_data = []
    col_info = []

    for chrom in chroms:
        for _, label, haps in datasets:
            if label not in loaded_data:
                continue

            data = loaded_data[label]

            for target_hap in haps:
                if chrom == "chrX" and label in male_genomes and target_hap == "hap1":
                    continue

                found_seq = None
                for key_tuple, seq_tensor in data.items():
                    _, k_hap, k_chrom = key_tuple
                    if k_chrom == chrom and k_hap == target_hap:
                        found_seq = seq_tensor.numpy() if isinstance(seq_tensor, torch.Tensor) else np.array(seq_tensor)
                        break

                if found_seq is not None:
                    total = len(found_seq)
                    cnt = collections.Counter(found_seq)
                    col_counts = [(cnt.get(length, 0) / total * 100) if total > 0 else 0 for length in selected_lengths]
                else:
                    col_counts = [0.0] * len(selected_lengths)

                heatmap_data.append(col_counts)
                col_info.append((chrom, label, target_hap))

    heatmap_matrix = np.array(heatmap_data).T

    colors = ["#ffffff", "#ffcccc", "#ff0000", "#8b0000", "#000080"]
    nodes = [0.0, 0.3, 0.5, 0.7, 1.0]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    heatmap_matrix_display = np.where(heatmap_matrix > 0, heatmap_matrix, 0.01)
    log_ticks = [0.01, 0.1, 1, 10, 80]
    log_ticklabels = ["0.01%", "0.1%", "1%", "10%", "80%"]
    norm = LogNorm(vmin=0.01, vmax=80)

    num_cols = heatmap_matrix.shape[1]
    fig_width = max(20, num_cols * 0.25)
    fig, ax = plt.subplots(figsize=(fig_width, 12))

    bottom_tick_locs = []
    bottom_tick_labels = []

    col_idx = 0
    for chrom in chroms:
        for _, label, _ in datasets:
            if label not in loaded_data:
                continue
            count = sum(1 for (c, l, _) in col_info if c == chrom and l == label)
            if count == 0:
                continue
            center = col_idx + count / 2.0
            bottom_tick_locs.append(center)
            bottom_tick_labels.append(label)
            col_idx += count

    sns.heatmap(
        heatmap_matrix_display,
        ax=ax,
        cmap=cmap,
        norm=norm,
        yticklabels=selected_lengths,
        xticklabels=False,
        annot=False,
        square=False,
        cbar_kws={"label": "Percentage of occ.", "ticks": log_ticks},
    )

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.set_ticklabels(log_ticklabels)

    ax.tick_params(axis="y", labelright=True, labelleft=False, labelsize=13, rotation=0)
    ax.set_xticks(bottom_tick_locs)
    ax.set_xticklabels(bottom_tick_labels, fontsize=10, rotation=90)
    ax.tick_params(axis="x", length=0)

    ax.set_ylabel("Fragment length (bp)", fontsize=16)
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.yaxis.set_label_position("right")

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    chrom_start = {}
    chrom_end = {}
    for idx, (c, _, _) in enumerate(col_info):
        if c not in chrom_start:
            chrom_start[c] = idx
        chrom_end[c] = idx + 1

    tick_locs = []
    tick_labels = []
    for chrom in chroms:
        if chrom in chrom_start:
            center = (chrom_start[chrom] + chrom_end[chrom]) / 2.0
            tick_locs.append(center)
            tick_labels.append(chrom)

    ax_top.set_xticks(tick_locs)
    ax_top.set_xticklabels(tick_labels, rotation=0, fontsize=12, fontweight="bold")
    ax_top.tick_params(axis="x", length=0)

    ax_top.spines["top"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    for i in range(len(selected_lengths) - 1):
        if selected_lengths[i + 1] - selected_lengths[i] > 20:
            ax.axhline(i + 1, color="gray", linewidth=2, alpha=0.5)

    total_cols = len(col_info)
    for chrom in chroms:
        if chrom not in chrom_end:
            continue
        x_pos = chrom_end[chrom]
        if x_pos < total_cols:
            ax.axvline(x_pos, color="gray", lw=2, alpha=0.9)

        base = chrom_start[chrom]
        col_cursor = base
        for _, label, _ in datasets:
            if label not in loaded_data:
                continue
            count = sum(1 for (c, l, _) in col_info if c == chrom and l == label)
            if count == 0:
                continue
            col_cursor += count
            if col_cursor < chrom_end[chrom]:
                ax.axvline(col_cursor, color="gray", lw=1, alpha=0.5, ls="--", dashes=(5, 5))

    legend_text = "Order per Chromosome: CHM13 | HG002(h1,h2) | RPE1(h1,h2) | YAO(h1,h2)"
    ax.text(0, -2, legend_text, ha="left", va="bottom", fontsize=14, transform=ax.get_xaxis_transform())

    ax.set_title("CENP-B box distances across T2T Genomes", fontsize=20, pad=30)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.90)

    svg_path = output_dir / "multiple_genomes_heatmap.svg"
    png_path = output_dir / "multiple_genomes_heatmap.png"
    plt.savefig(svg_path, dpi=150, format="svg")
    plt.savefig(png_path, dpi=150, format="png")
    print(f"Saved {svg_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
