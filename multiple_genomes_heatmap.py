from matplotlib import ticker
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import os
import pandas as pd
from pathlib import Path

def main():
    # 1. Load Data
    np.random.seed(42)
    data_dir = Path("/home/tao/hdd/genomics/data/T2T")
    
    # Define datasets to load: (filename, label_prefix, expected_haps)
    datasets = [
        ("chm13.pt", "CHM13", [""]),          # CHM13 is haploid-like
        ("HG002v1.1.pt", "HG002", ["hap1", "hap2"]),
        ("RPE1v1.1.pt", "RPE1", ["hap1", "hap2"]),
        ("YAOv2.0.pt", "YAO", ["hap1", "hap2"])
    ]
    # Males: only one X chromosome, skip hap2 for chrX
    male_genomes = {"HG002", "YAO"}

    loaded_data = {}

    for filename, label, haps in datasets:
        path = data_dir / filename
        if not path.exists():
            print(f"Error: {path} does not exist.")
            continue
            
        print(f"Loading {path}...")
        try:
            data = torch.load(path)
            loaded_data[label] = data
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    # 2. Prepare analysis parameters
    chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    
    # Selected lengths from previous script
    selected_lengths = []
    selected_lengths.extend(range(148, 156))
    selected_lengths.extend(range(318, 326))
    selected_lengths.extend([489, 491, 492, 493, 494, 495, 496, 497])
    selected_lengths.append(511)
    selected_lengths.extend(range(658, 668)) 
    selected_lengths.append(678)
    selected_lengths.extend([830, 832, 833, 834, 837])
    selected_lengths.extend([1001, 1003, 1172, 1345])

    heatmap_data = [] # List of columns
    x_labels = []     # Labels for columns
    col_info = []     # (chrom, label, hap) for each column
    
    # Iterate through chromosomes
    for chrom in chroms:
        # For each chromosome, iterate through datasets in order
        for filename, label, haps in datasets:
            if label not in loaded_data:
                continue
                
            data = loaded_data[label]
            
            # Find the key corresponding to this chromosome and required haplotype
            # The keys are (individual, haplotype, chromosome)
            # We need to find the specific key because individual name might vary (e.g. CHM13v2 vs CHM13)
            
            # Helper to find sequence
            for target_hap in haps:
                # Skip hap1 for male genomes on chrX (hap1=Y, hap2=X)
                if chrom == "chrX" and label in male_genomes and target_hap == "hap1":
                    continue

                found_seq = None
                
                # Search for matching key
                # We iterate data.items() is inefficient if large, but keys should be few (~50)
                # Better to construct expected key if we know the individual name
                # But individual name strings vary (HG002v1.1 vs HG002).
                # Let's search.
                
                for key_tuple, seq_tensor in data.items():
                    # key_tuple is (ind, hap, chrom_name)
                    k_ind, k_hap, k_chrom = key_tuple
                    
                    if k_chrom == chrom:
                        # Check haplotype match
                        # CHM13 has empty string hap, or maybe inconsistent?
                        # The tool output showed CHM13v2 with "" hap.
                        
                        if k_hap == target_hap:
                            if isinstance(seq_tensor, torch.Tensor):
                                found_seq = seq_tensor.numpy()
                            else:
                                found_seq = np.array(seq_tensor)
                            break
                
                col_counts = []
                if found_seq is not None:
                    total = len(found_seq)
                    cnt = collections.Counter(found_seq)
                    for length in selected_lengths:
                        count = cnt.get(length, 0)
                        pct = (count / total * 100) if total > 0 else 0
                        col_counts.append(pct)
                else:
                    col_counts = [0.0] * len(selected_lengths)
                
                heatmap_data.append(col_counts)
                col_info.append((chrom, label, target_hap))

                # Label construction
                short_label = label
                if target_hap:
                    h_suffix = target_hap.replace("hap", "h")
                    x_labels.append(f"{short_label}\n{h_suffix}")
                else:
                    x_labels.append(f"{short_label}")


    # Transpose to get (lengths x samples)
    heatmap_matrix = np.array(heatmap_data).T
    
    # Custom colormap
    colors = ["#ffffff", "#ffcccc", "#ff0000", "#8b0000", "#000080"]
    nodes = [0.0, 0.3, 0.5, 0.7, 1.0] 
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    # Clip zeros for log scale (log can't handle 0)
    heatmap_matrix_display = np.where(heatmap_matrix > 0, heatmap_matrix, 0.01)

    # Log-scale norm and ticks
    log_ticks = [0.01, 0.1, 1, 10, 80]
    log_ticklabels = ["0.01%", "0.1%", "1%", "10%", "80%"]
    norm = LogNorm(vmin=0.01, vmax=80)
    
    # Calculate width based on number of columns
    num_cols = heatmap_matrix.shape[1]
    # Estimate width: approx 0.25 inch per column?
    fig_width = max(20, num_cols * 0.25)
    
    fig, ax = plt.subplots(figsize=(fig_width, 12))
    
    # X-axis labels
    # Top X-axis: Chromosome names
    # Bottom X-axis: Genome names
    
    # We will use ax.secondary_xaxis for the top labels
    # And configure the primary x-axis (bottom) for genome labels
    
    # Bottom labels:
    # Instead of h1/h2, we want genome names.
    # To avoid crowding, maybe we just use the first letter or short code if dense?
    # But user asked for "Genome names".
    # Pattern per chromosome: CHM13, HG002, HG002, RPE1, RPE1, YAO, YAO
    
    bottom_labels = []
    # We constructed heatmap_data by iterating chroms then datasets
    # So the order repeats for each chrom.
    
    genome_labels_map = {
        "CHM13": "CHM13",
        "HG002": "HG002", 
        "RPE1": "RPE1",
        "YAO": "YAO"
    }
    
    # To reduce clutter, let's just label the center of each genome block?
    # Or label every column but rotate 90 degrees.
    # Let's label every column with rotation.
    
    # We will generate custom tick positions and labels for the bottom axis
    # to aggregate haplotypes.
    
    bottom_tick_locs = []
    bottom_tick_labels = []

    col_idx = 0
    for chrom in chroms:
        for filename, label, haps in datasets:
            if label not in loaded_data:
                continue
            # Count actual columns for this (chrom, label) combination
            count = sum(
                1 for (c, l, h) in col_info
                if c == chrom and l == label
            )
            if count == 0:
                continue
            center = col_idx + count / 2.0
            bottom_tick_locs.append(center)
            bottom_tick_labels.append(label)
            col_idx += count

    sns.heatmap(heatmap_matrix_display, ax=ax, cmap=cmap, norm=norm,
                     yticklabels=selected_lengths, 
                     xticklabels=False, # We will set them manually
                     annot=False, square=False,
                     cbar_kws={'label': 'Percentage of occ.', 'ticks': log_ticks})

    # Format colorbar tick labels as percentages
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(log_ticklabels)

    ax.tick_params(axis='y', labelright=True, labelleft=False, labelsize=13, rotation=0)
    
    # Set manual bottom ticks
    ax.set_xticks(bottom_tick_locs)
    ax.set_xticklabels(bottom_tick_labels, fontsize=10, rotation=90)
    ax.tick_params(axis='x', length=0)
    
    ax.set_ylabel("Fragment length (bp)", fontsize=16)
    ax.xaxis.set_ticks_position('none') # No tick marks
 
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_label_position("right")
    
    # TOP X-AXIS for Chromosome names
    # Create a secondary x-axis
    ax_top = ax.twiny()

    # Set limits to match the main axis
    ax_top.set_xlim(ax.get_xlim())

    # Compute per-chromosome column ranges dynamically from col_info
    chrom_start = {}
    chrom_end = {}
    for idx, (c, l, h) in enumerate(col_info):
        if c not in chrom_start:
            chrom_start[c] = idx
        chrom_end[c] = idx + 1  # exclusive end

    tick_locs = []
    tick_labels = []
    for chrom in chroms:
        if chrom in chrom_start:
            center = (chrom_start[chrom] + chrom_end[chrom]) / 2.0
            tick_locs.append(center)
            tick_labels.append(chrom)

    ax_top.set_xticks(tick_locs)
    ax_top.set_xticklabels(tick_labels, rotation=0, fontsize=12, fontweight='bold')
    ax_top.tick_params(axis='x', length=0) # Hide tick marks

    # Remove spines
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    # Horizontal lines for clusters Y axis
    for i in range(len(selected_lengths) - 1):
        if selected_lengths[i+1] - selected_lengths[i] > 20:
            ax.axhline(i + 1, color='gray', linewidth=2, alpha=0.5)

    # Vertical lines for Chromosome boundaries (computed dynamically)
    total_cols = len(col_info)

    for chrom in chroms:
        if chrom not in chrom_end:
            continue
        # Draw solid line at end of chromosome block
        x_pos = chrom_end[chrom]
        if x_pos < total_cols:
            ax.axvline(x_pos, color='gray', lw=2, alpha=0.9)

        # Draw dashed lines between genome groups within this chromosome
        base = chrom_start[chrom]
        col_cursor = base
        for filename, label, haps in datasets:
            if label not in loaded_data:
                continue
            count = sum(
                1 for (c, l, h) in col_info
                if c == chrom and l == label
            )
            if count == 0:
                continue
            col_cursor += count
            # Dashed separator after each genome group (except the last)
            if col_cursor < chrom_end[chrom]:
                ax.axvline(col_cursor, color='gray', lw=1, alpha=0.5, ls='--', dashes=(5, 5))

    # Remove the old text-based chromosome labels loop
    # (Previously lines 170-174 in valid file, here we verify context)

    # Add Legend/Key for the columns?
    # Maybe just text annotation explaining the order: CHM13, HG002, RPE1, YAO
    legend_text = "Order per Chromosome: CHM13 | HG002(h1,h2) | RPE1(h1,h2) | YAO(h1,h2)"
    ax.text(0, -2, legend_text, ha='left', va='bottom', fontsize=14, transform=ax.get_xaxis_transform())

    ax.set_title("CENP-B box distances across T2T Genomes", fontsize=20, pad=30)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.90)
    
    output_filename = "multiple_genomes_heatmap.svg"
    plt.savefig(output_filename, dpi=150, format='svg')
    print(f"Saved {output_filename}")

if __name__ == "__main__":
    main()
