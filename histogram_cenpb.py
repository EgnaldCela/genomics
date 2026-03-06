
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
    data_path = Path("/home/tao/hdd/genomics/data/T2T/RPE1v1.1.pt")
    if not data_path.exists():
        print(f"Error: {data_path} does not exist.")
        return

    print(f"Loading {data_path}...")
    
    # Load dictionary directly
    try:
        data = torch.load(data_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print(f"Loaded {len(data)} sequences.")
    
    # Inspect keys
    if not data:
        print("Data is empty.")
        return
        
    sample_key = next(iter(data.keys()))
    print(f"Sample key: {sample_key} (type: {type(sample_key)})")
    
    # 2. Analyze frequent distances to match the plot
    all_distances = []
    
    # Filter for chromosomes 1-22 and X
    chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    haps = ["hap1", "hap2"]
    
    sequences_map = {}
    
    for key, seq_tensor in data.items():
        # Key structure: (individual, haplotype, chromosome)
        if isinstance(key, tuple) and len(key) == 3:
            ind, hap, chrom = key
            
            # Filter for RPE1v1.1 just in case file has mix, though filename suggests specific
            # Actually the key likely contains 'RPE1v1.1'
            if chrom in chroms and hap in haps:
                # sequences might be tensors
                if isinstance(seq_tensor, torch.Tensor):
                    seq = seq_tensor.numpy()
                else:
                    seq = np.array(seq_tensor)
                    
                sequences_map[(chrom, hap)] = seq
                all_distances.extend(seq)
        else:
            print(f"Skipping key {key}: unexpected format")

    counts = collections.Counter(all_distances)
    most_common = counts.most_common(50)
    print("\nTop 50 most common distances:")
    for dist, count in most_common:
        print(f"{dist}: {count}")

    # 3. Define specific fragment lengths based on image and common values
    # Based on visual inspection of the image provided in prompt
    # and common biology of CENP-B boxes ~171bp monomers?
    # The image shows clusters around 150, 320, 490, 660, 830, 1000+
    # These are roughly multiples of ~171bp? No.
    # 171? No, 171 is alpha satellite monomer length.
    # CENP-B box is 17bp motif.
    # Distances between boxes.
    
    # Let's use the exact values from the image labels
    # Labels: 
    # 148, 149, 150, 151, 152, 153, 154, 155
    # 318, 319, 320, 321, 322, 323, 324, 325
    # 489, 491, 492, 493, 494, 495, 496, 497
    # 511
    # 658, 659, 660, 661, 662, 663, 664, 665, 666, 667
    # 678
    # 830, 832, 833, 834, 837
    # 1001, 1003, 1172, 1345

    selected_lengths = []
    selected_lengths.extend(range(148, 156))
    selected_lengths.extend(range(318, 326))
    selected_lengths.extend([489, 491, 492, 493, 494, 495, 496, 497])
    selected_lengths.append(511)
    selected_lengths.extend(range(658, 668)) # 658-667
    selected_lengths.append(678)
    selected_lengths.extend([830, 832, 833, 834, 837])
    selected_lengths.extend([1001, 1003, 1172, 1345])
    
    # Count occurrences per chromosome/haplotype for these lengths
    # And compute PERCENTAGE
    
    heatmap_data = [] # List of columns
    x_labels = []
    
    for chrom in chroms:
        # For each chromosome, we want hap1 then hap2
        for hap in haps:
            key = (chrom, hap)
            col_counts = []
            
            if key in sequences_map:
                seq = sequences_map[key]
                total = len(seq)
                cnt = collections.Counter(seq)
                
                for length in selected_lengths:
                    count = cnt.get(length, 0)
                    pct = (count / total * 100) if total > 0 else 0
                    col_counts.append(pct)
            else:
                col_counts = [0] * len(selected_lengths)
                
            heatmap_data.append(col_counts)
            x_labels.append(f"{chrom}\n{hap.replace('hap', '')}")

    # Transpose to get (lengths x samples)
    heatmap_matrix = np.array(heatmap_data).T
    
    # Plot
    plt.figure(figsize=(20, 15))
    
    # Create mask for zero values to plot them white/distinct if needed?
    # The image shows white background for 0.
    # sns.heatmap supports mask.
    
    ax = sns.heatmap(heatmap_matrix, cmap="RdPu", center=0, vmin=0, vmax=80,
                     yticklabels=selected_lengths, xticklabels=x_labels,
                     annot=False, square=False,
                     cbar_kws={'label': 'Percentage of occ.'})
    
    # RdBu is Red-Blue. Image is White-Red-Blue? 
    # Image: White (0) -> Light Red -> Dark Red -> Dark Blue (high)
    # The scale bar shows 0 (white) to red to blue (80).
    # So it is a diverging colormap or custom sequential.
    # "coolwarm" or "RdBu_r" might work but we need 0 to be white.
    # RdBu: Red (low) -> White (mid) -> Blue (high).
    # If 0 is low, it's Red. We want White.
    # So we probably want a custom cmap: White -> Red -> Blue.
    
    # Create custom colormap
    # White -> Red -> Dark Red -> Dark Blue
    # The image shows clear distinction. 
    # 0 is white.
    # Low values (up to maybe 20-30%) are reddish.
    # High values (dominant frequencies) are dark blue.
    # The colorbar goes up to 80.
    
    colors = ["#ffffff", "#ffcccc", "#ff0000", "#8b0000", "#000080"]
    # Adjust nodes to control the transition points
    # 0 -> White
    # 0.05 -> Light Red (low freq)
    # 0.3 -> Red (mid freq)
    # 0.5 -> Dark Red (mid-high freq)
    # 1.0 -> Dark Blue (high freq ~80%)
    nodes = [0.0, 0.3, 0.5, 0.7, 1.0] 
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
    
    # Prepare the figure and axes
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Use seaborn heatmap
    # yticklabels need to match selected_lengths
    # xticklabels need to match columns
    
    # Generate labels: 1, 2, 1, 2 ...
    x_tick_labels = ["1", "2"] * len(chroms)
    
    sns.heatmap(heatmap_matrix, ax=ax, cmap=cmap, vmin=0, vmax=80,
                     yticklabels=selected_lengths, 
                     xticklabels=x_tick_labels,
                     annot=False, square=False,
                     cbar_kws={'label': 'Percentage of occ.', 'ticks': [0, 20, 40, 60, 80]})

    # Move y-axis labels to the right
    ax.tick_params(axis='y', labelright=True, labelleft=False, labelsize=13, rotation=0)
    ax.set_ylabel("Fragment length (bp)", fontsize=16)
    ax.xaxis.set_ticks_position('none') # Hide x-axis ticks
    ax.yaxis.set_ticks_position('none') 
    
    ax.yaxis.set_label_position("right")

    # Add harder horizontal lines between clusters
    # Detect gaps in selected_lengths
    for i in range(len(selected_lengths) - 1):
        current_val = selected_lengths[i]
        next_val = selected_lengths[i+1]
        
        # If the gap is larger than 20, consider it a cluster break
        if next_val - current_val > 20:
            # Draw line at index i+1 (which is the boundary in heatmap coordinates 0 to N)
            # The heatmap rows are indexed 0 (top) to N-1 (bottom) or vice versa?
            # sns.heatmap usually plots row 0 at the top.
            # So index i+1 corresponds to the line between row i and row i+1
            ax.axhline(i + 1, color='black', linewidth=1, alpha=0.7)

    ax.set_title("CENP-B box distances (bp) in the diploid RPE-1 genome", fontsize=20, pad=20)
    ax.set_ylabel("Fragment length (bp)", fontsize=16)
    plt.grid(False)  # Remove grid lines for cleaner look
    
    # Remove default x-label "HAP" as we will add custom labels
    ax.set_xlabel("", fontsize=16)
    
    # Add chromosome labels below the haplotype labels
    
    # Chromosome list
    chrom_labels = chroms
    
    # Place labels
    # Heatmap x-coordinates are 0.5, 1.5, ...
    # We want labels centered under each pair (hap1, hap2)
    # Pair 0 (chr1): cols 0, 1 -> centered at 1.0
    # Pair i: cols 2*i, 2*i+1 -> centered at 2*i + 1.0
    
    for i, label in enumerate(chrom_labels):
        x_pos = 2 * i + 1.0
        # y_pos is slightly below the plot. 
        # transform=ax.get_xaxis_transform() puts y in axes coordinates (0=bottom, 1=top)
        # But for text outside, might be easier to use data coords relative to ylim
        # The y-axis limits depend on len(selected_lengths).
        # Let's use ax.text with transform=ax.transData or relative to bottom.
        
        # Using transform=ax.transData (default)
        # y-axis is inverted in heatmap (0 at top, N at bottom).
        # So "below" is > len(selected_lengths).
        y_pos = len(selected_lengths) + 0.5 
        
        ax.text(x_pos, y_pos, label, 
                ha='center', va='top', fontsize=10, rotation=90)
        
        # Add a subtle vertical line to separate chromosomes
        ax.axvline(i *2 +1, color='gray', lw=1, alpha=0.5, ls='--', dashes=(10, 10))
        if i < len(chrom_labels):
            ax.axvline((i + 1) * 2, color='gray', lw=1, alpha=0.7)

    # Add "HAP:" label at the bottom right corner
    ax.text(len(chrom_labels) * 2 - 2, len(selected_lengths) + 3, "HAP:", ha='right', va='top', fontsize=12)
    plt.tight_layout()
    # Adjust bottom margin to make room for labels
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig("reproduce_heatmap.svg", dpi=150, format='svg')
    print("Saved reproduce_heatmap.svg")

if __name__ == "__main__":
    main()
