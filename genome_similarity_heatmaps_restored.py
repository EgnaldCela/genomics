import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.dataloader import ChromosomeDataLoader
from src.metrics import (jaccard, kl_similarity_symmetric, 
                         kl_divergence_symmetric)

def compute_matrices(hap1_data, hap2_data, sim_func, dist_func):
    n = len(hap1_data)
    sim_matrix = np.zeros((n, n))
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = sim_func(hap1_data[i], hap2_data[j])
            d = dist_func(hap1_data[i], hap2_data[j])
            if isinstance(s, torch.Tensor): s = s.item()
            if isinstance(d, torch.Tensor): d = d.item()
            sim_matrix[i, j] = s
            dist_matrix[i, j] = d
    return sim_matrix, dist_matrix

def euclidean_dist(p, q):
    return torch.norm(p - q).item()

def euclidean_sim(p, q):
    return 1.0 / (1.0 + euclidean_dist(p, q))

def main():
    data_dir = "data/T2T"
    loader = ChromosomeDataLoader(data_dir=data_dir)
    individuals = loader.available_individuals
    chr_order = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    
    metrics = {
        'Jaccard': (jaccard, lambda p, q: 1.0 - jaccard(p, q)),
        'KL_Symmetric': (kl_similarity_symmetric, kl_divergence_symmetric),
        'Euclidean': (euclidean_sim, euclidean_dist)
    }
    
    output_dir = "plots/heatmaps_hybrid"
    os.makedirs(output_dir, exist_ok=True)
    
    for individual in individuals:
        print(f"Processing {individual}...")
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
        if not hap1_data: continue
            
        for name, (sim_func, dist_func) in metrics.items():
            plt.figure(figsize=(12, 10))
            sim_matrix, dist_matrix = compute_matrices(hap1_data, hap2_data, sim_func, dist_func)
            
            # vmin = 0.4 for Euclidean, 0.0 for others
            vmin = 0.4 if name == 'Euclidean' else 0.0
            
            ax = sns.heatmap(
                sim_matrix, xticklabels=valid_chrs, yticklabels=valid_chrs, 
                cmap='magma', annot=dist_matrix, fmt=".2f", square=True,
                vmin=vmin, vmax=1, cbar_kws={'label': 'Distance (d)'}
            )
            
            # Simplified colorbar: only distance labels
            cbar = ax.collections[0].colorbar
            ticks = cbar.get_ticks()
            dist_labels = []
            for t in ticks:
                if name == 'Jaccard': d = 1.0 - t
                elif 'KL' in name: d = -np.log(max(t, 1e-9))
                elif name == 'Euclidean': d = (1.0 / max(t, 1e-9)) - 1.0
                else: d = 0
                dist_labels.append(f"{d:.1f}")
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(dist_labels)
            
            plt.title(f"{individual} - {name}", fontsize=16)
            plt.xlabel("Hap2"), plt.ylabel("Hap1")
            plt.xticks(rotation=45, ha='right'), plt.tight_layout()
            
            filename = f"{output_dir}/{individual.replace('.', '_')}_{name.lower()}.png"
            plt.savefig(filename, dpi=100), plt.close()

if __name__ == "__main__":
    main()
