import os
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import colorcet as cc
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataloader import ChromosomeDataLoader
from src.metrics import kl_divergence_symmetric

# 1. Load Data
data_dir = "data/T2T"
loader = ChromosomeDataLoader(data_dir=data_dir)

def run_tsne_analysis(exclude_list=None, mark_individual=None, suffix=""):
    """
    General function to run t-SNE and plot results.
    """
    if exclude_list is None:
        exclude_list = []
        
    # Reload data
    X, y, metadata = loader.load_data(as_distribution=True, max_val=5000, return_tensors=True)
    df_meta = pd.DataFrame(metadata, columns=['individual', 'haplotype', 'chromosome'])
    
    # Filter out excluded individuals
    if exclude_list:
        mask = ~df_meta['individual'].isin(exclude_list)
        X = X[mask]
        df_meta = df_meta[mask].reset_index(drop=True)
        print(f"\n--- Running analysis excluding {exclude_list} ---")
        print(f"Remaining individuals: {df_meta['individual'].unique()}")
    else:
        print("\n--- Running analysis with all individuals ---")
        print(f"Individuals present: {df_meta['individual'].unique()}")
        
    chr_order = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    df_meta['chromosome'] = pd.Categorical(df_meta['chromosome'], categories=chr_order, ordered=True)
    labels = df_meta['chromosome'].values
    
    # Compute Distance Matrix (KL Symmetric)
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = kl_divergence_symmetric(X[i], X[j]).item()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=1e6, neginf=0.0)
    
    # t-SNE
    tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=8)
    coords = tsne.fit_transform(dist_matrix)
    
    # Metrics
    sil = silhouette_score(coords, labels)
    print(f"Metrics ({suffix}): Silhouette={sil:.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 9))
    coords_jittered = coords + np.random.normal(0, 0.4, size=coords.shape)
    
    palette = cc.glasbey_category10[:24]
    
    # Main scatter by chromosome
    ax = sns.scatterplot(
        x=coords_jittered[:, 0], y=coords_jittered[:, 1], 
        hue=labels, palette=palette, s=200, alpha=0.7, legend=False
    )
    
    # Chromosome Centroid Labels (for orientation)
    unique_chrs = np.unique(labels)
    for chrom in unique_chrs:
        mask = labels == chrom
        if mask.any():
            centroid = coords_jittered[mask].mean(axis=0)
            plt.text(centroid[0], centroid[1]-3, str(chrom), fontsize=15, fontweight='bold',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Marking a specific individual
    if mark_individual:
        mark_mask = df_meta['individual'] == mark_individual
        n_marked = np.sum(mark_mask)
        print(f"Highlighting {mark_individual}: found {n_marked} points.")
        
        if n_marked > 0:
            plt.scatter(
                coords_jittered[mark_mask, 0], coords_jittered[mark_mask, 1], 
                facecolors='none', edgecolors='grey', s=400, linewidths=2, alpha=0.9,
                zorder=10, label=f'Highlighted: {mark_individual}'
            )
            plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    
    plt.title(f"t-SNE Chromosome Clustering ({suffix.replace('_', ' ').strip()})", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    
    filename = f"tsne_kl_symmetric_{suffix}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot to {filename}")

if __name__ == "__main__":
    # Case 1: All except hg38
    run_tsne_analysis(exclude_list=['hg38'], suffix="no_hg38")
    
    # Case 2: All except hg38, mark CHM13v2
    run_tsne_analysis(exclude_list=['hg38'], mark_individual='CHM13v2', suffix="no_hg38_mark_chm13")
    
    # Case 3: All including hg38, mark hg38
    run_tsne_analysis(exclude_list=[], mark_individual='hg38', suffix="with_hg38_mark_hg38")
