# %% [markdown]
# # Genomic Centeny Map Clustering Analysis
# This notebook unifies experiments for evaluating and visualizing chromosome clustering 
# using different distance metrics.

# %%
%load_ext autoreload
%autoreload 2
import os
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import colorcet as cc

from src.dataloader import ChromosomeDataLoader
from src.metrics import jaccard, kl_divergence_symmetric, kl_divergence
from src.plot.clustering import plot_embedding

# %% [markdown]
# ## 1. Load Data
# We use the T2T genomes preprocessed into Centeny maps (distances between CENP-B boxes).

# %%
data_dir = "data/T2T"
loader = ChromosomeDataLoader(data_dir=data_dir)
X, y, metadata = loader.load_data(as_distribution=True, max_val=5000, return_tensors=True)

df_meta = pd.DataFrame(metadata, columns=['individual', 'haplotype', 'chromosome'])
chr_order = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
df_meta['chromosome'] = pd.Categorical(df_meta['chromosome'], categories=chr_order, ordered=True)
labels = df_meta['chromosome'].values

# %% [markdown]
# ## 2. Distance Matrix Calculation
# Function for computing symmetric distance matrices based on different metrics.

# %%
def compute_distance_matrix(X, metric_func):
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    X_torch = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
    for i in range(n):
        for j in range(i + 1, n):
            d = metric_func(X_torch[i], X_torch[j])
            if isinstance(d, torch.Tensor):
                d = d.item()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

# %% [markdown]
# ## 3. Experiment: Compare All Metrics
# Quantitative evaluation (Silhouette, 1-NN Accuracy, NMI) and visualization.

# %%
metrics = {
    # 'Jaccard': lambda p, q: 1.0 - jaccard(p, q),
    'KL_Symmetric': lambda p, q: kl_divergence_symmetric(p, q),
    # 'KL_Non_Symmetric': lambda p, q: kl_divergence(p, q),
    'Raw_Euclidean': None
}

eval_results = []
palette = cc.glasbey_category10[:24]

for name, metric_func in metrics.items():
    print(f"\n--- Processing Metric: {name} ---")
    
    if metric_func is not None:
        dist_matrix = compute_distance_matrix(X, metric_func)
        dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=1e6, neginf=0.0)
        # Symmetrize if needed
        if name == 'KL_Non_Symmetric':
             dist_matrix = (dist_matrix + dist_matrix.T) / 2
        
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=8)
        coords = tsne.fit_transform(dist_matrix)
    else:
        # Euclidean case on histograms
        tsne = TSNE(n_components=2, metric='euclidean', init='random', random_state=9)
        coords = tsne.fit_transform(X.numpy())
    
    # Quantitative Eval
    # 1. Silhouette
    sil = silhouette_score(coords, labels)
    
    # 2. 1-NN Accuracy (Leave-One-Out consistency)
    dist_sq = np.sum((coords[:, np.newaxis] - coords[np.newaxis, :])**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    nearest_neighbors = np.argmin(dist_sq, axis=1)
    knn_acc = np.mean(labels[nearest_neighbors] == labels)
    
    # 3. NMI (via K-Means clustering)
    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42, n_init=10)
    clusters = kmeans.fit_predict(coords)
    nmi = normalized_mutual_info_score(labels, clusters)
    
    eval_results.append({
        'Metric': name,
        'Silhouette': sil,
        '1-NN Acc': knn_acc,
        'NMI': nmi
    })
    
    # Visualization
    plot_embedding(
        coords, 
        labels, 
        # title=f"t-SNE with {name}", 
        filename=f"tsne_{name.lower()}.png", 
        show_labels=True # Show text labels for the best metric
    )

# %% [markdown]
# ## 4. Final Comparison
# Summary table of the clustering quality results.

# %%
df_results = pd.DataFrame(eval_results)
print("\nClustering Quality Metrics Summary:")
df_results

# %%
