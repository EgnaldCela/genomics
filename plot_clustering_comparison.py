import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

from src.dataloader import ChromosomeDataLoader
from src.metrics import jaccard, kl_divergence_symmetric

# 1. Load Data
data_dir = "data/T2T"
loader = ChromosomeDataLoader(data_dir=data_dir)
X, y, metadata = loader.load_data(as_distribution=True, max_val=5000, return_tensors=True)

df_meta = pd.DataFrame(metadata, columns=['individual', 'haplotype', 'chromosome'])
chr_order = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
df_meta['chromosome'] = pd.Categorical(df_meta['chromosome'], categories=chr_order, ordered=True)
labels = df_meta['chromosome'].values

# 2. Distance Matrix Calculation
def compute_distance_matrix(X, metric_func):
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = metric_func(X[i], X[j]).item()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

# 3. Compute Metrics
metrics = {
    'Euclidean': None,
    'Jaccard': lambda p, q: 1.0 - jaccard(p, q),
    'KL': lambda p, q: kl_divergence_symmetric(p, q),
}

eval_results = []

for name, metric_func in metrics.items():
    print(f"Processing {name}...")
    if metric_func is not None:
        dist_matrix = compute_distance_matrix(X, metric_func)
        dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=1e6, neginf=0.0)
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=8)
        coords = tsne.fit_transform(dist_matrix)
    else:
        tsne = TSNE(n_components=2, metric='euclidean', init='random', random_state=8)
        coords = tsne.fit_transform(X.numpy())
    
    # Silhouette
    sil = silhouette_score(coords, labels)
    
    # 1-NN Accuracy
    dist_sq = np.sum((coords[:, np.newaxis] - coords[np.newaxis, :])**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    nearest_neighbors = np.argmin(dist_sq, axis=1)
    knn_acc = np.mean(labels[nearest_neighbors] == labels)
    
    # NMI
    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42, n_init=10)
    clusters = kmeans.fit_predict(coords)
    nmi = normalized_mutual_info_score(labels, clusters)
    
    eval_results.append({
        'Metric': name,
        'Silhouette': sil,
        '1-NN Acc': knn_acc,
        'NMI': nmi
    })

df_results = pd.DataFrame(eval_results)

# 4. Plotting
df_melted = df_results.melt(id_vars='Metric', var_name='Quality Measure', value_name='Score')

plt.figure(figsize=(7, 7))
sns.set_style("whitegrid")

ax = sns.barplot(data=df_melted, x='Score', y='Quality Measure', hue='Metric', palette='viridis', orient='h')

# Use modern bar_label for cleaner annotations
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=-50, fontweight='bold', color='white', fontsize=11)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.title("Comparison of Clustering Quality Across Metrics", fontsize=16, fontweight='bold')
plt.ylabel("", fontsize=12)
plt.xlabel("", fontsize=12)
plt.xlim(0, 1)
plt.legend(title="Distance Metric", loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

filename = "clustering_quality_comparison.png"
plt.savefig(filename, dpi=300)
print(f"Plot saved to {filename}")
plt.show()
