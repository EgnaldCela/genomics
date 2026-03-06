import os
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import colorcet as cc
import plotly.express as px
import plotly.graph_objects as go

from src.dataloader import ChromosomeDataLoader
from src.metrics import kl_divergence_symmetric, jaccard

# 1. Load Data
data_dir = "data/T2T"
loader = ChromosomeDataLoader(data_dir=data_dir)

METRICS = {
    'KL_Symmetric': lambda p, q: kl_divergence_symmetric(p, q).item(),
    'Jaccard': lambda p, q: (1.0 - jaccard(p, q)).item(),
    'Raw_Euclidean': None,  # handled separately via sklearn euclidean
}

def compute_distance_matrix(X, metric_func):
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = metric_func(X[i], X[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=1e6, neginf=0.0)
    return dist_matrix

def generate_interactive_tsne(metric_name="KL_Symmetric", filename=None):
    if filename is None:
        filename = f"plots/clusterings/interactive_tsne_{metric_name.lower()}.html"
    print(f"\n--- Generating Interactive t-SNE Plot [{metric_name}] ---")
    
    # Load all data
    X, y, metadata = loader.load_data(as_distribution=True, max_val=5000, return_tensors=True)
    df_meta = pd.DataFrame(metadata, columns=['individual', 'haplotype', 'chromosome'])
    
    chr_order = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    df_meta['chromosome'] = pd.Categorical(df_meta['chromosome'], categories=chr_order, ordered=True)
    
    # Compute Distance Matrix
    metric_func = METRICS[metric_name]
    if metric_func is not None:
        print("  Computing distance matrix...")
        dist_matrix = compute_distance_matrix(X, metric_func)
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=8)
        coords = tsne.fit_transform(dist_matrix)
    else:
        # Raw Euclidean — run directly on histogram vectors
        print("  Running t-SNE with euclidean metric...")
        tsne = TSNE(n_components=2, metric='euclidean', init='random', random_state=8)
        coords = tsne.fit_transform(X.numpy())
    
    df_meta['TSNE1'] = coords[:, 0]
    df_meta['TSNE2'] = coords[:, 1]
    
    # Color palette for chromosomes — convert float RGB to hex for Plotly
    def _to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    color_sequence = [_to_hex(c) for c in cc.glasbey_category10[:24]]
    chrom_color = {chrom: color_sequence[i] for i, chrom in enumerate(chr_order)}

    # Build figure with one trace per (individual, chromosome) to enable genome toggling
    individuals = sorted(df_meta['individual'].unique())
    fig = go.Figure()
    trace_individuals = []
    first_per_chrom = set()

    for ind in individuals:
        for chrom in chr_order:
            mask = (df_meta['individual'] == ind) & (df_meta['chromosome'].astype(str) == chrom)
            if not mask.any():
                continue
            subset = df_meta[mask]
            show_legend = chrom not in first_per_chrom
            if show_legend:
                first_per_chrom.add(chrom)

            is_chm13 = 'chm13' in ind.lower()
            if is_chm13:
                marker = dict(
                    symbol='diamond',
                    color=chrom_color[chrom],
                    size=15,
                    opacity=0.8,
                    line=dict(color=chrom_color[chrom], width=1),
                )
            else:
                marker = dict(
                    symbol='circle',
                    color=chrom_color[chrom],
                    size=13,
                    opacity=0.4,
                    line=dict(color='white', width=1),
                )
            fig.add_trace(go.Scatter(
                x=subset['TSNE1'].values,
                y=subset['TSNE2'].values,
                mode='markers',
                name=chrom,
                legendgroup=chrom,
                showlegend=show_legend,
                marker=marker,
                customdata=subset[['individual', 'haplotype']].values,
                hovertemplate='<b>%{text}</b><br>Individual: %{customdata[0]}<br>Haplotype: %{customdata[1]}<extra></extra>',
                text=subset['chromosome'].astype(str).values,
            ))
            trace_individuals.append(ind)

    # Compute per-chromosome centroids for labels
    centroids = df_meta.groupby('chromosome', observed=True)[['TSNE1', 'TSNE2']].median()
    y_offset = (df_meta['TSNE2'].max() - df_meta['TSNE2'].min()) * 0.018

    fig.add_trace(go.Scatter(
        x=centroids['TSNE1'],
        y=centroids['TSNE2'] - y_offset,
        mode='text',
        text=centroids.index.astype(str).tolist(),
        textposition='bottom center',
        textfont=dict(size=13, color='black', family='Arial Black', weight='bold'),
        showlegend=False,
        hoverinfo='skip',
    ))
    trace_individuals.append(None)  # centroid label trace

    # Build per-genome toggle buttons
    all_visible = [True] * len(trace_individuals)
    genome_buttons = [dict(label="All Genomes", method="restyle", args=[{"visible": all_visible}])]
    for ind in individuals:
        visible = [(t == ind or t is None) for t in trace_individuals]
        genome_buttons.append(dict(label=ind, method="restyle", args=[{"visible": visible}]))

    fig.update_layout(
        title=f"t-SNE of Genomic Centeny Maps [{metric_name}]",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_white",
        width=1100,
        height=800,
        legend_title_text='Chromosomes (Click to filter)',
        hoverlabel=dict(bgcolor="white", font_size=12),
        updatemenus=[dict(
            buttons=genome_buttons,
            direction="down",
            showactive=True,
            x=1.18,
            xanchor="left",
            y=1.0,
            yanchor="top",
            bgcolor="white",
            bordercolor="gray",
            font=dict(size=11),
        )],
        annotations=[dict(
            text="<b>Filter Genome:</b>",
            x=1.18,
            xref="paper",
            y=1.04,
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=12),
        )],
    )

    # Save to HTML with high-resolution download button
    fig.write_html(filename, config={
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'tsne_centeny',
            'scale': 4,  # 4x resolution (~4400x3200px at default size)
        }
    })
    print(f"Interactive plot saved to {filename}")

if __name__ == "__main__":
    os.makedirs("plots/clusterings", exist_ok=True)
    for metric_name in METRICS:
        generate_interactive_tsne(metric_name=metric_name)
