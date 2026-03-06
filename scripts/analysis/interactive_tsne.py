import os

import colorcet as cc
import pandas as pd
import plotly.graph_objects as go
import torch

from src.analysis import compute_pairwise_distance_matrix, compute_tsne_embedding
from src.dataloader import ChromosomeDataLoader
from src.metrics import jaccard, kl_divergence_symmetric


DATA_DIR = "data/T2T"
OUTPUT_DIR = "outputs/analysis/clusterings"
CHROMOSOME_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

loader = ChromosomeDataLoader(data_dir=DATA_DIR)

METRICS = {
    "KL_Symmetric": lambda p, q: kl_divergence_symmetric(p, q),
    "Jaccard": lambda p, q: 1.0 - jaccard(p, q),
    "Raw_Euclidean": None,
}


def _to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def generate_interactive_tsne(metric_name="KL_Symmetric", filename=None):
    if filename is None:
        filename = f"{OUTPUT_DIR}/interactive_tsne_{metric_name.lower()}.html"
    print(f"\n--- Generating Interactive t-SNE Plot [{metric_name}] ---")

    X, _, metadata = loader.load_data(as_distribution=True, max_val=5000, return_tensors=True)
    X = torch.as_tensor(X)
    df_meta = pd.DataFrame(metadata, columns=["individual", "haplotype", "chromosome"])
    df_meta["chromosome"] = pd.Categorical(df_meta["chromosome"], categories=CHROMOSOME_ORDER, ordered=True)

    metric_func = METRICS[metric_name]
    if metric_func is not None:
        print("  Computing distance matrix...")
        dist_matrix = compute_pairwise_distance_matrix(X, metric_func)
        coords = compute_tsne_embedding(dist_matrix, metric="precomputed", random_state=8)
    else:
        print("  Running t-SNE with euclidean metric...")
        coords = compute_tsne_embedding(X, metric="euclidean", random_state=8)

    df_meta["TSNE1"] = coords[:, 0]
    df_meta["TSNE2"] = coords[:, 1]

    color_sequence = [_to_hex(c) for c in cc.glasbey_category10[:24]]
    chrom_color = {chrom: color_sequence[i] for i, chrom in enumerate(CHROMOSOME_ORDER)}

    individuals = sorted(df_meta["individual"].unique())
    fig = go.Figure()
    trace_individuals = []
    first_per_chrom = set()

    for ind in individuals:
        for chrom in CHROMOSOME_ORDER:
            mask = (df_meta["individual"] == ind) & (df_meta["chromosome"].astype(str) == chrom)
            if not mask.any():
                continue
            subset = df_meta[mask]
            show_legend = chrom not in first_per_chrom
            if show_legend:
                first_per_chrom.add(chrom)

            is_chm13 = "chm13" in ind.lower()
            if is_chm13:
                marker = {
                    "symbol": "diamond",
                    "color": chrom_color[chrom],
                    "size": 15,
                    "opacity": 0.8,
                    "line": {"color": chrom_color[chrom], "width": 1},
                }
            else:
                marker = {
                    "symbol": "circle",
                    "color": chrom_color[chrom],
                    "size": 13,
                    "opacity": 0.4,
                    "line": {"color": "white", "width": 1},
                }

            fig.add_trace(
                go.Scatter(
                    x=subset["TSNE1"].values,
                    y=subset["TSNE2"].values,
                    mode="markers",
                    name=chrom,
                    legendgroup=chrom,
                    showlegend=show_legend,
                    marker=marker,
                    customdata=subset[["individual", "haplotype"]].values,
                    hovertemplate="<b>%{text}</b><br>Individual: %{customdata[0]}<br>Haplotype: %{customdata[1]}<extra></extra>",
                    text=subset["chromosome"].astype(str).values,
                )
            )
            trace_individuals.append(ind)

    centroids = df_meta.groupby("chromosome", observed=True)[["TSNE1", "TSNE2"]].median()
    y_offset = (df_meta["TSNE2"].max() - df_meta["TSNE2"].min()) * 0.018

    fig.add_trace(
        go.Scatter(
            x=centroids["TSNE1"],
            y=centroids["TSNE2"] - y_offset,
            mode="text",
            text=centroids.index.astype(str).tolist(),
            textposition="bottom center",
            textfont={"size": 13, "color": "black", "family": "Arial Black", "weight": "bold"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    trace_individuals.append(None)

    all_visible = [True] * len(trace_individuals)
    genome_buttons = [{"label": "All Genomes", "method": "restyle", "args": [{"visible": all_visible}]}]
    for ind in individuals:
        visible = [(t == ind or t is None) for t in trace_individuals]
        genome_buttons.append({"label": ind, "method": "restyle", "args": [{"visible": visible}]})

    fig.update_layout(
        title=f"t-SNE of Genomic Centeny Maps [{metric_name}]",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_white",
        width=1100,
        height=800,
        legend_title_text="Chromosomes (Click to filter)",
        hoverlabel={"bgcolor": "white", "font_size": 12},
        updatemenus=[
            {
                "buttons": genome_buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.18,
                "xanchor": "left",
                "y": 1.0,
                "yanchor": "top",
                "bgcolor": "white",
                "bordercolor": "gray",
                "font": {"size": 11},
            }
        ],
        annotations=[
            {
                "text": "<b>Filter Genome:</b>",
                "x": 1.18,
                "xref": "paper",
                "y": 1.04,
                "yref": "paper",
                "showarrow": False,
                "align": "left",
                "font": {"size": 12},
            }
        ],
    )

    fig.write_html(
        filename,
        config={
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "tsne_centeny",
                "scale": 4,
            }
        },
    )
    print(f"Interactive plot saved to {filename}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for metric_name in METRICS:
        generate_interactive_tsne(metric_name=metric_name)
