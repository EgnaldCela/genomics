import os

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataloader import ChromosomeDataLoader
from src.metrics import kl_divergence_symmetric


def apply_contamination(sequence, contaminant_sequence, percentage):
    """
    Mix a percentage of contaminant sequence into the original sequence.
    percentage is in [0, 1], where 0.1 means +10% contaminant length.
    """
    if percentage == 0:
        return sequence

    target_len = int(len(sequence) * percentage)

    if len(contaminant_sequence) < target_len:
        repeats = (target_len // len(contaminant_sequence)) + 1
        contaminant_pool = torch.cat([contaminant_sequence] * repeats)
    else:
        contaminant_pool = contaminant_sequence

    start_idx = np.random.randint(0, len(contaminant_pool) - target_len + 1)
    contaminant_chunk = contaminant_pool[start_idx : start_idx + target_len]
    mixed_sequence = torch.cat([sequence, contaminant_chunk])
    return mixed_sequence


def load_all_chromosomes(data_dir, individual, chr_order):
    """
    Load chromosomes for an individual into {chrom: sequence_tensor}.
    Prefers hap1, then hap2, then any available haplotype.
    """
    path = os.path.join(data_dir, f"{individual}.pt")
    raw = torch.load(path)

    by_chrom = {}
    for (_, hap, chrom), v in raw.items():
        if chrom not in chr_order:
            continue
        by_chrom.setdefault(chrom, []).append((hap, v))

    chrom_seqs = {}
    for chrom, entries in by_chrom.items():
        hap_order = ["hap1", "hap2", ""]
        chosen = None
        for preferred in hap_order:
            match = [v for (h, v) in entries if h == preferred]
            if match:
                chosen = match[0]
                break
        if chosen is None:
            chosen = entries[0][1]
        chrom_seqs[chrom] = chosen

    return chrom_seqs


def run_for_individual(individual, data_dir, loader, chr_order, contamination_levels, n_sims, palette, output_dir, chm13_seqs):
    print(f"\n=== {individual} ===")
    chrom_seqs = load_all_chromosomes(data_dir, individual, chr_order)
    available = [c for c in chr_order if c in chrom_seqs]
    if len(available) < 2:
        print(f"  Skipping {individual}: fewer than 2 chromosomes available.")
        return
    print(f"  Loaded {len(available)} chromosomes.")

    all_results = []
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, target_chrom in enumerate(available):
        seq_orig = chrom_seqs[target_chrom]

        if target_chrom not in chm13_seqs:
            print(f"  Skipping {target_chrom}: not in CHM13.")
            continue
        h_ref = loader.sequence_to_histogram(chm13_seqs[target_chrom], max_val=5000)

        contaminant_chroms = [c for c in available if c != target_chrom]
        print(f"  {target_chrom} (N contaminants = {len(contaminant_chroms)})...")

        metric_values = []
        for p in contamination_levels:
            scores = []
            for cont_chrom in contaminant_chroms:
                seq_cont = chrom_seqs[cont_chrom]
                for _ in range(n_sims):
                    seq_cont_mixed = apply_contamination(seq_orig, seq_cont, p)
                    h_cont_mixed = loader.sequence_to_histogram(seq_cont_mixed, max_val=5000)
                    score = kl_divergence_symmetric(h_ref, h_cont_mixed).item()
                    scores.append(score)
            metric_values.append(np.mean(scores))

        color = palette[idx % len(palette)]
        ax.plot(contamination_levels * 100, metric_values, color=color, alpha=0.7, label=target_chrom, linewidth=1.5)
        all_results.append({"chromosome": target_chrom, "scores": metric_values})

    if all_results:
        avg_scores = np.mean([r["scores"] for r in all_results], axis=0)
        ax.plot(contamination_levels * 100, avg_scores, color="black", linewidth=3, linestyle="--", label="Average")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Contamination added (% of original chromosome length)", fontsize=12)
    ax.set_ylabel("KL", fontsize=12)
    plt.xticks(
        ticks=[0.1, 0.5, 1, 5, 10, 50, 100, 300],
        labels=["0.1%", "0.5%", "1%", "5%", "10%", "50%", "100%", "300%"],
        fontsize=10,
    )
    ax.set_title(individual, fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"contamination_{individual}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def main():
    data_dir = "data/T2T"
    loader = ChromosomeDataLoader(data_dir=data_dir)

    chr_order = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    palette = cc.glasbey_category10[: len(chr_order)]

    print("Loading CHM13 reference chromosomes...")
    chm13_seqs = load_all_chromosomes(data_dir, "chm13", chr_order)

    individuals = [f.stem for f in sorted(loader.data_dir.glob("*.pt"))]
    print(f"Individuals found: {individuals}")

    contamination_levels = np.geomspace(1e-3, 3.0, 25)
    n_sims = 3

    output_dir = "outputs/analysis/robustness"
    os.makedirs(output_dir, exist_ok=True)

    for individual in individuals:
        run_for_individual(
            individual,
            data_dir,
            loader,
            chr_order,
            contamination_levels,
            n_sims,
            palette,
            output_dir,
            chm13_seqs,
        )


if __name__ == "__main__":
    print("Starting robustness contamination analysis...")
    main()
