import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.dataloader import ChromosomeDataLoader
from src.metrics import kl_divergence_symmetric
import colorcet as cc

def apply_genomic_noise_simulation(sequence, error_rate_per_bp):
    """
    Full chromosome simulation of genomic noise.
    L' = L + insertions - deletions
    where insertions and deletions follow Poisson(L * error_rate/2).
    """
    if error_rate_per_bp == 0:
        return sequence
    
    L = sequence.float()
    rate = L * (error_rate_per_bp / 2.0)
    
    # Sample Poisson noise
    insertions = torch.poisson(rate)
    deletions = torch.poisson(rate)
    
    noisy_sequence = sequence + insertions.int() - deletions.int()
    noisy_sequence = torch.clamp(noisy_sequence, min=1)
    
    return noisy_sequence

def main():
    data_dir = "data/T2T"
    loader = ChromosomeDataLoader(data_dir=data_dir)
    individuals = loader.available_individuals
    # Exclude chrY as requested
    chr_order = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    palette = cc.glasbey_category10[:24]
    
    # Use chm13 as the reference genome
    ref_individual = "chm13"
    print(f"Loading reference genome: {ref_individual}")
    
    # Error rates range
    error_rates_per_bp = np.geomspace(1e-5, 1e-2, 20)
    error_rates_labels = error_rates_per_bp * 1000 # Errors per 1000 bp
    
    # Number of simulations to average over for smoothing
    n_sims = 10
    
    os.makedirs("plots/robustness", exist_ok=True)
    
    for individual in individuals:
        if individual == ref_individual:
            continue
            
        print(f"Processing robustness simulation vs {ref_individual} for {individual}...")
        results = []
        
        for idx, chrom in enumerate(chr_order):
            try:
                # Load raw sequence from target individual (hap1)
                seq_target = loader.get_chromosome(individual, chrom, "hap1", as_distribution=False)
                
                # Load reference histogram (CHM13 hap1)
                h_ref = loader.get_chromosome(ref_individual, chrom, "hap1", as_distribution=True, max_val=5000)
                
                color = palette[idx % len(palette)]
                chr_results = []
                
                for e in error_rates_per_bp:
                    sim_dists = []
                    for _ in range(n_sims):
                        seq_noisy = apply_genomic_noise_simulation(seq_target, e)
                        h_noisy = loader.sequence_to_histogram(seq_noisy, max_val=5000)
                        dist = kl_divergence_symmetric(h_ref, h_noisy).item()
                        sim_dists.append(max(dist, 1e-15))
                    
                    # Average the simulations for this error rate
                    chr_results.append(np.mean(sim_dists))
                
                results.append({
                    'chromosome': chrom,
                    'distances': chr_results,
                    'color': color
                })
            except (KeyError, ValueError):
                continue
        
        if not results: continue
        
        # Increase figure width for the full legend
        fig, ax = plt.subplots(figsize=(15, 8))
        all_dists = []
        
        for res in results:
            ax.plot(error_rates_labels, res['distances'], color=res['color'], alpha=0.8, label=res['chromosome'], linewidth=2)
            all_dists.append(res['distances'])
            
        avg_dists = np.mean(all_dists, axis=0)
        ax.plot(error_rates_labels, avg_dists, color='black', linewidth=5, label='Trend (Average)', zorder=10)
        
        ax.set_title(f"Metric Robustness vs {ref_individual}: {individual}", fontsize=18, fontweight='bold', pad=30)
        ax.set_xlabel("Genomic Error Rate (errors per 1000 bp)", fontsize=14)
        ax.set_ylabel(f"KL Symmetric Distance to {ref_individual}", fontsize=14)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.3)
        
        # Secondary X-axis for Q-scores
        def forward(x):
            return 30 - 10 * np.log10(x + 1e-25)
        def inverse(x):
            return 10**((30 - x) / 10)
            
        secax = ax.secondary_xaxis('top', functions=(forward, inverse))
        secax.set_xlabel('Phred Quality Score (Q)', fontsize=14, labelpad=10)
        secax.set_ticks([80, 70, 60, 50, 40, 30, 20])
        
        # Legend: Show all chromosomes, placed outside to the right
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, ncol=1, frameon=True, shadow=True)
            
        plt.tight_layout()
        filename = f"plots/robustness/robustness_vs_{ref_individual.replace('.', '_')}_{individual.replace('.', '_')}.png"
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
