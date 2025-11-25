""" Scikit-learn compatible data loader for chromosome sequences.
Loads preprocessed .pt files containing chromosome distance tensors.
Can return raw sequences or normalized histograms (distributions).
"""
import os
import torch
import numpy as np
from pathlib import Path


class ChromosomeDataLoader:
    """
    SKLearn-compatible data loader for chromosome sequences.
    
    Usage:
        loader = ChromosomeDataLoader(data_dir="/path/to/T2T")
        
        # Load raw sequences
        X, y, metadata = loader.load_data(
            individuals=["HG03521", "HG01975"],
            chromosomes=["chr1", "chr2"],
            haplotypes=["hap1", "hap2"]
        )
        
        # Load as distributions (histograms)
        X, y, metadata = loader.load_data(
            individuals=["HG03521"],
            as_distribution=True,
            max_val=5000
        )
    """
    
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Path to directory containing .pt files (e.g., HG03521.pt)
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        self.available_files = list(self.data_dir.glob("*.pt"))
        self.available_individuals = [f.stem for f in self.available_files]
    
    @staticmethod
    def sequence_to_histogram(sequence, max_val=None):
        """
        Convert a sequence tensor to a normalized histogram (distribution).
        Based on dict_to_hist from metrics.py
        
        Args:
            sequence: Torch tensor of sequence distances
            max_val: Maximum value for histogram bins. If None, uses max of sequence.
        
        Returns:
            Normalized histogram as torch tensor (probability distribution)
        """
        sequence = torch.as_tensor(sequence, dtype=torch.int64)
        
        # Compute max_val if not provided
        local_max = int(torch.max(sequence).item()) if max_val is None else int(max_val)
        
        # Build histogram
        bins = local_max + 1
        hist = torch.bincount(sequence, minlength=bins).float()
        
        # Normalize to probability distribution
        hist = hist / hist.sum()
        
        return hist
        
    def load_data(self, individuals=None, chromosomes=None, haplotypes=None, 
                  pad_length=None, return_tensors=False, as_distribution=False, max_val=None):
        """
        Load chromosome sequences with optional filtering.
        
        Args:
            individuals: List of individual IDs (e.g., ["HG03521"]). If None, loads all.
            chromosomes: List of chromosomes (e.g., ["chr1", "chr21"]). If None, loads all.
            haplotypes: List of haplotypes (e.g., ["hap1", "hap2"]). If None, loads all.
            pad_length: If specified, pad/truncate sequences to this length (only for raw sequences).
            return_tensors: If True, return torch tensors; otherwise numpy arrays.
            as_distribution: If True, convert sequences to normalized histograms.
            max_val: Maximum value for histogram bins (only used if as_distribution=True).
        
        Returns:
            X: Array/tensor of sequences or distributions, shape (n_samples, seq_length or n_bins)
            y: Array of integer labels (individual index)
            metadata: List of tuples (individual, haplotype, chromosome) for each sample
        """
        X_list = []
        y_list = []
        metadata = []
        
        # Determine which individuals to load
        if individuals is None:
            individuals = self.available_individuals
        
        # Map individuals to integer labels
        individual_to_label = {ind: i for i, ind in enumerate(individuals)}
        
        for individual in individuals:
            filepath = self.data_dir / f"{individual}.pt"
            if not filepath.exists():
                print(f"Warning: File not found for {individual}, skipping.")
                continue
            
            # Load the dictionary: {(individual, hap, chr): tensor}
            genome_dict = torch.load(filepath, weights_only=True)
            
            for key, sequence in genome_dict.items():
                ind, hap, chrom = key
                
                # Apply filters
                if chromosomes is not None and chrom not in chromosomes:
                    continue
                if haplotypes is not None and hap not in haplotypes:
                    continue
                
                # Process sequence
                seq = sequence.cpu() if isinstance(sequence, torch.Tensor) else torch.tensor(sequence)
                
                # Convert to distribution if requested
                if as_distribution:
                    seq = self.sequence_to_histogram(seq, max_val=max_val)
                else:
                    # Pad or truncate if specified (only for raw sequences)
                    if pad_length is not None:
                        if len(seq) < pad_length:
                            seq = torch.cat([seq, torch.zeros(pad_length - len(seq), dtype=seq.dtype)])
                        else:
                            seq = seq[:pad_length]
                
                X_list.append(seq)
                y_list.append(individual_to_label[individual])
                metadata.append(key)
        
        # Convert to arrays
        if return_tensors:
            X = torch.stack(X_list) if X_list else torch.empty(0)
            y = torch.tensor(y_list, dtype=torch.long)
        else:
            X = np.array([s.numpy() for s in X_list]) if X_list else np.empty((0, 0))
            y = np.array(y_list, dtype=np.int64)
        
        return X, y, metadata
    
    def get_chromosome(self, individual, chromosome, haplotype, as_distribution=False, max_val=None):
        """
        Load a single specific chromosome.
        
        Args:
            individual: Individual ID (e.g., "HG03521")
            chromosome: Chromosome name (e.g., "chr1")
            haplotype: Haplotype (e.g., "hap1")
            as_distribution: If True, return as normalized histogram
            max_val: Maximum value for histogram bins (only used if as_distribution=True)
        
        Returns:
            Tensor of sequence distances or distribution
        """
        filepath = self.data_dir / f"{individual}.pt"
        if not filepath.exists():
            raise ValueError(f"File not found: {filepath}")
        
        genome_dict = torch.load(filepath, weights_only=True)
        key = (individual, haplotype, chromosome)
        
        if key not in genome_dict:
            raise KeyError(f"Key not found: {key}")
        
        sequence = genome_dict[key]
        
        if as_distribution:
            return self.sequence_to_histogram(sequence, max_val=max_val)
        
        return sequence
    
    def info(self):
        """Print summary of available data."""
        print(f"Data directory: {self.data_dir}")
        print(f"Available individuals: {len(self.available_individuals)}")
        print(f"  {', '.join(self.available_individuals[:5])}", end="")
        if len(self.available_individuals) > 5:
            print(f" ... and {len(self.available_individuals) - 5} more")
        else:
            print()
        
        # Sample one file to show structure
        if self.available_files:
            sample_dict = torch.load(self.available_files[0], weights_only=True)
            print(f"\nSample file structure ({self.available_files[0].name}):")
            print(f"  Total sequences: {len(sample_dict)}")
            sample_key = list(sample_dict.keys())[0]
            print(f"  Example key: {sample_key}")
            print(f"  Example sequence length: {len(sample_dict[sample_key])}")


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = ChromosomeDataLoader(data_dir="/media/pinas/egnald/genomics/data/T2T")
    
    # Show available data
    loader.info()
    
    print("\n" + "="*60)
    print("Example 1: Load raw sequences with padding")
    print("="*60)
    X, y, metadata = loader.load_data(
        individuals=["HG03521"],
        chromosomes=["chr1", "chr2"],
        haplotypes=["hap1"],
        pad_length=10000
    )
    print(f"Shape: {X.shape}")
    print(f"First 3 samples metadata: {metadata[:3]}")
    
    print("\n" + "="*60)
    print("Example 2: Load as distributions (histograms)")
    print("="*60)
    X_dist, y_dist, metadata_dist = loader.load_data(
        individuals=["HG03521"],
        chromosomes=["chr1", "chr2"],
        haplotypes=["hap1"],
        as_distribution=True,
        max_val=5000
    )
    print(f"Shape: {X_dist.shape}")
    print(f"Each row sums to 1.0 (normalized): {X_dist[0].sum():.4f}")
    print(f"Distribution has {X_dist.shape[1]} bins")
    
    print("\n" + "="*60)
    print("Example 3: Get single chromosome as distribution")
    print("="*60)
    single_dist = loader.get_chromosome(
        "HG03521", "chr1", "hap1", 
        as_distribution=True, 
        max_val=5000
    )
    print(f"Single chromosome distribution shape: {single_dist.shape}")
    print(f"Sum (should be 1.0): {single_dist.sum():.4f}")
    
