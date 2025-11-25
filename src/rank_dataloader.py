import os
import torch
import numpy as np
from pathlib import Path

from .dataset import ChromosomeDataLoader


class ChromosomeRankingDataLoader(ChromosomeDataLoader):
    """
    Data loader for training ranking models (e.g., LGBMRanker).
    Returns pairs of (original, corrupted) distributions where original is higher quality.
    
    Usage:
        loader = ChromosomeRankingDataLoader(data_dir="/path/to/T2T")
        X, y, groups = loader.load_ranking_data(
            noise_type="gaussian",
            noise_level=0.1,
            individuals=["HG03521"]
        )
        
        # Train LGBMRanker
        from lightgbm import LGBMRanker
        ranker = LGBMRanker()
        ranker.fit(X, y, group=groups)
    """
    
    def add_noise(self, distribution, noise_type="gaussian", noise_level=0.1, seed=None):
        """
        Add noise to a probability distribution.
        
        Args:
            distribution: Normalized histogram (torch tensor)
            noise_type: Type of noise - "gaussian", "uniform", "dropout", or "shuffle"
            noise_level: Amount of noise (0 to 1)
            seed: Random seed for reproducibility
        
        Returns:
            Corrupted distribution (re-normalized to sum to 1)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        dist = distribution.clone()
        
        if noise_type == "gaussian":
            # Add Gaussian noise and renormalize
            noise = torch.randn_like(dist) * noise_level
            dist = dist + noise
            dist = torch.clamp(dist, min=0)  # Ensure non-negative
            
        elif noise_type == "uniform":
            # Mix with uniform distribution
            uniform = torch.ones_like(dist) / len(dist)
            dist = (1 - noise_level) * dist + noise_level * uniform
            
        elif noise_type == "dropout":
            # Randomly zero out bins
            mask = torch.rand_like(dist) > noise_level
            dist = dist * mask
            
        elif noise_type == "shuffle":
            # Randomly shuffle a fraction of values
            n_shuffle = int(len(dist) * noise_level)
            indices = torch.randperm(len(dist))[:n_shuffle]
            shuffled_values = dist[indices][torch.randperm(n_shuffle)]
            dist[indices] = shuffled_values
            
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
        
        # Re-normalize to probability distribution
        dist_sum = dist.sum()
        if dist_sum > 0:
            dist = dist / dist_sum
        else:
            # If all zeros, return uniform distribution
            dist = torch.ones_like(dist) / len(dist)
        
        return dist
    
    def load_ranking_data(self, individuals=None, chromosomes=None, haplotypes=None,
                         max_val=None, noise_type="gaussian", noise_level=0.1,
                         return_tensors=False, seed=None):
        """
        Load data formatted for LGBMRanker training.
        
        Each sample generates a pair: (original, corrupted), where original has higher quality.
        
        Args:
            individuals: List of individual IDs. If None, loads all.
            chromosomes: List of chromosomes. If None, loads all.
            haplotypes: List of haplotypes. If None, loads all.
            max_val: Maximum value for histogram bins.
            noise_type: Type of noise - "gaussian", "uniform", "dropout", or "shuffle"
            noise_level: Amount of noise (0 to 1)
            return_tensors: If True, return torch tensors; otherwise numpy arrays.
            seed: Random seed for reproducibility
        
        Returns:
            X: Array of shape (2*n_samples, n_features) - concatenated [original, corrupted]
            y: Array of shape (2*n_samples,) - relevance labels [1, 0, 1, 0, ...]
                Original distributions get label=1, corrupted get label=0
            groups: Array of group sizes for LGBMRanker - [2, 2, 2, ...] 
                Each group contains one original and one corrupted distribution
            metadata: List of tuples (individual, haplotype, chromosome, "original"/"corrupted")
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        X_list = []
        y_list = []
        groups_list = []
        metadata = []
        
        # Load original distributions
        X_orig, _, meta_orig = self.load_data(
            individuals=individuals,
            chromosomes=chromosomes,
            haplotypes=haplotypes,
            as_distribution=True,
            max_val=max_val,
            return_tensors=True
        )
        
        # For each original distribution, create a corrupted version
        for i, orig_dist in enumerate(X_orig):
            corrupted_dist = self.add_noise(orig_dist, noise_type, noise_level, seed)
            
            # Add pair: (original=1, corrupted=0)
            X_list.append(orig_dist)
            X_list.append(corrupted_dist)
            
            y_list.append(1)  # Original is higher quality
            y_list.append(0)  # Corrupted is lower quality
            
            groups_list.append(2)  # Each group has 2 items
            
            # Metadata
            orig_meta = meta_orig[i]
            metadata.append((*orig_meta, "original"))
            metadata.append((*orig_meta, "corrupted"))
        
        # Convert to arrays
        if return_tensors:
            X = torch.stack(X_list)
            y = torch.tensor(y_list, dtype=torch.long)
            groups = torch.tensor(groups_list, dtype=torch.long)
        else:
            X = np.array([x.numpy() for x in X_list])
            y = np.array(y_list, dtype=np.int64)
            groups = np.array(groups_list, dtype=np.int64)
        
        return X, y, groups, metadata

# Example usage
if __name__ == "__main__":
    ranking_loader = ChromosomeRankingDataLoader(data_dir="/media/pinas/egnald/genomics/data/T2T")
    
    X_rank, y_rank, groups, meta_rank = ranking_loader.load_ranking_data(
        individuals=["HG03521"],
        chromosomes=["chr1", "chr2"],
        haplotypes=["hap1"],
        noise_type="gaussian",
        noise_level=0.1,
        max_val=5000,
        seed=42
    )
    
    print(f"X shape: {X_rank.shape}")
    print(f"y shape (relevance labels): {y_rank.shape}")
    print(f"groups shape: {groups.shape}")
    print(f"Number of ranking pairs: {len(groups)}")
    print(f"Labels (1=original, 0=corrupted): {y_rank[:6]}")
    print(f"Groups (each pair is a group): {groups[:3]}")
    print(f"First 4 metadata entries:")
    for i in range(4):
        print(f"  {meta_rank[i]}")
    
    # Demo: How to use with LGBMRanker
    print("\n" + "="*60)
    print("Example 5: Train LGBMRanker (requires lightgbm)")
    print("="*60)
    try:
        from lightgbm import LGBMRanker
        
        ranker = LGBMRanker(n_estimators=100, learning_rate=0.1)
        ranker.fit(X_rank, y_rank, group=groups)
        print("✓ LGBMRanker trained successfully!")
        
        # Predict on same data (just for demo)
        predictions = ranker.predict(X_rank)
        print(f"Predictions shape: {predictions.shape}")
        print(f"First 6 predictions: {predictions[:6]}")
        print(f"  (Higher scores = higher quality)")
        
    except ImportError:
        print("lightgbm not installed. Install with: pip install lightgbm")

