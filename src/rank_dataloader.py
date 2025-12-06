import os
import torch
import numpy as np
from dataloader import ChromosomeDataLoader 

class BioArtifactRankingLoader(ChromosomeDataLoader):
    """
    Advanced ranking loader that simulates realistic assembly quality tiers.
    Generates pairs of (LowNoise, HighNoise) distributions.
    """
    
    # (4) Base probabilities: How likely is this error type to appear at all?
    # Jitter is innate to sequencing, so prob=1.0. Chimera is rare.
    BASE_PROBS = {
        "jitter": 1.0,      
        "merge": 0.4,       # SNPs/Indels in boxes are moderately common
        "collapse": 0.2,    # Repeats collapse often, but not every contig
        "chimera": 0.05     # Structural chimeras are catastrophic but rarer
    }

    def apply_complex_corruption(self, sequence, noise_level=0.1, seed=None):
        """
        Applies a MIX of errors based on a 'noise_level' scalar.
        
        Args:
            sequence: Raw integer tensor.
            noise_level: Float (0.0 to 1.0). 
                         0.0 = Perfect clean data.
                         1.0 = Garbage assembly.
                         Controls BOTH the severity of errors and the likelihood 
                         of rare errors (like Chimera) triggering.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        seq = sequence.clone().float()
        n = len(seq)
        if n < 5: return sequence.long()

        # ---------------------------------------------------------
        # 1. Jitter (Always applied, severity scales with noise_level)
        # ---------------------------------------------------------
        if np.random.rand() < self.BASE_PROBS["jitter"]:
            # At noise_level 0.1 -> 2% variation. At 1.0 -> 20% variation.
            sigma = 0.2 * noise_level 
            noise = torch.randn_like(seq) * (seq * sigma)
            seq = seq + noise
            seq = torch.clamp(seq, min=1)

        # ---------------------------------------------------------
        # 2. Merge (SNP in Box) - Prob scales with noise_level
        # ---------------------------------------------------------
        # Probability of *triggering* this error type increases with noise_level
        # But we clamp it so even bad assemblies don't ALWAYS have this.
        prob_trigger = self.BASE_PROBS["merge"] * (0.5 + 0.5 * noise_level)
        
        if np.random.rand() < prob_trigger:
            # Severity: How many boxes are lost? 
            # Low noise: 1-2%. High noise: up to 20%.
            severity = 0.01 + (0.2 * noise_level)
            
            num_merges = int(n * severity)
            if num_merges > 0:
                seq_list = seq.tolist()
                indices = np.random.choice(len(seq_list)-1, size=num_merges, replace=False)
                indices = np.sort(indices)[::-1]
                
                BOX_LEN = 17
                for idx in indices:
                    if idx + 1 < len(seq_list):
                        new_val = seq_list[idx] + BOX_LEN + seq_list[idx+1]
                        seq_list[idx] = new_val
                        seq_list.pop(idx+1)
                seq = torch.tensor(seq_list)

        # ---------------------------------------------------------
        # 3. Collapse (Truncation)
        # ---------------------------------------------------------
        prob_trigger = self.BASE_PROBS["collapse"] * (0.5 + 0.5 * noise_level)
        
        if np.random.rand() < prob_trigger:
            # At noise_level 1.0, we might lose 50% of the sequence.
            # At noise_level 0.1, we might lose 5%.
            loss_ratio = 0.5 * noise_level
            target_len = int(len(seq) * (1.0 - loss_ratio))
            target_len = max(5, target_len)
            
            if target_len < len(seq):
                max_start = len(seq) - target_len
                start = torch.randint(0, max_start + 1, (1,)).item()
                seq = seq[start : start + target_len]

        # ---------------------------------------------------------
        # 4. Chimera (Structural Jump)
        # ---------------------------------------------------------
        # Only appears in moderate-to-high noise levels
        prob_trigger = self.BASE_PROBS["chimera"] * noise_level
        
        if np.random.rand() < prob_trigger:
            # Type A (Cliff) or Type B (Step)
            if np.random.rand() < 0.5:
                # Cliff
                idx = np.random.randint(0, len(seq))
                seq[idx] = seq[idx] + 100000 # Massive outlier
            else:
                # Step
                split_idx = np.random.randint(int(len(seq)*0.2), int(len(seq)*0.8))
                shift = 500 + (1000 * noise_level)
                seq[split_idx:] += shift

        return seq.long()

    def load_ranking_data(self, individuals=None, chromosomes=None, haplotypes=None,
                         max_val=5000, 
                         n_pairs_per_sample=3,  # (1) Param for N pairs
                         seed=42):
        """
        Generates pairs of (Better, Worse) distributions.
        
        Args:
            n_pairs_per_sample: Number of ranking pairs to generate per original sequence.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        X_list = []
        y_list = []
        groups_list = []
        metadata = []
        
        # Load raw integers
        X_raw, _, meta_raw = self.load_data(
            individuals=individuals, 
            chromosomes=chromosomes, 
            haplotypes=haplotypes,
            as_distribution=False,
            return_tensors=True
        )
        
        print(f"Base sequences: {len(X_raw)}")
        print(f"Generating {n_pairs_per_sample} pairs per sequence...")
        
        for i, raw_seq in enumerate(X_raw):
            if len(raw_seq) < 10: continue
            
            # (1) Loop N times per sample
            for _ in range(n_pairs_per_sample):
                
                # (2) Define "Better" vs "Worse" noise levels
                # "Better" sample: Low noise (0.0 - 0.2)
                # "Worse" sample:  High noise (0.4 - 1.0)
                # Note: Even the "Better" sample has SOME noise (simulating realistic 'good' assembly)
                
                noise_lvl_good = np.random.uniform(0.00, 0.20)
                noise_lvl_bad  = np.random.uniform(0.40, 1.00)
                
                # (3) Apply combinatorial errors
                seq_good = self.apply_complex_corruption(raw_seq, noise_level=noise_lvl_good)
                seq_bad  = self.apply_complex_corruption(raw_seq, noise_level=noise_lvl_bad)
                
                # Convert to histograms
                dist_good = self.sequence_to_histogram(seq_good, max_val=max_val)
                dist_bad  = self.sequence_to_histogram(seq_bad, max_val=max_val)
                
                # Add pair
                X_list.append(dist_good)
                X_list.append(dist_bad)
                
                # Label: 1 means the first item (good) is the relevant one
                y_list.append(1)
                y_list.append(0)
                
                groups_list.append(2)
                
                base_meta = meta_raw[i]
                metadata.append((*base_meta, f"better_lvl{noise_lvl_good:.2f}"))
                metadata.append((*base_meta, f"worse_lvl{noise_lvl_bad:.2f}"))
                
        if len(X_list) > 0:
            X = torch.stack(X_list)
            y = torch.tensor(y_list, dtype=torch.long)
            groups = torch.tensor(groups_list, dtype=torch.long)
        else:
            X = torch.empty(0)
            
        return X, y, groups, metadata
