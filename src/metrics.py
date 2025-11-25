"""
Defining Jaccard, KL-divergence
"""
import torch

# current_device = "cuda" if torch.cuda.is_available() else "cpu"
current_device = "cpu"

def dict_to_hist(data_dict, max_val=None):
    """
    Convert dictionary { (genomic_code, hap, chromosome): torch tensor }
    into { (genomic_code, hap, chromosome): histogram (torch tensor on GPU/CPU) }.
    
    Each histogram is normalized into a probability distribution.
    """
    results = {}

    for key, arr in data_dict.items():
        # Convert numpy array -> torch tensor on CPU (for compatibility purposes from old code)
        chrom = torch.as_tensor(arr, dtype=torch.int64, device=current_device)

        # Compute max_val if not provided
        local_max = int(torch.max(chrom).item()) if max_val is None else int(max_val)

        # Build histogram
        bins = local_max + 1
        hist = torch.bincount(chrom, minlength=bins).float()

        # Normalize
        hist = hist / hist.sum()

        results[key] = hist

    return results


# compute jaccard distance (not index)
def jaccard(arr1_hist, arr2_hist):
    """ 
    Given two tensors each representing a distribution over values, it returns the jaccard distance.
    """
    # move to CPU or GPU if needed
    arr1_hist = arr1_hist.to("cuda:0")
    arr2_hist = arr2_hist.to("cuda:0")

    # Make both histograms the same length
    max_len = max(arr1_hist.shape[0], arr2_hist.shape[0])
    if arr1_hist.shape[0] < max_len:
        arr1_hist = torch.cat([arr1_hist, torch.zeros(max_len - arr1_hist.shape[0], device=arr1_hist.device)])
    if arr2_hist.shape[0] < max_len:
        arr2_hist = torch.cat([arr2_hist, torch.zeros(max_len - arr2_hist.shape[0], device=arr2_hist.device)])

    # compute jaccard
    intersection = torch.sum(torch.minimum(arr1_hist, arr2_hist))
    union = torch.sum(torch.maximum(arr1_hist, arr2_hist))
    # i changed it
    # jaccard_distance = 1 - intersection/union
    jaccard_distance = intersection / union

    return jaccard_distance


def kl_divergence(p, q, eps=1e-12):
    """
    Compute Kullback-Leibler divergence KL(p || q) between two histograms.

    Args:
        p (torch.Tensor): first histogram assumed as "the true" distribution
        q (torch.Tensor): second histogram, assumed as the approx distribution
        eps (float): small constant to avoid log(0)

    Returns:
        float: KL divergence
    """
    p = p.float()
    q = q.float()
    
    # ensure same length by padding with zeros
    max_len = max(p.shape[0], q.shape[0])
    if p.shape[0] < max_len:
        p = torch.cat([p, torch.zeros(max_len - p.shape[0], dtype=p.dtype, device=p.device)])
    if q.shape[0] < max_len:
        q = torch.cat([q, torch.zeros(max_len - q.shape[0], dtype=q.dtype, device=q.device)])

    # print(p.sum())

    # Compute KL divergence
    KL = torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)))

    return KL


def kl_divergence_symmetric(p, q, eps=1e-12):

    # Convert to float tensors
    p = p.float()
    q = q.float()
    
    # Ensure same length by padding with zeros
    max_len = max(p.shape[0], q.shape[0])
    if p.shape[0] < max_len:
        p = torch.cat([p, torch.zeros(max_len - p.shape[0], dtype=p.dtype, device=p.device)])
    if q.shape[0] < max_len:
        q = torch.cat([q, torch.zeros(max_len - q.shape[0], dtype=q.dtype, device=q.device)])

    # Normalize to probability distributions if not already
    # print(p.sum())

    # Add epsilon for numerical stability
    p = p + eps
    q = q + eps

    kl_pq = torch.sum(p * torch.log(p / q))
    kl_qp = torch.sum(q * torch.log(q / p))
    
    # Symmetric KL is the average of both directions
    symmetric_kl = (kl_pq + kl_qp) / 2

    return symmetric_kl 


def kl_similarity(p, q, eps=1e-12):
    
    value = kl_divergence(p, q, eps=eps) 
    similarity = torch.exp(-value)
    return similarity


def kl_similarity_symmetric(p, q, eps=1e-12):
    
    value = kl_divergence_symmetric(p, q, eps=eps)
    similarity = torch.exp(-value)
    return similarity

