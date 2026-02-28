import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scores import *
import os


current_device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_chromosomes(genome, plot_title=None, metric=None, device=current_device):
    """
    Compares all chromosomes of a given genome against each other
    Outputs 22x22 matrix where entry[i,j] is the score of chromosome i of genome computed with respect to chromosome j

    Args:   
        genome: a dictionary with tensors belonging only to a single haplotype
                assumes you have filtered by haplotype before passing it to the function
    """

    # genome is a dictionary {(CHM13, hap1, chr2): torch.tensor[123, 231, ...]}
    chromosomes = [key for key in genome.keys() if ("X" not in key[2]) and ("Y" not in key[2])]
    chromosomes.sort(key=lambda x: (len(x[2]), x[2]))

    if not metric:
        raise ValueError("NO METRIC DEFINED")
    
    # n is the number of chromosomes without X or Y
    n = len(chromosomes)

    assert n==22

    # initialize matrix with zeros to begin with
    matrix = np.zeros((n, n))
    
    # assume you have a list of 22 tensors (one for each chromosome)
    l = [genome[key].to(device) for key in chromosomes]

    # Compute pairwise distances
    for i in range(n):
        arr1_orig = l[i]  # Get the original tensor
        len1 = len(arr1_orig)
        
        for j in range(i, n):
            arr2_orig = l[j] # Get the original tensor
            len2 = len(arr2_orig)

            max_len = max(len1, len2)
            
            # Create new, padded variables inside the loop
            arr1_pad = arr1_orig if len1 == max_len else \
                torch.cat([arr1_orig, torch.zeros(max_len - len1, device=device, dtype=arr1_orig.dtype)])
            
            arr2_pad = arr2_orig if len2 == max_len else \
                torch.cat([arr2_orig, torch.zeros(max_len - len2, device=device, dtype=arr2_orig.dtype)])

            matrix[i, j] = metric(arr1_pad, arr2_pad)
            
            # If the metric is asymmetric (like KL), we must compute the other direction
            # Note: We must re-pad if lengths are different, as arr2_pad might not
            # be the same as arr1_orig and vice-versa.
            # But since we pad both to max_len, arr1_pad and arr2_pad are already
            # the same length, so we can just swap them.
            matrix[j, i] = metric(arr2_pad, arr1_pad)
    

    labels = [f"{k[0]}-{k[1]}-{k[2]}" for k in chromosomes]

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="Blues",
                square=True, annot=True, fmt=".3f")
    plt.title(plot_title if plot_title else "")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return matrix






def plot_chromosomes_across_genomes(genome_list, plot_title=None, metric=None, device=current_device, save=False, save_path=None, save_title=None):
    """
    Given a list of genomes, it scores a single chromosome among those genomes
    Assumes genome_list has each genome already filtered, meaning you have fixed the chromosome and hap

    Args:
        genome_list: list of genomes where for each genome chromosome and hap you have filtered the chromosome beforehand
        and the haplotype as well
        plot_title: The title to display at the top of the heatmap
        save (bool, optional): If `True`, the generated heatmap will be saved to disk. Defaults to `False`.
        save_path (str, optional): The directory path where the plot should be saved. Required if `save=True`.
        save_title (str, optional): The filename for the saved plot (e.g., 'comparison_plot.png'). Required if `save=True`

    """


    if metric is None:
        raise ValueError("NO METRIC DEFINED")
    
    n = len(genome_list)
    matrix = np.zeros((n, n))
    
    # collect chromosome tensors from each genome
    chrom_tensors = []
    labels = []

    for genome in genome_list:
        assert len(genome) == 1, f"Expected exactly 1 chromosome in genome dict, found {len(genome)}"
        for k,v in genome.items():
            chrom_tensors.append(v.to(device))
            labels.append(f"{k[0]}-{k[1]}-{k[2]}")
    
    
    # Compute pairwise distances
    for i in range(n):
        arr1 = chrom_tensors[i]
        len1 = len(arr1)
        for j in range(i, n):
            arr2 = chrom_tensors[j]
            len2 = len(arr2)

            # pad to same length
            max_len = max(len1, len2)
            arr1_pad = arr1 if len1 == max_len else torch.cat([arr1, torch.zeros(max_len - len1, device=device, dtype=arr1.dtype)])
            arr2_pad = arr2 if len2 == max_len else torch.cat([arr2, torch.zeros(max_len - len2, device=device, dtype=arr2.dtype)])

            matrix[i, j] = metric(arr1_pad, arr2_pad)
            matrix[j, i] = metric(arr2_pad, arr1_pad)  # not always symmetric
    

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="Blues",
                square=True, annot=True, fmt=".2f")
    plt.title(plot_title if plot_title else "")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        if save_path and save_title:
            # Create the directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Ensure the title has a file extension, default to .png
            if not any(save_title.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']):
                save_title += ".png"
                
            full_path = os.path.join(save_path, save_title)
            
            # Save the plot
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved successfully to: {full_path}")
        else:
            print("Warning: Both 'save_path' and 'save_title' must be provided to save the plot.")

    plt.show()

    return matrix




def plot_chromosomes_two_genomes(genome1, genome2, plot_title=None, metric=None, device=current_device, save=False, save_path=None, save_title=None):

    """
    Compares all chromosomes of a given genome against chromosomes of another chromosome
    Outputs 22x22 matrix where entry[i,j] is the score of chromosome i of genome 1
    computed with respect to chromosome j of genome 2

    In case of KL, keep in mind that genome 2 is assumed to be the reference genome (p)

    Assumes both genomes are already filtered out, hap2 is differentiated from hap1 

    Args:   
        genome1: a dictionary with tensors belonging only to a single haplotype
                assumes you have filtered by haplotype before passing it to the function
        genome2: the true genome
        plot_title: The title to display at the top of the heatmap
        save (bool, optional): If `True`, the generated heatmap will be saved to disk. Defaults to `False`.
        save_path (str, optional): The directory path where the plot should be saved. Required if `save=True`.
        save_title (str, optional): The filename for the saved plot (e.g., 'comparison_plot.png'). Required if `save=True`
    
    Returns:
        numpy.ndarray: A 22x22 matrix containing the pairwise metric scores between the chromosomes 
                        of genome2 (rows) and genome1 (columns).

    """

    # genome2 is the true distribution!!
    
    if not metric:
        raise ValueError("NO METRIC DEFINED")
    
    # Extract chromosomes (excluding X and Y) from both genomes
    chromosomes1 = [key for key in genome1.keys() if ("X" not in key[2]) and ("Y" not in key[2])]
    chromosomes1.sort(key=lambda x: (len(x[2]), x[2]))
    
    # chromosomes of the true dist
    chromosomes2 = [key for key in genome2.keys() if ("X" not in key[2]) and ("Y" not in key[2])]
    chromosomes2.sort(key=lambda x: (len(x[2]), x[2]))
    
    n1 = len(chromosomes1)
    n2 = len(chromosomes2)

    for i in range(n1):
       chr1_name = chromosomes1[i][2]  # e.g., "chr1"
       chr2_name = chromosomes2[i][2]  # e.g., "chr1"
       assert chr1_name == chr2_name, f"Chromosome mismatch at position {i}: {chr1_name} vs {chr2_name}"
    
    assert n1 == 22, f"Genome 1 should have 22 chromosomes, found {n1}"
    assert n2 == 22, f"Genome 2 should have 22 chromosomes, found {n2}"
    
    # initialize matrix (rows = genome2, cols = genome1)
    matrix = np.zeros((n2, n1))
    
    # Get tensors for both genomes
    tensors1 = [genome1[key].to(device) for key in chromosomes1]
    tensors2 = [genome2[key].to(device) for key in chromosomes2] # tensors of true dist
    
    # Compute cross-genome pairwise distances
    for i in range(n2):  # genome2 chromosomes (placed in y-axis)
        arr2 = tensors2[i] # tensor of true dist
        len2 = len(arr2)
        
        for j in range(n1):  # genome1 chromosomes (placed in x-axis)
            arr1 = tensors1[j]
            len1 = len(arr1)
            
            # Pad to same length
            max_len = max(len1, len2)
            
            arr1_padded = arr1
            arr2_padded = arr2
            
            if len1 < max_len:
                arr1_padded = torch.cat([arr1, torch.zeros(max_len - len1, device=device, dtype=arr1.dtype)])
            
            if len2 < max_len:
                arr2_padded = torch.cat([arr2, torch.zeros(max_len - len2, device=device, dtype=arr2.dtype)])
            
            # Compute metric between genome2[i] true dist and genome1[j] (q dist)
            matrix[i, j] = metric(arr2_padded, arr1_padded)
    

    
    labels1 = [f"{k[0]}-{k[1]}-{k[2]}" for k in chromosomes2]  # Genome 2 (Rows -> Y-axis)
    labels2 = [f"{k[0]}-{k[1]}-{k[2]}" for k in chromosomes1]  # Genome 1 (Cols -> X-axis)

    # CORRECT: Maps Genome 1 labels (labels2) to X-axis
    #          Maps Genome 2 labels (labels1) to Y-axis


    # Plot heatmap
    plt.figure(figsize=(12, 10))
    
    # 2. Use the CORRECT heatmap call
    sns.heatmap(matrix, xticklabels=labels2, yticklabels=labels1, cmap="Blues",
                square=True, annot=True, fmt=".3f", annot_kws={'fontsize': 6})

    # 3. Add title and labels to THIS plot
    title = plot_title if plot_title else "Cross-Genome Chromosome Comparison"
    plt.title(title)
    plt.xlabel("Genome 1 Chromosomes (Approx, q)")
    plt.ylabel("Genome 2 Chromosomes (True, p)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        if save_path and save_title:
            # Create the directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Ensure the title has a file extension, default to .png
            if not any(save_title.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']):
                save_title += ".png"
                
            full_path = os.path.join(save_path, save_title)
            
            # Save the plot
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved successfully to: {full_path}")
        else:
            print("Warning: Both 'save_path' and 'save_title' must be provided to save the plot.")

    plt.show()

    return matrix