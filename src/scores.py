"""
This file is used to define "scores" that evaluate the performance of a matrix

Input: A matrix of similarities (Jaccard, KL) of genome vs genome plot
Output: Score (int)


1. Frobenius Norm from identity (minimize)
2. Average of row-wise entropies (minimize)
3. InfoNCE 
4. Trace_ratio (maximize it)
5. Variance (average row-wise)
"""

import numpy as np

def trace_ratio(X):
    return np.trace(X) / (np.sum(np.abs(X)) + 1e-9)

def frobenius_norm(X):

    # creates identity with same shape as X
    I = np.eye(X.shape[0])
    score = np.linalg.norm(X-I)

    return score


def average_row_variance(X):
    """
    Computes the variance for each row and returns the average across all rows.
    """
    # Calculate variance along each row (axis=1)
    row_vars = np.var(X, axis=1)
    
    # Return the average of those row variances
    return np.mean(row_vars)

def entropy(X):

    # Idea: We want the columns to be as close to one hot encoding possible (entropy of zero)
    
    # normalize each column of the matrix

    cumsum_columnwise = np.sum(X, axis=0)

    # to avoid division by zero, we force division by 1 (doesn't change anything)
    cumsum_columnwise[cumsum_columnwise == 0] = 1.0

    # here we normalize all columns
    probs = X / cumsum_columnwise

    # take log of all entries of columns
    logprobs = np.zeros_like(probs)
    mask = probs > 0
    logprobs[mask] = np.log(probs[mask])
    
    element_entropies = -1 * probs * logprobs
    
    col_entropies = np.sum(element_entropies, axis=0)

    return np.mean(col_entropies)


def info_nce(X, temperature=0.1):    
    # identify the "Positives" (The correct matches on the diagonal)
    # We extract the diagonal elements: M[0,0], M[1,1], etc.
    positives = np.diag(X)
    
    # temperature Scaling
    scaled_X = X / temperature
    scaled_positives = positives / temperature
    
    # calculate the Numerator
    numerator = np.exp(scaled_positives)
    
    # calculate the denominator 
    #  sum the exponentials across the row (axis=1)
    # this represents the sum of similarity to ALL chromosomes (correct + incorrect)
    denominator = np.sum(np.exp(scaled_X), axis=1)
    
    nll = -np.log(numerator / (denominator + 1e-9))
    
    return np.mean(nll)


def symmetric_info_nce(X, temperature=0.1):

    loss_rows = info_nce(X, temperature)
    loss_cols = info_nce(X.T, temperature)
    return (loss_rows + loss_cols) / 2


def evaluate_matrix(matrix):

    frobenius_score = frobenius_norm(matrix)
    entropy_score = entropy(matrix)
    info_nce_score = info_nce(matrix)
    variance = average_row_variance(matrix)
    trace_ratio = trace_ratio(matrix)

    print(
        f"Average Entropy (max value log(22)=3.09: {entropy_score:.3f} \n \
          Frobenius Norm: {frobenius_score:.3f} \n  \
          Info-NCE Loss: {info_nce_score:.3f} \
          Variance: {variance:.3f} \
          Trace Ratio: {trace_ratio:.3f}"
          )
    
    return None



