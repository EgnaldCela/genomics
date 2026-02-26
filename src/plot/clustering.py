
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import colorcet as cc

def plot_embedding(coords, labels, title=None, filename=None, show_labels=False, palette=None):
    """
    Standardized plotting function for genome/chromosome embeddings.
    """
    if palette is None:
        palette = cc.glasbey_category10[:24]
        
    plt.figure(figsize=(10, 10))
    
    # Add jitter to prevent overplotting
    coords_jittered = coords + np.random.normal(0, .4, size=coords.shape) 
    
    ax = sns.scatterplot(
        x=coords_jittered[:, 0], y=coords_jittered[:, 1] +1 , 
        hue=labels,
        palette=palette, s=140,
        legend=not show_labels # If we show text labels, we usually don't need the legend
    )
    
    if show_labels:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            if mask.any():
                # Get centroid of jittered points
                centroid = coords_jittered[mask].mean(axis=0)
                plt.text(
                    centroid[0], centroid[1], 
                    str(label), 
                    fontsize=15, 
                    fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
                )
    
    if title:
        plt.title(title, fontsize=18)
        
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")
        
    plt.show()
    plt.close()
