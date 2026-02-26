import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Data from previous evaluation
    data = {
        'Metric': ['Jaccard', 'KL_Symmetric', 'KL_Non_Symmetric', 'Raw_Euclidean'],
        'Silhouette': [0.746414, 0.789319, 0.777224, 0.698615],
        '1-NN Acc': [0.980435, 0.995652, 0.993478, 0.982609],
        'NMI': [0.979031, 0.981783, 0.977794, 0.954227]
    }
    
    df = pd.DataFrame(data)
    
    # Melt the dataframe for easier plotting with seaborn
    df_melted = df.melt(id_vars='Metric', var_name='Evaluation Metric', value_name='Score')
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 20
    
    ax = sns.barplot(data=df_melted, x='Evaluation Metric', y='Score', hue='Metric', palette='muted')
    
    # plt.title('Distance Metric Comparison by Evaluation Score', fontsize=18)
    plt.ylabel('Score (Higher is Better)', fontsize=16)
    plt.xlabel('', fontsize=16)
    plt.ylim(0.6, 1.05) # Focus on the differences
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                           fontsize=13)
    
    ax.tick_params(labelsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    
    output_file = "clustering_evaluation_plot.png"
    plt.savefig(output_file, dpi=300)
    print(f"Evaluation plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
