import pandas as pd
import matplotlib.pyplot as plt

csv_path = "exp1_metrics_genlen2500_temp05.csv"
df = pd.read_csv(csv_path)
df['chromosome'] = df['chromosome'].astype(str)

plt.figure(figsize=(12, 6))

bars = plt.bar(df['chromosome'], df['similarity'], color="skyblue", edgecolor="black")

plt.scatter(df['chromosome'], df['similarity'], color="darkblue", zorder=3)

plt.title("Similarity (%) by Chromosome", fontsize=14, fontweight="bold")
plt.xlabel("Chromosome")
plt.ylabel("Similarity (%)")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.text(0.02, 0.95, "Temp = 0.5", transform=plt.gca().transAxes,
         fontsize=12, color="darkred", 
         bbox=dict(facecolor="white", alpha=0.7, edgecolor="darkred"))

plt.tight_layout()
plt.savefig("similarity_barplot_temp05.png", dpi=300)
plt.show()
