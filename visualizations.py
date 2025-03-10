# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import ast

drug_df = pd.read_csv("drug_embeddings_with_concentration.csv")
gene_df = pd.read_csv("gene_embeddings.csv")

drug_df["drug_embedding"] = drug_df["drug_embedding"].apply(ast.literal_eval)
gene_df["gene_embedding"] = gene_df["gene_embedding"].apply(ast.literal_eval)

drug_embeddings = np.vstack(drug_df["drug_embedding"].values)
gene_embeddings = np.vstack(gene_df["gene_embedding"].values)

all_embeddings = np.vstack((drug_embeddings, gene_embeddings))
all_labels = list(drug_df["drug_name"]) + list(gene_df["gene_name"])

drug_names = list(drug_df["drug_name"])
gene_names = list(gene_df["gene_name"])

# %% PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(all_embeddings)

labels = ["drug"] * len(drug_embeddings) + ["gene"] * len(gene_embeddings)
pca_df = pd.DataFrame(reduced_embeddings, columns=["PC1", "PC2"])
pca_df["Type"] = labels

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Type", alpha=0.7, edgecolor="k")
plt.title("PCA of Drug and Gene Embeddings (concentration included)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(title="Type")
plt.show()

# %% pairwise cosine similarity for top 20 gene-drug pairs
cosine_sim_matrix = cosine_similarity(drug_embeddings, gene_embeddings)

cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=drug_names, columns=gene_names)

top_n = 20  

cosine_sim_long = cosine_sim_df.unstack().reset_index()
cosine_sim_long.columns = ["Gene", "Drug", "Similarity"]
top_pairs = cosine_sim_long.sort_values(by="Similarity", ascending=False).head(top_n)

print("\nTop Drug-Gene Pairs by Similarity:")
print(top_pairs)

plt.figure(figsize=(12, 6))

pivot_df = top_pairs.pivot(index="Drug", columns="Gene", values="Similarity")

sns.heatmap(pivot_df, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
plt.title(f"Top {top_n} Drug-Gene Similarities")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.show()
# %%
