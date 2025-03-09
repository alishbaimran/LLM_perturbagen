# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import ast

drug_df = pd.read_csv("drug_embeddings.csv")
gene_df = pd.read_csv("gene_embeddings.csv")

drug_df["drug_embedding"] = drug_df["drug_embedding"].apply(ast.literal_eval)
gene_df["gene_embedding"] = gene_df["gene_embedding"].apply(ast.literal_eval)

drug_embeddings = np.vstack(drug_df["drug_embedding"].values)
gene_embeddings = np.vstack(gene_df["gene_embedding"].values)

all_embeddings = np.vstack((drug_embeddings, gene_embeddings))

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(all_embeddings)

labels = ["drug"] * len(drug_embeddings) + ["gene"] * len(gene_embeddings)
pca_df = pd.DataFrame(reduced_embeddings, columns=["PC1", "PC2"])
pca_df["Type"] = labels

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Type", alpha=0.7, edgecolor="k")
plt.title("PCA Projection of Drug and Gene Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Type")
plt.show()
