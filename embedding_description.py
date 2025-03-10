# %%
import numpy as np
import pandas as pd
import ast
import os
import time
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


client = OpenAI()

tahoe_drugs = np.load("/processed_datasets/scRecount/tahoe/all_tahoe_perturbations.npy", allow_pickle=True)
drug_targets_df = pd.read_excel("tahoe_drug_smiles.xlsx")
replogle_genes = np.load("/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/all_replogle_perturbations.npy", allow_pickle=True)

try:
    new_genes_df = pd.read_csv("new_target_genes_not_in_replogle.csv")
    new_genes = new_genes_df["target_gene"].tolist()
except FileNotFoundError:
    tahoe_drug_names = [ast.literal_eval(drug_str)[0][0] for drug_str in tahoe_drugs]
    csv_drug_names = drug_targets_df["drug"].astype(str).tolist()
    
    tahoe_drug_names_lower = {drug.lower().strip() for drug in tahoe_drug_names}
    csv_drug_names_lower = {drug.lower().strip(): idx for idx, drug in enumerate(csv_drug_names)}
    
    matching_drugs_lower = tahoe_drug_names_lower & csv_drug_names_lower.keys()
    
    target_genes = []
    for drug_lower in matching_drugs_lower:
        idx = csv_drug_names_lower[drug_lower]
        drug_targets = drug_targets_df.iloc[idx]["targets"]
        if pd.notna(drug_targets) and drug_targets and drug_targets.lower() != "none":
            target_genes.extend([gene.strip() for gene in drug_targets.split(",")])
    
    replogle_genes_lower = {gene.lower().strip() for gene in replogle_genes}
    target_genes_lower = {gene.lower().strip() for gene in target_genes}
    
    new_genes = list(target_genes_lower - replogle_genes_lower)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def generate_drug_description(drug_name, concentration):
    prompt = (f"Describe the mechanism of action and cellular effects of {drug_name} "
              f"when applied at {concentration}. Focus on pathways, target proteins, "
              f"and phenotypic outcomes.")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a biology expert generating drug descriptions."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_gene_description(gene_name):
    prompt = (f"Describe the biological function of {gene_name} and the effects of perturbing it "
              f"in human cells. Focus on pathways, downstream effects, and phenotypic changes.")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a biology expert generating gene function descriptions."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_and_embed_drug_prompts(drug_list):
    """
    Generates LLM-based drug descriptions and computes embeddings.
    """
    descriptions, embeddings = [], []
    
    print(f"Generating descriptions and embeddings for {len(drug_list)} drugs...")

    for i, (drug_name, concentration) in enumerate(tqdm(drug_list)):
        description = generate_drug_description(drug_name, concentration)
        descriptions.append(description)
        embeddings.append(get_embedding(description))

        # Print first 3 descriptions for validation
        if i < 3:
            print(f"Drug {i+1}: {drug_name} ({concentration})")
            print(f"Generated Description: {description}")

        time.sleep(0.1)

    result_df = pd.DataFrame({
        "drug_name": [drug[0] for drug in drug_list],
        "concentration": [drug[1] for drug in drug_list],
        "drug_description": descriptions,
        "drug_embedding": embeddings  
    })
    
    result_df.to_csv("notasync_des_drug_embeddings_with_concentration.csv", index=False)
    print(f" Saved {len(drug_list)} drug embeddings to des_drug_embeddings_with_concentration.csv")
    return result_df


def generate_and_embed_gene_prompts(gene_list):
    """
    Generates LLM-based gene descriptions and computes embeddings.
    """
    descriptions, embeddings = [], []
    
    print(f"Generating descriptions and embeddings for {len(gene_list)} genes...")

    for i, gene in enumerate(tqdm(gene_list)):
        description = generate_gene_description(gene)
        descriptions.append(description)
        embeddings.append(get_embedding(description))

        # Print first 3 descriptions for validation
        if i < 3:
            print(f"Gene {i+1}: {gene}")
            print(f"Generated Description: {description}")

        time.sleep(0.1)

    result_df = pd.DataFrame({
        "gene_name": gene_list,
        "gene_description": descriptions,
        "gene_embedding": embeddings  
    })
    
    result_df.to_csv("notasync_des_gene_embeddings.csv", index=False)
    print(f"Saved {len(gene_list)} gene embeddings to des_gene_embeddings.csv")
    return result_df

unique_tahoe_drugs_with_conc = {}
for drug_str in tahoe_drugs:
    drug_tuple = ast.literal_eval(drug_str)
    drug_name, concentration, unit = drug_tuple[0]
    if drug_name not in unique_tahoe_drugs_with_conc:
        unique_tahoe_drugs_with_conc[drug_name] = f"{concentration} {unit}"
unique_tahoe_drugs_with_conc = list(unique_tahoe_drugs_with_conc.items())

all_genes = list(set([gene.upper() for gene in list(replogle_genes) + new_genes]))

# Dataset Statistics
print("\nDataset Statistics:")
print(f"Number of drugs in Tahoe dataset: {len(tahoe_drugs)}")
print(f"Number of unique drugs in Tahoe dataset: {len(unique_tahoe_drugs_with_conc)}")
print(f"Number of genes in Replogle dataset: {len(replogle_genes)}")
print(f"Number of unique genes in Replogle dataset: {len(set(gene.upper() for gene in replogle_genes))}")
print(f"Number of new unique target genes found: {len(new_genes)}")
print(f"Total number of unique genes (Replogle + new): {len(all_genes)}")

# Generate embeddings for drugs and genes
drug_df = generate_and_embed_drug_prompts(unique_tahoe_drugs_with_conc)
gene_df = generate_and_embed_gene_prompts(all_genes)

# Example Output
print("\nExample drug description and embedding:")
print(f"Description: {drug_df.iloc[0]['drug_description']}")
print(f"Embedding (first 5 values): {drug_df.iloc[0]['drug_embedding'][:5]}")

print("\nExample gene description and embedding:")
print(f"Description: {gene_df.iloc[0]['gene_description']}")
print(f"Embedding (first 5 values): {gene_df.iloc[0]['gene_embedding'][:5]}")

# Similarity between first drug and gene
first_drug_embedding = np.array(drug_df.iloc[0]['drug_embedding']).reshape(1, -1)
first_gene_embedding = np.array(gene_df.iloc[0]['gene_embedding']).reshape(1, -1)
similarity = cosine_similarity(first_drug_embedding, first_gene_embedding)[0][0]

print(f"\nSimilarity between '{drug_df.iloc[0]['drug_name']}' ({drug_df.iloc[0]['concentration']}) "
      f"and '{gene_df.iloc[0]['gene_name']}': {similarity:.4f}")

# %%
