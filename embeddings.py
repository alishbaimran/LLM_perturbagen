# %%
import numpy as np
import pandas as pd
import ast
import os
from openai import OpenAI
import time
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI() 

tahoe_drugs = np.load("/processed_datasets/scRecount/tahoe/all_tahoe_perturbations.npy", allow_pickle=True)
drug_targets_df = pd.read_excel("tahoe_drug_smiles.xlsx")
replogle_genes = np.load("/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/all_replogle_perturbations.npy", allow_pickle=True)

try:
    new_genes_df = pd.read_csv("new_target_genes_not_in_replogle.csv")
    new_genes = new_genes_df["target_gene"].tolist()
except FileNotFoundError:
    tahoe_drug_names = []
    for drug_str in tahoe_drugs:
        drug_tuple = ast.literal_eval(drug_str)
        drug_name = drug_tuple[0][0]
        tahoe_drug_names.append(drug_name)
    
    csv_drug_names = drug_targets_df["drug"].astype(str).tolist()
    
    tahoe_drug_names_lower = [drug.lower().strip() for drug in tahoe_drug_names]
    csv_drug_names_lower = [drug.lower().strip() for drug in csv_drug_names]
    matching_drugs_lower = set(tahoe_drug_names_lower) & set(csv_drug_names_lower)
    
    csv_drug_case_map = {drug.lower().strip(): idx for idx, drug in enumerate(csv_drug_names)}
    
    target_genes = []
    for drug_lower in matching_drugs_lower:
        if drug_lower in csv_drug_case_map:
            idx = csv_drug_case_map[drug_lower]
            drug_targets = drug_targets_df.iloc[idx]["targets"]
            if pd.notna(drug_targets) and drug_targets and drug_targets.lower() != "none":
                if isinstance(drug_targets, str):
                    genes = [gene.strip() for gene in drug_targets.split(",")]
                    target_genes.extend(genes)
    
    replogle_genes_lower = [gene.lower().strip() for gene in replogle_genes]
    replogle_genes_set = set(replogle_genes_lower)
    target_genes_lower = [gene.lower().strip() for gene in target_genes]
    target_genes_set = set(target_genes_lower)
    
    new_genes = list(target_genes_set - replogle_genes_set)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(
        input=[text], 
        model=model
    ).data[0].embedding
    return embedding

def generate_and_embed_drug_prompts(drug_list):
    """
    Generates prompts and computes embeddings for drugs including actual drug concentrations.
    """
    prompts = []
    embeddings = []
    
    print(f"Generating prompts and embeddings for {len(drug_list)} drugs with concentrations...")

    for drug_name, concentration in tqdm(drug_list):
        prompt = f"what happens if I treat a cell with {concentration} of {drug_name}"
        prompts.append(prompt)
        
        embedding = get_embedding(prompt)
        embeddings.append(embedding)

        time.sleep(0.1)  

    result_df = pd.DataFrame({
        "drug_name": [drug[0] for drug in drug_list],
        "concentration": [drug[1] for drug in drug_list],
        "drug_prompt": prompts,
        "drug_embedding": embeddings  
    })
    
    result_df.to_csv("des_drug_embeddings_with_concentration.csv", index=False)
    print(f"Saved {len(drug_list)} drug embeddings to drug_embeddings_with_concentration.csv")
    return result_df

def generate_and_embed_gene_prompts(gene_list):
    """
    Generates prompts and computes embeddings for genes.
    """
    prompts = []
    embeddings = []
    
    print(f"Generating prompts and embeddings for {len(gene_list)} genes...")

    for gene in tqdm(gene_list):
        prompt = f"what happens if I add perturbation to {gene} in a cell"
        prompts.append(prompt)
        
        embedding = get_embedding(prompt)
        embeddings.append(embedding)

        time.sleep(0.1)  

    result_df = pd.DataFrame({
        "gene_name": gene_list,
        "gene_prompt": prompts,
        "gene_embedding": embeddings  
    })
    
    result_df.to_csv("des_gene_embeddings.csv", index=False)
    print(f"Saved {len(gene_list)} gene embeddings to gene_embeddings.csv")
    return result_df

unique_tahoe_drugs_with_conc = {}

for drug_str in tahoe_drugs:
    drug_tuple = ast.literal_eval(drug_str)  
    drug_name, concentration, unit = drug_tuple[0]  
    
    if drug_name not in unique_tahoe_drugs_with_conc:  
        unique_tahoe_drugs_with_conc[drug_name] = f"{concentration} {unit}"  

unique_tahoe_drugs_with_conc = list(unique_tahoe_drugs_with_conc.items())

print(f"Number of unique drugs with concentrations: {len(unique_tahoe_drugs_with_conc)}")


# print(unique_tahoe_drugs)
all_genes = list(replogle_genes) + new_genes
all_genes = list(set([gene.upper() for gene in all_genes]))  
# print(all_genes)

print("\nDataset Statistics:")
print(f"Number of drugs in Tahoe dataset: {len(tahoe_drugs)}")
print(f"Number of unique drugs in Tahoe dataset: {len(unique_tahoe_drugs_with_conc)}")
print(f"Number of genes in Replogle dataset: {len(replogle_genes)}")
print(f"Number of unique genes in Replogle dataset: {len(set(gene.upper() for gene in replogle_genes))}")
print(f"Number of new unique target genes found: {len(new_genes)}")
print(f"Total number of unique genes (Replogle + new): {len(all_genes)}")

# %%  Generate embeddings for drugs and genes
# drug_df = generate_and_embed_drug_prompts(unique_tahoe_drugs_with_conc)
# gene_df = generate_and_embed_gene_prompts(all_genes)

# print("\nExample drug prompt and embedding:")
# print(f"Prompt: {drug_df.iloc[0]['drug_prompt']}")
# print(f"Embedding (first 5 values): {drug_df.iloc[0]['drug_embedding'][:5]}")

# print("\nExample gene prompt and embedding:")
# print(f"Prompt: {gene_df.iloc[0]['gene_prompt']}")
# print(f"Embedding (first 5 values): {gene_df.iloc[0]['gene_embedding'][:5]}")

# # %% Similarity between first drug and gene as example
# first_drug_embedding = np.array(drug_df.iloc[0]['drug_embedding']).reshape(1, -1)
# first_gene_embedding = np.array(gene_df.iloc[0]['gene_embedding']).reshape(1, -1)
# similarity = cosine_similarity(first_drug_embedding, first_gene_embedding)[0][0]

# print(f"\nSimilarity between '{drug_df.iloc[0]['drug_name']}' ({drug_df.iloc[0]['concentration']}) and '{gene_df.iloc[0]['gene_name']}': {similarity:.4f}")
# %%