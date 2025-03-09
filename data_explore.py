# %%
import numpy as np
import pandas as pd
import ast
# # %%
# tahoe_drugs = np.load("/processed_datasets/scRecount/tahoe/all_tahoe_perturbations.npy", allow_pickle=True)

# drug_targets_df = pd.read_excel("tahoe_drug_smiles.xlsx")

# print(tahoe_drugs)
# # %%
# tahoe_drug_names = [ast.literal_eval(drug)[0] for drug in tahoe_drugs]

# csv_drug_names = drug_targets_df["drug"].astype(str).str.lower().str.strip().tolist()

# # %%
# tahoe_drug_names = []
# for drug_str in tahoe_drugs:
#     drug_tuple = ast.literal_eval(drug_str)
#     drug_name = drug_tuple[0][0]  
#     tahoe_drug_names.append(drug_name)

# # %%
# tahoe_drug_names_lower = [drug.lower().strip() for drug in tahoe_drug_names]
# csv_drug_names_lower = [drug.lower().strip() for drug in csv_drug_names]
# print(tahoe_drug_names_lower[:5])
# print(csv_drug_names_lower[:5])
# total_tahoe = len(tahoe_drug_names)
# unique_tahoe = len(set(tahoe_drug_names_lower))

# total_csv = len(csv_drug_names)
# unique_csv = len(set(csv_drug_names_lower))

# matching_drugs = set(tahoe_drug_names_lower) & set(csv_drug_names_lower)
# num_matches = len(matching_drugs)

# print(f"Tahoe Drugs Statistics:")
# print(f"  Total drugs: {total_tahoe}")
# print(f"  Unique drugs: {unique_tahoe}")
# print(f"\nCSV Drugs Statistics:")
# print(f"  Total drugs: {total_csv}")
# print(f"  Unique drugs: {unique_csv}")
# print(f"\nMatching Statistics:")
# print(f"  Number of matching drugs: {num_matches}")
# print(f"  Percentage of Tahoe drugs in CSV: {(num_matches/unique_tahoe)*100:.2f}%")
# print(f"  Percentage of CSV drugs in Tahoe: {(num_matches/unique_csv)*100:.2f}%")

# print("\nSample of matching drugs:")
# sample_size = min(5, num_matches)
# for drug in list(matching_drugs)[:sample_size]:
#     print(f"  - {drug}")
# %%


import numpy as np
import pandas as pd
import ast

tahoe_drugs = np.load("/processed_datasets/scRecount/tahoe/all_tahoe_perturbations.npy", allow_pickle=True)
drug_targets_df = pd.read_excel("tahoe_drug_smiles.xlsx")
replogle_genes = np.load("/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/all_replogle_perturbations.npy", allow_pickle=True)

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
matching_drugs_with_targets = []

print("Processing matching drugs to find target genes...")
for drug_lower in matching_drugs_lower:
    if drug_lower in csv_drug_case_map:
        idx = csv_drug_case_map[drug_lower]
        drug_targets = drug_targets_df.iloc[idx]["targets"]
        
        if pd.notna(drug_targets) and drug_targets and drug_targets.lower() != "none":
            if isinstance(drug_targets, str):
                genes = [gene.strip() for gene in drug_targets.split(",")]
                target_genes.extend(genes)
                matching_drugs_with_targets.append(drug_targets_df.iloc[idx]["drug"])

replogle_genes_lower = [gene.lower().strip() for gene in replogle_genes]
replogle_genes_set = set(replogle_genes_lower)

target_genes_lower = [gene.lower().strip() for gene in target_genes]
target_genes_set = set(target_genes_lower)

new_genes = target_genes_set - replogle_genes_set

print(f"\nResults:")
print(f"Total number of matching drugs: {len(matching_drugs_lower)}")
print(f"Number of matching drugs with valid target information: {len(matching_drugs_with_targets)}")
print(f"Total unique target genes found: {len(target_genes_set)}")
print(f"Number of Replogle genes: {len(replogle_genes_set)}")
print(f"Number of target genes not in Replogle: {len(new_genes)}")

print("\nSample of target genes not in Replogle dataset:")
sample_size = min(10, len(new_genes))
for gene in list(new_genes)[:sample_size]:
    print(f"  - {gene}")

new_genes_df = pd.DataFrame(list(new_genes), columns=["target_gene"])
# new_genes_df.to_csv("new_target_genes_not_in_replogle.csv", index=False)
# print(f"\nSaved {len(new_genes)} new target genes to 'new_target_genes_not_in_replogle.csv'")
# %%
