# %%
import numpy as np
import pandas as pd

# %%
tahoe_drugs = np.load("/processed_datasets/scRecount/tahoe/all_tahoe_perturbations.npy", allow_pickle=True)

drug_targets_df = pd.read_excel("tahoe_drug_smiles.xlsx")
print(tahoe_drugs)

# %%
tahoe_drug_names = [drug[0] for drug in tahoe_drugs]
csv_drug_names = drug_targets_df["drug"].astype(str).str.lower().str.strip().tolist()

# %%
total_tahoe = len(tahoe_drug_names)
unique_tahoe = len(set(tahoe_drug_names))

total_csv = len(csv_drug_names)
unique_csv = len(set(csv_drug_names))

matching_drugs = set(tahoe_drug_names) & set(csv_drug_names)
num_matches = len(matching_drugs)

# %%
print(f"Total drugs in Tahoe: {total_tahoe}")
print(f"Unique drugs in Tahoe: {unique_tahoe}")
print(f"Total drugs in CSV: {total_csv}")
print(f"Unique drugs in CSV: {unique_csv}")
print(f"Number of matching drugs: {num_matches}")

# %%
