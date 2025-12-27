import os
import pandas as pd
from sklearn import preprocessing as skpreprocessing

# User gives only main directory path
MAIN_DIR = os.getenv("DATASET_PATH", "")

# Search for insurance.csv inside all subdirectories
csv_path = None
for root, dirs, files in os.walk(MAIN_DIR):
    if "insurance.csv" in files:
        csv_path = os.path.join(root, "insurance.csv")
        break

if csv_path is None:
    raise FileNotFoundError("insurance.csv not found in the given main directory.")

print(f"Found: {csv_path}")

# Load file
data = pd.read_csv(csv_path)

# Preprocessing
insurance_input = data.drop(['timestamp','unique_id'], axis=1)

for col in ['sex', 'smoker', 'region']:
    if insurance_input[col].dtype == 'object':
        le = skpreprocessing.LabelEncoder()
        insurance_input[col] = le.fit_transform(insurance_input[col])
        print(f"Completed Label encoding on {col}")

# Save output in MAIN_DIR
output_file_name = os.path.join(MAIN_DIR, "pre_processing.csv")
insurance_input.to_csv(output_file_name, index=False)

print(f"Preprocessing file is stored in {output_file_name}")
