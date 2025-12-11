import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as skpreprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
INPUT_DATA_PATH = os.getenv("DATASET_PATH", "")
INPUT_FILE = INPUT_DATA_PATH + "/insurance.csv"
print(f"INPUT_FILE: {INPUT_FILE}")
data = pd.read_csv(INPUT_FILE)
insurance_input = data.drop(['timestamp','unique_id'],axis=1)
for col in ['sex', 'smoker', 'region']:
    if (insurance_input[col].dtype == 'object'):
        le = skpreprocessing.LabelEncoder()
        le = le.fit(insurance_input[col])
        insurance_input[col] = le.transform(insurance_input[col])
        print('Completed Label encoding on',col)
output_file_name = "/pre_processing.csv"
insurance_input.to_csv(f"{INPUT_DATA_PATH}{output_file_name}", index=False)
print(f"preprocessing file is stored in {INPUT_DATA_PATH}{output_file_name} location ")
