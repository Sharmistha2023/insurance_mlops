import pandas as pd
from snowflake import connector
from snowflake.connector.pandas_tools import write_pandas
from sklearn import preprocessing as skpreprocessing
import os
from snowflake import snowpark
from snowflake.ml import dataset
from snowflake.ml.dataset import load_dataset

conn = connector.connect()
cur = conn.cursor()
print("Connected to Snowflake")
cur.execute("SELECT * FROM INSURANCE")
df_version1 = cur.fetch_pandas_all()
print("Version data loaded")
connection_parameters= {}
session = snowpark.Session.builder.configs(connection_parameters).create()
sp_df = session.create_dataframe(df_version1)
ds1 = dataset.create_from_dataframe(
    session,
    "my_dataset_new",
    "version102",
    input_dataframe=sp_df)
print(f"version102: {ds1}")
df_processed = df_version1.drop(['TIMESTAMP', 'UNIQUE_ID'], axis=1)
for col in ['SEX', 'SMOKER', 'REGION']:
    if df_processed[col].dtype == 'object' or df_processed[col].dtype == 'bool':
        if df_processed[col].dtype == 'bool':
            df_processed[col] = df_processed[col].astype(int)
        else:
            le = skpreprocessing.LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
        print(f"Completed Label Encoding on {col}")

resultant_table_name = os.getenv("OUT_TABLE_NAME","PREPROCESSED_DATA")
success, nchunks, nrows, _ = write_pandas(
    conn,
    df_processed,
    table_name=f'{resultant_table_name}',       # just the table name
    auto_create_table=True,
    overwrite=True
)
print(f"Temporary table created, success={success}, rows={nrows}")
sp_df_new = session.create_dataframe(df_processed)
ds2 = dataset.create_from_dataframe(
    session,
    "my_dataset_new",
    "version103",
    input_dataframe=sp_df_new)
print(f"version103:{ds2}")

my_dataset_instance = load_dataset(
    session=session,
    name='MY_DATASET_NEW',
    # Optional: specify the desired version
    version="version103"
)
sp_df = my_dataset_instance.read.to_snowpark_dataframe()
sp_df.show()
df_version4= sp_df.to_pandas()
print(".................")
print(df_version4)
conn.commit()  
cur.close()
conn.close()
print("Done")
