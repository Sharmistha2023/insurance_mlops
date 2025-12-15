import pandas as pd
from snowflake import connector
from snowflake.connector.pandas_tools import write_pandas
from sklearn import preprocessing as skpreprocessing
import os

#Connect to Snowflake
conn = connector.connect()
cur = conn.cursor()
print("Connected to Snowflake")
cur.execute("SELECT * FROM INSURANCE")
df_version1 = cur.fetch_pandas_all()
print("Version data loaded")
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
conn.commit()  
cur.close()
conn.close()
print("Done")
