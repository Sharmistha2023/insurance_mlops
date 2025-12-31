import pandas as pd
from snowflake import connector
from snowflake.connector.pandas_tools import write_pandas
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ray
from ray.runtime_context import get_runtime_context
## connecting with Snowflake
conn = connector.connect()
cur = conn.cursor()
print("Connected to Snowflake")
table_name = os.getenv("SNOWFLAKE_TABLE", "PREPROCESSED_DATA")

#cur.execute("SELECT * FROM INSURANCE_TMP")
# mentioned table name
cur.execute(f"SELECT * FROM {table_name}")
df_version1 = cur.fetch_pandas_all()
print("Version data loaded")
# Keep track of models.
OUTPUT_MODEL_DIR = os.getcwd()+"/model"
## create OUTPUT_MODEL_DIR
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
data = df_version1
insurance_input = data.drop(['CHARGES'],axis=1)
insurance_target = data['CHARGES']  
def safe_get_job_id():
    try:
        if ray.is_initialized():
            return get_runtime_context().get_job_id()
        else:
            return "no-ray-job"
    except Exception:
        return "no-ray-job"
# # Load dataset
MLFLOW_EXPERIMENT_NAME = os.getenv("PROJECT_NAME", "Default")
X_train, X_test, y_train, y_test = train_test_split(
    insurance_input, insurance_target, test_size=0.25, random_state=42
)
ray.init()  # Connects to cluster if on DKube; otherwise initializes local Ray
cluster_name = os.environ.get("HOSTNAME", "raycluster").split("-")[0]
job_id = safe_get_job_id()
print(f"Ray cluster: {cluster_name}")
print(f"Ray job_id : {job_id}")

# -----------------------------
# Ray remote function for training
# -----------------------------
@ray.remote
def train_and_log_model(X_train, y_train, X_test, y_test, experiment_name, tags):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.set_tags(tags)

        # Train RandomForest
        model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)

        # Log model
        mlflow.sklearn.log_model(model, "model")
    
    return f"Model logged successfully! Metrics -> MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}"

# -----------------------------
# MLflow tags
# -----------------------------
tags = {
    "experiment_name": MLFLOW_EXPERIMENT_NAME,
    "job_id": job_id,
    "ray cluster": cluster_name
}

# -----------------------------
# Execute Ray task
# -----------------------------
result_ref = train_and_log_model.remote(X_train, y_train, X_test, y_test, MLFLOW_EXPERIMENT_NAME, tags)
result = ray.get(result_ref)
print(result)
