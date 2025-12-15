import pandas as pd
from snowflake import connector
from snowflake.connector.pandas_tools import write_pandas
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as skpreprocessing
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")
import requests, argparse
requests.packages.urllib3.disable_warnings()
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10,
                        help='The number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.01,
                        help="learning rate for optimizer")
args = parser.parse_args()
MLFLOW_EXPERIMENT_NAME = os.getenv('PROJECT_NAME','')
# EPOCHS, DATASET_URL could be specified as Environment parameters at the time of creating JL or Run
# Experiment with this parameter. 
NUM_EPOCHS = int(os.getenv("EPOCHS", args.epochs))
LEARNING_RATE = args.learning_rate
## connecting with Snowflake
conn = connector.connect()
cur = conn.cursor()
print("Connected to Snowflake")
table_name = os.getenv("SNOWFLAKE_TABLE", "")

#cur.execute("SELECT * FROM INSURANCE_TMP")
# mentioned table name
cur.execute(f"SELECT * FROM {table_name}")
df_version1 = cur.fetch_pandas_all()
print("Version data loaded")
# Keep track of models.
OUTPUT_MODEL_DIR = os.getcwd()+"/model"
## create OUTPUT_MODEL_DIR
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
# #### MLFLOW TRACKING INITIALIZATION
if MLFLOW_EXPERIMENT_NAME:
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not exp:
        print("Creating experiment...")
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
else:
    mlflow.set_experiment(experiment_name='Default')
data = df_version1
insurance_input = data.drop(['CHARGES'],axis=1)
insurance_target = data['CHARGES']  
#standardize data
x_scaled = StandardScaler().fit_transform(insurance_input)
x_train, x_test, y_train, y_test = train_test_split(x_scaled,
                                                    insurance_target.values,
                                                    test_size = 0.25,
                                                    random_state=1211)
tf.random.set_seed(42)  #first we set random seed
model = keras.Sequential([
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])
model.compile(loss='mean_absolute_error',
             optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
# mlflow metric logging
class loggingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric("train_loss", logs["loss"], step=epoch)
        mlflow.log_metric("val_loss", logs["val_loss"], step=epoch)
        # output accuracy metric for katib to collect from stdout
        print(f"loss={round(logs['loss'],2)}")


with mlflow.start_run(run_name="insurance") as run:
    
    model.fit(x_train, y_train, epochs = NUM_EPOCHS, verbose=0,
                validation_split=0.1, callbacks=[loggingCallback()])
    
    # Exporting model
    model.save(os.path.join(OUTPUT_MODEL_DIR, 'model.keras'))
    mlflow.log_artifact(os.path.join(OUTPUT_MODEL_DIR, 'model.keras'), artifact_path="model")    
    signature = infer_signature(x_test, model.predict(x_test))
    mlflow.keras.log_model(model, artifact_path="keras_model", signature=signature)
    mlflow.log_params({
    "NUM_EPOCHS": NUM_EPOCHS,
    "LEARNING_RATE": LEARNING_RATE})
print("Training Complete !")
