import os
import tensorflow as tf
import mlflow
import mlflow.keras
from ray import serve


@serve.deployment
class InsuranceModel:
    def __init__(self):
        # print("🔄 Loading SavedModel:", os.environ["MODEL_PATH"])
        # self.model = mlflow.tensorflow.load_model(os.environ["MODEL_PATH"])
        # print("✅ Model loaded successfully!")
        # model_path = os.environ["MODEL_PATH"]
        # new_path = model_path + "/data"
        # #new_path = model_path + "/data"
        # print("🔄 Loading TensorFlow SavedModel:", new_path)

        # # Load TF SavedModel directly
        # self.model = tf.saved_model.load(new_path)
        model_path = os.environ["MODEL_PATH"]
        new_path = model_path + "/data"
        #print("🔄 Loading MLflow Keras model:", new_path)
        print("🔄 Loading MLflow Keras model:", model_path)
        self.model = mlflow.keras.load_model(model_path)

        print("✅ Model loaded successfully!")

    async def __call__(self, request):
        try:
            body = await request.json()
            inputs = tf.constant(body["data"], dtype=tf.float32)
            preds = self.model.predict(inputs)
            return {"predictions": preds.tolist()}

        except Exception as e:
            print("🔥 ERROR in prediction:", repr(e))
            raise e


app = InsuranceModel.bind()
