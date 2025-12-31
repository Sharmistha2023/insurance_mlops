import os
import mlflow
import numpy as np
import pandas as pd
from ray import serve
from fastapi import Request

@serve.deployment
class InsuranceDeployment:
    def __init__(self):
        model_path = os.getenv("MODEL_PATH")
        print("Loading MLflow model:", model_path)

        self.model = mlflow.sklearn.load_model(model_path)
        print("Model loaded successfully!")

    async def __call__(self, request: Request):
        try:
            body = await request.json()
            raw_data = body["data"]

            df = pd.DataFrame(raw_data, columns=[
                "AGE", "SEX", "BMI", "CHILDREN", "SMOKER", "REGION"
            ])

            preds = self.model.predict(df)
            return {"predictions": preds.tolist()}

        except Exception as e:
            print("🔥 Prediction error:", e)
            return {"error": str(e)}

app = InsuranceDeployment.bind()
