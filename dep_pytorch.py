import os
import torch
import mlflow
from ray import serve
from fastapi import Request

@serve.deployment
class InsuranceDeployment:
    def __init__(self):
        model_path = os.getenv("MODEL_PATH")
        if not model_path:
            raise ValueError("MODEL_PATH environment variable is missing")

        print(f"Loading MLflow PyTorch model from: {model_path}")
        self.model = mlflow.pytorch.load_model(model_path)
        self.model.eval()
        print("Model loaded successfully!")

    async def __call__(self, request: Request):
        try:
            body = await request.json()

            # Expecting JSON: {"data": [[...],[...]]}
            input_data = torch.tensor(body["data"], dtype=torch.float32)

            with torch.no_grad():
                outputs = self.model(input_data)

            return {"predictions": outputs.numpy().tolist()}

        except Exception as e:
            print("Prediction error:", repr(e))
            return {"error": str(e)}

app = InsuranceDeployment.bind()
