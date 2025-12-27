import os
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# Load dataset
# -----------------------------
EPOCH = int(os.getenv("EPOCH", 10))
INPUT_DATA_PATH = os.getenv("DATASET_PATH", "")
INPUT_FILE = INPUT_DATA_PATH + "/pre_processing.csv"
MLFLOW_EXPERIMENT_NAME = os.getenv('PROJECT_NAME','')
print("Loading:", INPUT_FILE)
data = pd.read_csv(INPUT_FILE)

if MLFLOW_EXPERIMENT_NAME:
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not exp:
        print("Creating experiment...")
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
else:
    mlflow.set_experiment(experiment_name='Default')
# Split features/labels
X = data.drop(["charges"], axis=1).values
y = data["charges"].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1211
)

# Convert to Torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# -----------------------------
# Define model
# -----------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = Net()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Train and Log to MLflow
# -----------------------------
#mlflow.set_experiment("in")

with mlflow.start_run():   # <--- no run_name
    for epoch in range(EPOCH):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        mlflow.log_metric("train_loss", loss.item(), step=epoch)
        print(f"Epoch {epoch} | Loss = {loss.item():.3f}")

    # Log as MLflow Model (required to enable "Register Model")
    mlflow.pytorch.log_model(model, artifact_path="model")

    print("\nModel logged successfully!")
    print("Now go to MLflow UI → Run → Artifacts → model → Register Model\n")
