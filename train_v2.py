import mlflow

import mlflow.pytorch

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

 

# Simulate more data (next version)

X = torch.randn(5000, 10)  # Increased dataset

y = torch.randint(0, 2, (5000,))

dataset = TensorDataset(X, y)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

 

class SimpleNet(nn.Module):

    def __init__(self):

        super(SimpleNet, self).__init__()

        self.fc = nn.Linear(10, 2)

    def forward(self, x):

        return self.fc(x)

 

def train_model():

    model = SimpleNet()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

 

    mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment("DeepLearningExperiment")

 

    with mlflow.start_run():

        for epoch in range(15):  # More epochs for v2

            total_loss = 0

            for batch_x, batch_y in loader:

                optimizer.zero_grad()

                outputs = model(batch_x)

                loss = criterion(outputs, batch_y)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

 

            mlflow.log_metric("loss", total_loss / len(loader), step=epoch)

 

        mlflow.log_param("epochs", 15)

        mlflow.log_param("dataset_size", len(X))

        mlflow.pytorch.log_model(model, "model")

 

        # Register as new version

        result = mlflow.register_model(

            f"runs:/{mlflow.active_run().info.run_id}/model",

            "DeepLearningModel"

        )

        print("New model version registered:", result.version)

 

if __name__ == "__main__":

    train_model()