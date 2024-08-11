# %%
# %%
import torch
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path

# Define the weight and bias
weight = 0.7
bias = 0.3

# Define the range for X
start = 0
end = 1
skip = 0.02

# %%

import torch

# %%

# Create X and y tensors
X = torch.arange(start, end, skip).unsqueeze(dim=1)
y = weight * X + bias

# %%

# Plot the data
plt.plot(X.numpy(), y.numpy())
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Relationship")
plt.show()

# %%

# %%

# Print the first 10 values of X and y
print("First 10 values of X:", X[:10])
print("First 10 values of y:", y[:10])

# Print the lengths of X and y
print("Length of X:", len(X))
print("Length of y:", len(y))

# %%

# Data splitting
training_split = int(0.8 * len(X))
training_X, training_y = X[:training_split], y[:training_split]
testing_X, testing_y = X[training_split:], y[training_split:]
print(len(training_X), len(training_y), len(testing_X), len(testing_y))

# %%


# %%


# %%


# Function to plot predictions
def plot_predictions(
    train_data=training_X,
    train_label=training_y,
    test_data=testing_X,
    test_label=testing_y,
    predictions=None,
):
    plt.figure(figsize=(10, 7))
    plt.scatter(
        train_data.numpy(), train_label.numpy(), label="Training Data", s=4, color="g"
    )
    plt.scatter(
        test_data.numpy(), test_label.numpy(), label="Testing Data", s=4, color="r"
    )
    if predictions is not None:
        plt.scatter(
            test_data.numpy(), predictions.numpy(), label="Predictions", s=4, color="y"
        )
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Model Predictions")
    plt.legend(prop={"size": 15})
    plt.show()


plot_predictions()

# %%

# %%


# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


torch.manual_seed(42)
model_0 = LinearRegression()
print(model_0.state_dict())

# %%

# %%

# Training setup
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# %%

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_0 = model_0.to(device)
training_X = training_X.to(device)
training_y = training_y.to(device)
testing_X = testing_X.to(device)
testing_y = testing_y.to(device)

# %%

# Training loop
epochs = 200
epochs_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(training_X)
    loss = loss_fn(y_pred, training_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.no_grad():
        test_pred = model_0(testing_X)
        test_loss = loss_fn(test_pred, testing_y)
        if epoch % 10 == 0:
            epochs_count.append(epoch)
            loss_values.append(loss.item())
            test_loss_values.append(test_loss.item())
            print(
                f"Epoch: {epoch} | Loss: {loss.item()} | Test Loss: {test_loss.item()}"
            )
            print(model_0.state_dict())

# %%

# %%

# Plot loss curves
plt.plot(epochs_count, loss_values, label="Training Loss")
plt.plot(epochs_count, test_loss_values, label="Testing Loss")
plt.title("Training and Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%

# %%

# Save the model
Model_path = Path("models")
Model_path.mkdir(parents=True, exist_ok=True)
Model_name = "0_1_model.pth"
Model_save_path = Model_path / Model_name
print(f"Saving model to {Model_save_path}")
torch.save(obj=model_0.state_dict(), f=Model_save_path)

# %%

# %%

# Load the model
load_model_0 = LinearRegression()
load_model_0.load_state_dict(torch.load(Model_save_path))
load_model_0 = load_model_0.to(device)

# Inference and comparison
model_0.eval()
load_model_0.eval()
with torch.no_grad():
    y_pred_new = model_0(testing_X)
    loaded_model_preds = load_model_0(testing_X)

# %%

print(torch.allclose(y_pred_new, loaded_model_preds))

# Plot predictions
plot_predictions(predictions=y_pred_new)

# %%
# %%
# setup device agnostic

device = "cuda" if torch.cuda.is_available() else "cpu"

device

# %%
# creating some data using linear regression  formula  y = weight + X * bias

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(
    dim=1
)  # without unsqueeze errors may pop out
y = weight * X + bias

X[:10], y[:10]

# %%

# %%
# split data

train_split = int(0.8 * len(X))

train_X, train_y = X[:train_split], y[:train_split]
test_X, test_y = X[train_split:], y[train_split:]

len(train_X), len(train_y), len(test_X), len(test_y)

# %%
# plot the data

plot_predictions(train_X, train_y, test_X, test_y)


# %%

# %% [markdown]
# ### Additional Linear Regression Model (Model V2)


# %%
# Define the linear regression model with nn.Linear layer
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super(LinearRegressionModelV2, self).__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1.state_dict())

# %%

# %%

# Model setup
model_1 = model_1.to(device)

# %%

# %%

# Setting up the loss function and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.02)

# %%

# %%

# Training loop for Model V2
epochs = 200
for epoch in range(epochs):
    model_1.train()
    y_pred = model_1(train_X)
    loss = loss_fn(y_pred, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_1.eval()
    with torch.no_grad():
        test_pred = model_1(test_X)
        test_loss = loss_fn(test_pred, test_y)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss.item()} | Test Loss: {test_loss.item()}")
        print(model_1.state_dict())

# %%

# %%

# Plot predictions for Model V2
model_1.eval()
with torch.no_grad():
    y_pred_v2 = model_1(test_X)

plot_predictions(predictions=y_pred_v2)


# %%
# saving the model

from pathlib import Path

Model_Path = Path("models")
Model_Path.mkdir(parents=True, exist_ok=True)

Model_name = "01_Model_python_workflow_01.pth"

Model_save_path = Model_path / Model_name

Model_save_path

# %%
# saving the model path state dict

print(f"Saving model to {Model_save_path}")

torch.save(obj=model_1.state_dict(), f=Model_save_path)


# %%
model_1.state_dict()

# %%
# Load a pytorch model

# creating a new instance for RegressionModelV2

load_model_1 = LinearRegressionModelV2()

# Load the save model state_dict

load_model_1.load_state_dict(torch.load(Model_save_path))

load_model_1.to(device)

# %%
next(load_model_1.parameters()).device

# %%
load_model_1.state_dict()

# %%
load_model_1.eval()

with torch.inference_mode():
    load_model_1_preds = load_model_1(test_X)
y_pred_v2 == load_model_1_preds

# %%
