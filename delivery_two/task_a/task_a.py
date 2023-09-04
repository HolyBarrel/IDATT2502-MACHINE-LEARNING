import torch
import matplotlib.pyplot as plt
import numpy as np

x_train = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
y_train = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

class NotOperatorModel: 
    def __init__(self):

        self.W = torch.rand((1, 1), dtype=torch.float32, requires_grad=True)
        self.b = torch.rand((1, 1), dtype=torch.float32, requires_grad=True)

    # Forward pass
    def forward(self, x):
        return x @ self.W + self.b
    
    # Binary cross entropy loss function
    def loss(self, y_pred, y_true):
        return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)

model = NotOperatorModel()

optimizer = torch.optim.SGD([model.W, model.b], lr=0.5)


for epoch in range(100000):
    model.loss(model.forward(x_train), y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Displays the trained parameters and loss
print(f"W = {model.W.detach().numpy()}, b = {model.b.detach().numpy()}, loss = {model.loss(model.forward(x_train), y_train)}")

# Generate a range of x values
x_range = torch.linspace(0, 1, 10000).view(-1, 1)

# Uses the model to get the y values
y_range = torch.sigmoid(model.forward(x_range)).detach().numpy()

# Plots the sigmoid curve
plt.figure()
plt.plot(x_range, y_range, label='Sigmoid Curve (Trained)', color='purple')

# Plots the true values
plt.scatter(x_train, y_train, color='black', label='True Values (for the NOT-operator)', marker='x', s=120)

# Plots the predicted values
predictions = torch.sigmoid(model.forward(x_train)).detach().numpy()
plt.scatter(x_train, predictions, color='green', label='Predicted Values', alpha=0.85)

plt.title('NOT Operator using Sigmoid Curve')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()