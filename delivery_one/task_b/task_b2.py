import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Read the data
data = pd.read_csv('day_length_weight.csv', sep=',', header=0, engine='python')

# Extract input and output data
days = data['# day'].values
length = data['length'].values
weight = data['weight'].values

# Convert to PyTorch tensors for training
x_train = torch.tensor(data[['# day', 'length']].values, dtype=torch.float)
y_train = torch.tensor(weight, dtype=torch.float).reshape(-1, 1)


class LinearRegression3DModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], dtype=torch.float, requires_grad=True)  
        self.b = torch.tensor([[0.0]], dtype=torch.float, requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))
    
model = LinearRegression3DModel()

# Optimizes the model using the parameters
optimizer = torch.optim.SGD([model.W, model.b], lr=1e-7)

for epoch in range(500000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Performs optimization step

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train))) 

xx, yy = np.meshgrid(np.linspace(min(x_train[:, 0]), max(x_train[:, 0]), 50),
                     np.linspace(min(x_train[:, 1]), max(x_train[:, 1]), 50))

zz = np.c_[xx.ravel(), yy.ravel()]

pred = model.f(torch.tensor(zz, dtype=torch.float)).detach().numpy()
zz = pred.reshape(xx.shape)

# Plot the surface
ax.plot_surface(xx, yy, zz, alpha=0.6, color="cyan")

# Scatters plot
ax.scatter(x_train[:, 0], x_train[:, 1], y_train, color='r', marker='o')

ax.set_xlabel('Day')
ax.set_ylabel('Length[cm]')
ax.set_zlabel('Weight')
ax.set_title('Estimated weight based on days old and length of babies')


plt.show()
