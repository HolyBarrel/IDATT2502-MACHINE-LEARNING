import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

# Read data from csv file
data = pd.read_csv('day_head_circumference.csv', sep=',', header=0, engine='python')

x_train = torch.tensor(data["# day"].values, dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor(data["headcircumference"].values, dtype=torch.float).reshape(-1, 1)

# Prints first 5 rows of data
print(data.head())


class NonLinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], dtype=torch.float, requires_grad=True)  
        self.b = torch.tensor([[0.0]], dtype=torch.float, requires_grad=True)
    
    # Predictor

    def f(self, x):
        return 20 *  torch.sigmoid((x @ self.W + self.b)) + 31 
    
    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = NonLinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam([model.W, model.b], lr=1e-7)

for epoch in range(100000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.title("Predicted head circumference given age")
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x days old')
plt.ylabel('y head circumference [cm]')


x_entries = torch.linspace(torch.min(x_train), torch.max(x_train), steps=750).reshape(-1, 1)
y_entries = model.f(x_entries).detach()

plt.plot(x_entries, y_entries, label='$f(x) = 20\\sigma(xW + b) + 31$')
plt.legend()
plt.show()
