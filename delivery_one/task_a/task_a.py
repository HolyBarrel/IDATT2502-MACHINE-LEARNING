import torch
import matplotlib.pyplot as plt
import pandas as pd

# Read data from csv file
data = pd.read_csv('length_weight.csv', sep=',', header=0, engine='python')

x_train = torch.tensor(data.iloc[:, 0].values, dtype=torch.double).reshape(-1, 1)
y_train = torch.tensor(data.iloc[:, 1].values, dtype=torch.double).reshape(-1, 1)

# Prints first 5 rows of data
print(data.head())

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], dtype=torch.double, requires_grad=True)  
        self.b = torch.tensor([[0.0]], dtype=torch.double, requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(500000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]]) 
plt.plot(x, model.f(x).detach(), label='$f(x) = xW+b$')
plt.legend()
plt.show()
