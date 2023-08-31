import torch
import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure()
ax = fig.add_subplot(projection='3d') 

# Read data from csv file
data = pd.read_csv('day_length_weight.csv', sep=',', header=0, engine='python')

x_train = torch.tensor(data["# day"].values, dtype=torch.double).reshape(-1, 1)
y_train = torch.tensor(data["length"].values, dtype=torch.double).reshape(-1, 1)
z_train = torch.tensor(data["weight"].values, dtype=torch.double).reshape(-1, 1)

# Prints first 5 rows of data
print(data.head())

class LinearRegression3DModel:

    def __init__(self):
        # Model variables
        self.W_1 = torch.tensor([[0.0]], dtype=torch.double, requires_grad=True)  
        self.W_2 = torch.tensor([[0.0]], dtype=torch.double, requires_grad=True)  
        self.b = torch.tensor([[0.0]], dtype=torch.double, requires_grad=True)

    # Predictor
    def f(self, x, y):
        return x @ self.W_1 + y @ self.W_2 + self.b  

    # Uses Mean Squared Error
    def loss(self, x, y, z):
        return torch.nn.functional.mse_loss(self.f(x, y), z)
    
    def parameters(self):
        return (self.W_1, self.W_2)
    
model = LinearRegression3DModel()

# Optimizes the model using the parameters (W_1 & W_2)  and b
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)

for epoch in range(100):
    model.loss(x_train, y_train, z_train).backward()  # Compute loss gradients
    optimizer.step()  # Performs optimization step

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W_1 = %s,W_2 = %s, b = %s, loss = %s" % (model.W_1, model.W_2, model.b, model.loss(x_train, y_train, z_train))) 

# Visualize result
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]]) 
y = torch.tensor([[torch.min(y_train)], [torch.max(y_train)]]) 
z = torch.tensor([[torch.min(z_train)], [torch.max(z_train)]]) 
ax.plot_wireframe(x, y, z,  color='r', label='$(x^{(i)},y^{(i)},z^{(i)})$')
ax.set_xlabel('x days old')
ax.set_ylabel('y length [cm]')
ax.set_zlabel('z weight [kg]')

ax.plot(x_train, y_train, z_train, 'o')
#plt.plot(x, y, , label='$f(x,y) = xW_1+yW_2+b$')
plt.legend()
plt.show()
