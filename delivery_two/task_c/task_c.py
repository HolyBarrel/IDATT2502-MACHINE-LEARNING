import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, art3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

class XOROperatorModel:

    def __init__(self):

        self.W_1 = torch.rand((2, 2), dtype=torch.float32, requires_grad=True)
        self.b_1 = torch.rand((1, 2), dtype=torch.float32, requires_grad=True)
        self.W_2 = torch.rand((2, 1), dtype=torch.float32, requires_grad=True)
        self.b_2 = torch.rand((1, 1), dtype=torch.float32, requires_grad=True)


    # First layer function
    def layer01(self, A):
        return torch.sigmoid(A @ self.W_1 + self.b_1)

    # Second layer function
    def layer11(self, B):
        return B @ self.W_2 + self.b_2 

    # Predictor
    def forward(self, X):
        return self.layer11(self.layer01(X))

    # Binary cross entropy loss function
    def loss(self, y_pred, y_true):
        return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)

    
model = XOROperatorModel()

optimizer = torch.optim.Adam([model.b_1, model.W_1, model.W_2, model.b_2], lr=0.001)

for epoch in range(10000):  
    model.loss(model.forward(x_train), y_train).backward()
    optimizer.step()
    optimizer.zero_grad()


# Gets the predicted output for the training data
with torch.no_grad():
    y_pred_train = model.forward(x_train)

print(f"W_1 = {model.W_1.detach().numpy()}, b_1 = {model.b_1.detach().numpy()}, W_2 = {model.W_2.detach().numpy()}, b_2 = {model.b_2.detach().numpy()}, loss = {model.loss(y_pred_train, y_train).item()}")



a, b = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))

test_data = torch.tensor(np.column_stack((a.ravel(), b.ravel())), dtype=torch.float32)

with torch.no_grad():
    y_pred = model.forward(test_data)
    y_pred_sigmoid = torch.sigmoid(y_pred).numpy()

y_pred_grid = y_pred_sigmoid.reshape(a.shape)

plt.title("XOR Logic Operation Learning Model")
ax.plot_surface(a, b, y_pred_grid, alpha=0.6, label='Predictions', color="cyan")



ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:, 0], c='r', marker='o',  label='True Values of XOR')



ax.grid(True)
ax.set_xlabel("$Input A$")
ax.set_ylabel("$Input B$")
ax.set_zlabel("$Output$")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_zticks([0, 1])
ax.set_xlim(-0.25, 1.25)
ax.set_ylim(-0.25, 1.25)
ax.set_zlim(-0.25, 1.25)



plt.show()