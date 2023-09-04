import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, art3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y_train = torch.tensor([[1.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

class NandOperatorModel:
    def __init__(self):
        self.W = torch.rand((2, 1), dtype=torch.float32, requires_grad=True)
        self.b = torch.rand((1, 1), dtype=torch.float32, requires_grad=True)

    # Forward pass
    def forward(self, x):
        return x @ self.W + self.b
    
    # Binary cross entropy loss function
    def loss(self, y_pred, y_true):
        return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)
    
model = NandOperatorModel()

optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)


for epoch in range(10000):
    model.loss(model.forward(x_train), y_train).backward()
    optimizer.step()
    optimizer.zero_grad()


a, b = np.meshgrid(np.linspace(0, 1, 15), np.linspace(0, 1, 15))

test_data = torch.tensor(np.column_stack((a.ravel(), b.ravel())), dtype=torch.float32)

with torch.no_grad():
    y_pred = model.forward(test_data)
    y_pred_sigmoid = torch.sigmoid(y_pred).numpy()


y_pred_grid = y_pred_sigmoid.reshape(a.shape)

plt.title("NAND Logic Operation Learning Model")
ax.plot_surface(a, b, y_pred_grid, alpha=0.6, label='Predictions', color="cyan")



ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:, 0], c='r', marker='o',  label='True Values of NAND')



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