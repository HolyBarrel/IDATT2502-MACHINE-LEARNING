import torch
import torch.nn as nn
import torchvision

# COPIED FROM https://gitlab.com/ntnu-tdat3025/cnn/mnist/-/blob/master/nn.py?ref_type=heads

# task_a.py is a further dev of nn.py 

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.conv = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(32 * 14 * 14, 10)

    def logits(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.dense(x.reshape(-1, 32 * 14 * 14))

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print("accuracy = %s" % model.accuracy(x_test, y_test))

    # Prints:
    """
    accuracy = tensor(0.9495)
    accuracy = tensor(0.9653)
    accuracy = tensor(0.9715)
    accuracy = tensor(0.9764)
    accuracy = tensor(0.9777)
    accuracy = tensor(0.9788)
    accuracy = tensor(0.9794)
    accuracy = tensor(0.9797)
    accuracy = tensor(0.9805)
    accuracy = tensor(0.9797)
    accuracy = tensor(0.9797)
    accuracy = tensor(0.9792)
    accuracy = tensor(0.9792)
    accuracy = tensor(0.9795)
    accuracy = tensor(0.9793)
    accuracy = tensor(0.9799)
    accuracy = tensor(0.9792)
    accuracy = tensor(0.9794)
    accuracy = tensor(0.9798)
    accuracy = tensor(0.9788)
    """