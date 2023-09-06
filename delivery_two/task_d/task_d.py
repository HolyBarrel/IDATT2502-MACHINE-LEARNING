import torch
import torchvision
import matplotlib.pyplot as plt
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Print the shape of the training data
print(x_train.shape)
print(y_train.shape)


# # Show the input of the first observation in the training set
# plt.imshow(x_train[0, :].reshape(28, 28))

# # Print the classification of the first observation in the training set
# print(y_train[0, :])

# # Save the input of the first observation in the training set
# plt.imsave('x_train_1.png', x_train[0, :].reshape(28, 28))

# plt.show()

# # Save the input of the first observation in the training set
class SoftMaxModel(torch.nn.Module):
    def __init__(self):
        super(SoftMaxModel, self).__init__()
        # 784 rows, 10 columns
        self.W = torch.nn.Parameter(torch.randn((784, 10), requires_grad=True))
        self.b = torch.nn.Parameter(torch.randn((1, 10), requires_grad=True))
        

    # Predictor
    def forward(self, x):
        return torch.nn.functional.softmax(x @ self.W + self.b, dim=1)

    # Using cross entropy loss function
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(x, torch.argmax(y, dim=1))
    
    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(torch.argmax(self.forward(x), dim=1), torch.argmax(y, dim=1)).float())

    

model = SoftMaxModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0101)

for epoch in range(1500):
    model.train()
    model.loss(model.forward(x_train), y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if(epoch % 75 == 0): print("Accuracy: ", model.accuracy(x_test, y_test))

    if model.accuracy(x_test, y_test) >= 0.9:
        print("Achieved desired accuracy, stopping training.")
        # Creates a directory to save the weight images if it doesn't exist
        output_directory = 'generated_images'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Plots the weights for the i'th class
        for i in range(10):
            plt.figure()
            # Get the weights for the i'th class
            weight_image = model.W.detach().numpy()[:, i].reshape(28, 28)
            
            # Plots the weights
            plt.imshow(weight_image, cmap='gray')  
            plt.title(f'Visualization of W for {i}')
            plt.axis('off')
            
            # Saves the image
            plt.savefig(os.path.join(output_directory, f'number_{i}_weights.png'))

            plt.show()



        # Plots the model guesses versus test data
        plt.figure()
        plt.scatter(torch.argmax(model.forward(x_test), dim=1).detach().numpy(), torch.argmax(y_test, dim=1).detach().numpy(), alpha=0.6)
        plt.xlabel('Model guess')
        plt.ylabel('True value')
        plt.savefig(os.path.join(output_directory, f'guesses_vs_true_values.png'))
        plt.show()

        # Number of images to show in the plot
        num_images = 10

        test_images = x_test[:num_images].reshape(-1, 28, 28)

        # Get model predictions for these images
        predictions = torch.argmax(model.forward(x_test[:num_images]), dim=1).detach().numpy()

        # Plot images with predictions
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.ravel()

        for i in range(num_images):
            axes[i].imshow(test_images[i], cmap='gray')
            axes[i].set_title(f"Prediction: {predictions[i]}")
            axes[i].axis('off')

        plt.subplots_adjust(wspace=0.5)
        plt.savefig(os.path.join(output_directory, 'model_guesses.png'))
        plt.show()

        break


