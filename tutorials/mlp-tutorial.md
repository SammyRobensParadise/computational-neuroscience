# Multi-layer perceptron activity

In this activity, you will make a multi-layer perceptron (MLP) model in the PyTorch deep learning package to perform classification of hand-written digits in the classic MNIST dataset.

There are 5 tasks for you to complete the example. Cells have clearly marked `# TODO` and `#####` comments for you to insert your code between. Variables assigned to `None` should keep the same name but assigned to their proper implementation.

1. Complete the implementation of the MLP model by completing the `__init__` and `forward` functions.
2. Setup the optimizer and loss function for training.
3. Fill in the missing steps in the train and test loop.
4. Train the model for 5 epochs.
5. Visualize the model predictions on some examples.


```python
# TODO: Run this cell to import relevant packages

import torch  # Main torch import for torch tensors (arrays)
import torch.nn as nn  # Neural network module for building deep learning models
import torch.nn.functional as F  # Functional module, includes activation functions
import torch.optim as optim  # Optimization module
import torchvision  # Vision / image processing package built on top of torch

from matplotlib import pyplot as plt  # Plotting and visualization
from sklearn.metrics import accuracy_score  # Computing accuracy metric
```


```python
# TODO: Run this cell to download the data and setup the pre-processing pipeline

# Common practice to normalize input data to neural networks (0 mean, unit variance)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # All inputs to PyTorch neural networks must be torch.Tensor
    torchvision.transforms.Normalize(mean=0.1307, std=0.3081)  # Subtracts mean and divides by std. Note that the raw data is between [0, 1]
])

# Download the MNIST data and lazily apply the transformation pipeline
train_data = torchvision.datasets.MNIST('./datafiles/', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST('./datafiles/', train=False, download=True, transform=transform)

# Setup data loaders
# Note: Iterating through the dataloader yields batches of (inputs, targets)
# where inputs is a torch.Tensor of shape (batch, 28, 28) and targets is a torch.Tensor of shape (batch,)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./datafiles/MNIST/raw/train-images-idx3-ubyte.gz


    100.0%


    Extracting ./datafiles/MNIST/raw/train-images-idx3-ubyte.gz to ./datafiles/MNIST/raw


    100.0%

    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./datafiles/MNIST/raw/train-labels-idx1-ubyte.gz
    Extracting ./datafiles/MNIST/raw/train-labels-idx1-ubyte.gz to ./datafiles/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz


    
    2.0%

    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./datafiles/MNIST/raw/t10k-images-idx3-ubyte.gz


    100.0%


    Extracting ./datafiles/MNIST/raw/t10k-images-idx3-ubyte.gz to ./datafiles/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz


    100.0%

    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./datafiles/MNIST/raw/t10k-labels-idx1-ubyte.gz
    Extracting ./datafiles/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./datafiles/MNIST/raw
    


    



```python
# TODO: Run this cell to visualize 20 examples from the test dataset

fig, axs = plt.subplots(4, 5, figsize=(5, 6))

plot_images = []
plot_labels = []

for i, ax in enumerate(axs.flatten(), start=1000):
    (image, label) = test_data[i]

    # Save this data for later
    plot_images.append(image)
    plot_labels.append(label)

    # Plot each image
    ax.imshow(image.squeeze(), cmap="viridis")
    ax.set_title(f"Label: {label}")
    ax.axis("off")
plt.show()

plot_images = torch.cat(plot_images)  # Combine all the images into a single batch for later

print(f"Each image is a torch.Tensor and has shape {image.shape}.")
print(f"The labels are the integers 0 to 9, representing the digits.")
```


    
![png](mlp-tutorial_files/mlp-tutorial_3_0.png)
    


    Each image is a torch.Tensor and has shape torch.Size([1, 28, 28]).
    The labels are the integers 0 to 9, representing the digits.


# 1. Complete the implementation of the MLP model

Although we draw diagrams of hidden layers as neurons with incoming and outgoing connections, in practice, we implement this with two linear layers (also called "dense layers") and a pointwise non-linearity in between. The first layer is a linear transform (matrix multiplication) from the input dimension to the hidden dimension. The second layer is a linear transform from the hidden dimension to the output dimension.

For this model, the input dimension is (28*28) as we will flatten the 2D images into a 1D vector. The hidden dimension is 100. The output dimension is 10, since we have 10 classes. We will use the ReLU non-linearity.

In PyTorch, a model is defined by subclassing the `nn.Module` class and we define behaviour in two methods. In the `__init__` method, we setup the model architecture such as number layers, the size of each layer, etc. In the `forward()` method, we define the operations performed by the model's layers on the input data to produce outputs.

Note: You do not need to apply a softmax to the outputs as this is automatically done with the appropriate loss function.

Relevant documentation:

- [PyTorch nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

- [PyTorch activation functions](https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions)

- [PyTorch linear layer documentation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)


```python
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Assign self.hidden to a torch linear layer of the correct size
        self.hidden = nn.Linear(28 * 28, 100)
        # TODO: Assign self.output to a torch linear layer of the correct size
        self.output = nn.Linear(100, 10)
        #####

    def forward(self, x):
        """
        Forward pass implementation for the network

        :param x: torch.Tensor of shape (batch, 28, 28), input images

        :returns: torch.Tensor of shape (batch, 10), output logits
        """
        x = torch.flatten(x, 1)  # shape (batch, 28*28)
        # TODO: Process x through self.hidden, relu, and self.output and return the result
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        return x
        #####
```


```python
# TODO: Run this cell to test your implementation. You should expect an output tensor of shape (2, 10)

mlp = MultiLayerPerceptron()
x = torch.randn(2, 28, 28)
z = mlp(x)
z.shape
```




    torch.Size([2, 10])



# 2. Setup optimizer and loss function

The current standard optimizer in deep learning is the Adam optimizer. Use a learning rate of $1\times 10^{-2}$. 

The task we are performing is multiclass classification (10 independent classes, one for each digit). The loss function to use for this task is cross entropy loss.

Relevant documentation:
- [PyTorch optimizers](https://pytorch.org/docs/stable/optim.html)

- [PyTorch loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions)


```python
# TODO: Instantiate your model and setup the optimizer

model = MultiLayerPerceptron()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
```


```python
# TODO: Setup the cross entropy loss function

loss_fn = nn.CrossEntropyLoss()
```

# 3. Fill in the missing steps of the train and test loop

During the training loop, we perform the following steps:

1. Fetch the next batch of inputs and targets from the dataloader
2. Zero the parameter gradients
3. Compute the model output predictions from the inputs
4. Compute the loss between the model outputs and the targets
5. Compute the parameter gradients with backpropagation
6. Perform a gradient descent step with the optimizer to update the model parameters

Relevant documentation:
- [PyTorch optimization step](https://pytorch.org/docs/stable/optim.html#taking-an-optimization-step)


```python
def train(model, train_loader, loss_fn, optimizer, epoch=-1):
    """
    Trains a model for one epoch (one pass through the entire training data).

    :param model: PyTorch model
    :param train_loader: PyTorch Dataloader for training data
    :param loss_fn: PyTorch loss function
    :param optimizer: PyTorch optimizer, initialized with model parameters
    :kwarg epoch: Integer epoch to use when printing loss and accuracy
    """
    total_loss = 0
    all_predictions = []
    all_targets = []

    model.train()  # Set model in training mode
    for i, (inputs, targets) in enumerate(train_loader):  # 1. Fetch next batch of data
        # TODO: Fill in the rest of the training loop
        # 2. Zero parameter gradients
        outputs = None  # 3. Compute model outputs
        loss = None  # 4. Compute loss between outputs and targets
        loss.backward()  # 5. Backpropagation for parameter gradients
        # 6. Gradient descent step
        #####

        # Track some values to compute statistics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=-1)  # Take the class with the highest output as the prediction
        all_predictions.extend(preds.tolist())
        all_targets.extend(targets.tolist())

        # Print some statistics every 100 batches
        if i % 100 == 0:
            running_loss = total_loss / (i + 1)
            print(f"Epoch {epoch + 1}, batch {i + 1}: loss = {running_loss:.2f}")

    # TODO: Compute the overall accuracy        
    acc = None
    #####

    # Print average loss and accuracy
    print(f"Epoch {epoch + 1} done. Average train loss = {total_loss / len(train_loader):.2f}, average train accuracy = {acc * 100:.3f}%")
```


```python
def test(model, test_loader, loss_fn, epoch=-1):
    """
    Tests a model for one epoch of test data.

    Note:
        In testing and evaluation, we do not perform gradient descent optimization, so steps 2, 5, and 6 are not needed.
        For performance, we also tell torch not to track gradients by using the `with torch.no_grad()` context.

    :param model: PyTorch model
    :param test_loader: PyTorch Dataloader for test data
    :param loss_fn: PyTorch loss function
    :kwarg epoch: Integer epoch to use when printing loss and accuracy
    """
    total_loss = 0
    all_predictions = []
    all_targets = []

    model.eval()  # Set model in evaluation mode
    for i, (inputs, targets) in enumerate(test_loader):  # 1. Fetch next batch of data
        with torch.no_grad():
            # TODO: Compute the model outputs and loss only. Do not update using the optimizer
            outputs = None  # 3. Compute model outputs
            loss = None  # 4. Compute loss between outputs and targets
            #####

            # Track some values to compute statistics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)  # Take the class with the highest output as the prediction
            all_predictions.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    # TODO: Compute the overall accuracy        
    acc = None
    #####

    # Print average loss and accuracy
    print(f"Epoch {epoch + 1} done. Average test loss = {total_loss / len(test_loader):.2f}, average test accuracy = {acc * 100:.3f}%")
```

# 4. Train the model for 5 epochs


```python
# TODO: Copy the setup for the model, optimizer, and loss function from Section 2 to here
# Then, run this cell to train the model for 5 epochs
model = None

#####

for epoch in range(5):
    # TODO: Fill in the rest of the arguments to the train and test functions
    train(model, ..., epoch=epoch)
    test(model, ..., epoch=epoch)
    #####
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /Users/sammyrobens-paradise/projects/computational-neuroscience/tutorials/mlp_tutorial.ipynb Cell 15 in <cell line: 7>()
          <a href='vscode-notebook-cell:/Users/sammyrobens-paradise/projects/computational-neuroscience/tutorials/mlp_tutorial.ipynb#X20sZmlsZQ%3D%3D?line=4'>5</a> #####
          <a href='vscode-notebook-cell:/Users/sammyrobens-paradise/projects/computational-neuroscience/tutorials/mlp_tutorial.ipynb#X20sZmlsZQ%3D%3D?line=6'>7</a> for epoch in range(5):
          <a href='vscode-notebook-cell:/Users/sammyrobens-paradise/projects/computational-neuroscience/tutorials/mlp_tutorial.ipynb#X20sZmlsZQ%3D%3D?line=7'>8</a>     # TODO: Fill in the rest of the arguments to the train and test functions
    ----> <a href='vscode-notebook-cell:/Users/sammyrobens-paradise/projects/computational-neuroscience/tutorials/mlp_tutorial.ipynb#X20sZmlsZQ%3D%3D?line=8'>9</a>     train(model, ..., epoch=epoch)
         <a href='vscode-notebook-cell:/Users/sammyrobens-paradise/projects/computational-neuroscience/tutorials/mlp_tutorial.ipynb#X20sZmlsZQ%3D%3D?line=9'>10</a>     test(model, ..., epoch=epoch)


    TypeError: train() missing 2 required positional arguments: 'loss_fn' and 'optimizer'


# 5. Visually compare the model predictions

We will lastly see the trained model's predictions on the 20 examples we visualized in the beginning.


```python
# TODO: Run this cell to visualize the data

# Evaluate the model on the plot_images
model.eval()

with torch.no_grad():
    plot_outputs = model(plot_images)
    plot_preds = torch.argmax(plot_outputs, dim=-1)

# Plot and show the labels
fig, axs = plt.subplots(4, 5, figsize=(7, 8))

for i, ax in enumerate(axs.flatten()):
    image = plot_images[i]
    label = plot_labels[i]
    pred = plot_preds[i]

    ax.imshow(image.squeeze(), cmap="viridis")
    ax.set_title(f"Prediction: {pred}\nLabel: {label}")
    ax.axis("off")
plt.show()
```
