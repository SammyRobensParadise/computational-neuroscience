# Convolutional neural network activity

In this activity, you will make a convolutional neural network (CNN) model in the PyTorch deep learning package to perform classification of hand-written digits in the classic MNIST dataset.

There are 5 tasks for you to complete the example. Cells have clearly marked `# TODO` and `#####` comments for you to insert your code between. Variables assigned to `None` should keep the same name but assigned to their proper implementation.

1. Complete the implementation of a CNN model that uses Conv-ReLU-MaxPool blocks
2. Complete the implementation of a CNN model that uses Conv-ReLU-BatchNorm blocks
3. Train the models for 5 epochs
4. Compare the model results


```python
# TODO: Run this cell to import relevant packages

import torch  # Main torch import for torch tensors (arrays)
import torch.nn as nn  # Neural network module for building deep learning models
import torch.nn.functional as F  # Functional module, includes activation functions
import torch.optim as optim  # Optimization module
import torchvision  # Vision / image processing package built on top of torch

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

# 1. Complete the implementation of the `Conv-ReLU-MaxPool` net

We will build a convolutional neural network to perform classification of the digits in the MNIST dataset. You can assume the inputs to the model have shape (batch, channels, height, width) = (B, 1, 28, 28).

The first conv nets were built with "blocks" of `Conv-ReLU-MaxPool` layers. A block is just a common structure that is repeated through the network. The max pooling layers were strided to reduce the image spatial dimensions when processing deeper into the network. The convolution layers increased the number of channels when processing deeper into the network. The ReLU activation is fairly standard in most deep learning. For image classification, these original networks applied dense layers to the flattened output of the final convolution layer.

The following is a specification for a 3-block `Conv-ReLU-MaxPool` net:

Block 1:
- A convolution layer that goes from 1 input channel to 32 output channels, with a 3x3 kernel and 'same' padding.
    - (B, 1, 28, 28) $\rarr$ (B, 32, 28, 28) 
- ReLU activation
- A max pooling layer with 2x2 kernel size and stride of 2.
    - (B, 32, 28, 28) $\rarr$ (B, 32, 14, 14)

Block 2:
- A convolution layer that goes from 32 input channel to 64 output channels, with a 3x3 kernel and 'same' padding.
    - (B, 32, 14, 14) $\rarr$ (B, 64, 14, 14) 
- ReLU activation
- A max pooling layer with 2x2 kernel size and stride of 2.
    - (B, 64, 14, 14) $\rarr$ (B, 64, 7, 7)

Block 3:
- A convolution layer that goes from 64 input channel to 100 output channels, with a 3x3 kernel and 'same' padding.
    - (B, 64, 7, 7) $\rarr$ (B, 100, 7, 7) 
- ReLU activation
- A max pooling layer with 2x2 kernel size and stride of 2.
    - (B, 100, 7, 7) $\rarr$ (B, 100, 3, 3)

Output:
- Flatten inputs along dimension 1
    - (B, 100, 3, 3) $\rarr$ (B, 900)
- Output linear layer from 900 to 10
    - (B, 900) $\rarr$ (B, 10)

Note: You do not need to apply a softmax to the outputs as this is automatically done with the appropriate loss function.

Relevant documentation:

- [PyTorch nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

- [PyTorch activation functions](https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions)

- [PyTorch linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)

- [PyTorch 2D convolution layer](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

- [PyTorch 2D max pooling layer](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)


```python
class ConvMaxPoolNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Activation function
        self.relu = nn.ReLU()

        # TODO: Create conv and max pool layers for block 1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # TODO: Create conv and max pool layers for block 2
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # TODO: Create conv and max pool layers for block 3
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=100, kernel_size=(3, 3), padding="same"
        )
        self.max_pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # TODO: Create output linear layer
        self.output = nn.Linear(900, 10)

        #####

    def forward(self, x):
        """
        Forward pass implementation for the network

        :param x: torch.Tensor of shape (batch, 1, 28, 28), input images

        :returns: torch.Tensor of shape (batch, 10), output logits
        """
        # TODO: Process x through block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)

        # TODO: Process x through block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)

        # TODO: Process x through block 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool3(x)

        # TODO: Flatten x and process through the output layer
        x = torch.flatten(x, 1)  # (B, 900)
        x = self.output(x)
        #####
        return x
```


```python
# TODO: Run this cell to help test your code
# Tip: Try to build the model layer by layer and test with this cell as you go
# Final expected output shape: (4, 10)
x = torch.randn(4, 1, 28, 28)
cnn_mp = ConvMaxPoolNetwork()

y = cnn_mp(x)
y.shape
```




    torch.Size([4, 10])



## 2. Complete the implementation of the `Conv-ReLU-BatchNorm` net


Many modern conv nets, like ResNet, are built on "blocks" of `Conv-ReLU-BatchNorm`. The convolutions are usually strided instead of using strided max pooling. BatchNorm is an operation that maintains Gaussian statistics of the batch. This ensures the inputs to the next block are normally distributed, aids in efficient training, and reduces overfitting (regularization). 

Instead of flattening before a final linear layer, modern conv nets often use a GlobalAveragePooling operation, that averages across the entirety of both spatial dimensions of each feature map (channel). In PyTorch, this can be be implemented with adaptive average pooling with an output size of 1x1. 

The following is a specification for a 3-block `Conv-ReLU-BatchNorm` net:

Block 1:
- A 3x3 kernel convolution layer that goes from 1 input channel to 32 output channels, stride 2, and padding 1 to ensure the output spatial dimensions are exactly halved.
    - (B, 1, 28, 28) $\rarr$ (B, 32, 14, 14) 
- ReLU activation
- A BatchNorm layer with 32 channels.
    - (B, 32, 14, 14) $\rarr$ (B, 32, 14, 14)

Block 2:
- A 3x3 kernel convolution layer that goes from 32 input channel to 64 output channels, stride 2, and padding 1.
    - (B, 32, 14, 14) $\rarr$ (B, 64, 7, 7) 
- ReLU activation
- A BatchNorm layer with 64 channels.
    - (B, 64, 7, 7) $\rarr$ (B, 64, 7, 7)

Block 3:
- A 3x3 kernel convolution layer that goes from 64 input channel to 100 output channels, stride 2, and padding 1.
    - (B, 64, 7, 7) $\rarr$ (B, 100, 3, 3) 
- ReLU activation
- A BatchNorm layer with 100 channels.
    - (B, 100, 3, 3) $\rarr$ (B, 100, 3, 3)

Output:
- AdaptiveAveragePooling layer with output size (1, 1) (equivalent to GlobalAveragePooling)
    - (B, 100, 3, 3) $\rarr$ (B, 100, 1, 1)
- Remove extra singleton dimensions with `torch.squeeze`
    - (B, 100, 1, 1) $\rarr$ (B, 100)
- Output linear layer from 100 to 10
    - (B, 100) $\rarr$ (B, 10)


```python
class ConvBatchNormNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Activation function
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(3, 3), stride=2, padding=1
        )
        self.batchNorm1 = nn.BatchNorm2d(num_features=32)

        # TODO: Create conv and batch norm layers for block 1
        self.batchNorm2 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1
        )

        # TODO: Create conv and batch norm layers for block 1
        self.batchNorm3 = nn.BatchNorm2d(num_features=100)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=100, kernel_size=(3, 3), stride=2, padding=1
        )

        # TODO: Create adaptive average pooling and output linear layers
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.output = nn.Linear(100, 10)

        #####

    def forward(self, x):
        """
        Forward pass implementation for the network

        :param x: torch.Tensor of shape (batch, 1, 28, 28), input images

        :returns: torch.Tensor of shape (batch, 10), output logits
        """
        # TODO: Process x through block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batchNorm1(x)

        # TODO: Process x through block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batchNorm2(x)

        # TODO: Process x through block 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.batchNorm3(x)

        # TODO: Flatten x and process through the output layer
   


        # TODO: Pool x, squeeze extra dims, and process through the output layer
        x = self.pool(x)
        x = torch.squeeze(x)  # (B, 100)
        x = self.output(x)
        #####
        return x
```


```python
# TODO: Run this cell to help test your code
# Tip: Try to build the model layer by layer and test with this cell as you go
# Final expected output shape: (4, 10)
cnn_bn = ConvBatchNormNetwork()
x = torch.randn(4, 1, 28, 28)
z = cnn_bn(x)
z.shape
```




    torch.Size([4, 10])



# 3. Train each model for 5 epochs
This is the same training loop code as last time, you do not need to modify it.


```python
# TODO: Run this cell to define the train function

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
        optimizer.zero_grad()  # 2. Zero parameter gradients
        outputs = model(inputs)  # 3. Compute model outputs
        loss = loss_fn(outputs, targets)  # 4. Compute loss between outputs and targets
        loss.backward()  # 5. Backpropagation for parameter gradients
        optimizer.step()  # 6. Gradient descent step

        # Track some values to compute statistics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=-1)  # Take the class with the highest output as the prediction
        all_predictions.extend(preds.tolist())
        all_targets.extend(targets.tolist())

        # Print some statistics every 100 batches
        if i % 100 == 0:
            running_loss = total_loss / (i + 1)
            print(f"Epoch {epoch + 1}, batch {i + 1}: loss = {running_loss:.2f}")
     
    acc = accuracy_score(all_targets, all_predictions)

    # Print average loss and accuracy
    print(f"Epoch {epoch + 1} done. Average train loss = {total_loss / len(train_loader):.2f}, average train accuracy = {acc * 100:.3f}%")
```


```python
# TODO: Run this cell to define the test function

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
            outputs = model(inputs)  # 3. Compute model outputs
            loss = loss_fn(outputs, targets)  # 4. Compute loss between outputs and targets

            # Track some values to compute statistics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)  # Take the class with the highest output as the prediction
            all_predictions.extend(preds.tolist())
            all_targets.extend(targets.tolist())
      
    acc = accuracy_score(all_targets, all_predictions)

    # Print average loss and accuracy
    print(f"Epoch {epoch + 1} done. Average test loss = {total_loss / len(test_loader):.2f}, average test accuracy = {acc * 100:.3f}%")
```


```python
# TODO: Run this cell to train the ConvMaxPoolNetwork for 5 epochs
cnn_mp = ConvMaxPoolNetwork()
optimizer = optim.Adam(cnn_mp.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    train(cnn_mp, train_loader, loss_fn, optimizer, epoch)
    test(cnn_mp, test_loader, loss_fn, epoch)
```

    Epoch 1, batch 1: loss = 2.31
    Epoch 1, batch 101: loss = 0.89
    Epoch 1, batch 201: loss = 0.55
    Epoch 1, batch 301: loss = 0.43
    Epoch 1, batch 401: loss = 0.36
    Epoch 1, batch 501: loss = 0.32
    Epoch 1, batch 601: loss = 0.30
    Epoch 1, batch 701: loss = 0.28
    Epoch 1, batch 801: loss = 0.26
    Epoch 1, batch 901: loss = 0.25
    Epoch 1 done. Average train loss = 0.24, average train accuracy = 92.527%
    Epoch 1 done. Average test loss = 0.13, average test accuracy = 95.950%
    Epoch 2, batch 1: loss = 0.12
    Epoch 2, batch 101: loss = 0.14
    Epoch 2, batch 201: loss = 0.13
    Epoch 2, batch 301: loss = 0.13
    Epoch 2, batch 401: loss = 0.13
    Epoch 2, batch 501: loss = 0.13
    Epoch 2, batch 601: loss = 0.13
    Epoch 2, batch 701: loss = 0.13
    Epoch 2, batch 801: loss = 0.13
    Epoch 2, batch 901: loss = 0.13
    Epoch 2 done. Average train loss = 0.13, average train accuracy = 96.075%
    Epoch 2 done. Average test loss = 0.11, average test accuracy = 96.480%
    Epoch 3, batch 1: loss = 0.22
    Epoch 3, batch 101: loss = 0.11
    Epoch 3, batch 201: loss = 0.11
    Epoch 3, batch 301: loss = 0.12
    Epoch 3, batch 401: loss = 0.11
    Epoch 3, batch 501: loss = 0.12
    Epoch 3, batch 601: loss = 0.12
    Epoch 3, batch 701: loss = 0.11
    Epoch 3, batch 801: loss = 0.11
    Epoch 3, batch 901: loss = 0.12
    Epoch 3 done. Average train loss = 0.12, average train accuracy = 96.493%
    Epoch 3 done. Average test loss = 0.11, average test accuracy = 96.720%
    Epoch 4, batch 1: loss = 0.08
    Epoch 4, batch 101: loss = 0.09
    Epoch 4, batch 201: loss = 0.10
    Epoch 4, batch 301: loss = 0.10
    Epoch 4, batch 401: loss = 0.11
    Epoch 4, batch 501: loss = 0.10
    Epoch 4, batch 601: loss = 0.10
    Epoch 4, batch 701: loss = 0.10
    Epoch 4, batch 801: loss = 0.11
    Epoch 4, batch 901: loss = 0.11
    Epoch 4 done. Average train loss = 0.11, average train accuracy = 96.727%
    Epoch 4 done. Average test loss = 0.10, average test accuracy = 96.680%
    Epoch 5, batch 1: loss = 0.15
    Epoch 5, batch 101: loss = 0.10
    Epoch 5, batch 201: loss = 0.10
    Epoch 5, batch 301: loss = 0.11
    Epoch 5, batch 401: loss = 0.11
    Epoch 5, batch 501: loss = 0.11
    Epoch 5, batch 601: loss = 0.11
    Epoch 5, batch 701: loss = 0.11
    Epoch 5, batch 801: loss = 0.11
    Epoch 5, batch 901: loss = 0.11
    Epoch 5 done. Average train loss = 0.11, average train accuracy = 96.730%
    Epoch 5 done. Average test loss = 0.10, average test accuracy = 97.030%



```python
# TODO: Run this cell to train the ConvBatchNormNetwork for 5 epochs
cnn_bn = ConvBatchNormNetwork()
optimizer = optim.Adam(cnn_bn.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    train(cnn_bn, train_loader, loss_fn, optimizer, epoch)
    test(cnn_bn, test_loader, loss_fn, epoch)
```

    Epoch 1, batch 1: loss = 2.30
    Epoch 1, batch 101: loss = 0.61
    Epoch 1, batch 201: loss = 0.40
    Epoch 1, batch 301: loss = 0.32
    Epoch 1, batch 401: loss = 0.27
    Epoch 1, batch 501: loss = 0.24
    Epoch 1, batch 601: loss = 0.21
    Epoch 1, batch 701: loss = 0.20
    Epoch 1, batch 801: loss = 0.18
    Epoch 1, batch 901: loss = 0.17
    Epoch 1 done. Average train loss = 0.17, average train accuracy = 94.907%
    Epoch 1 done. Average test loss = 0.07, average test accuracy = 97.820%
    Epoch 2, batch 1: loss = 0.03
    Epoch 2, batch 101: loss = 0.06
    Epoch 2, batch 201: loss = 0.06
    Epoch 2, batch 301: loss = 0.06
    Epoch 2, batch 401: loss = 0.06
    Epoch 2, batch 501: loss = 0.06
    Epoch 2, batch 601: loss = 0.06
    Epoch 2, batch 701: loss = 0.06
    Epoch 2, batch 801: loss = 0.06
    Epoch 2, batch 901: loss = 0.06
    Epoch 2 done. Average train loss = 0.06, average train accuracy = 98.195%
    Epoch 2 done. Average test loss = 0.05, average test accuracy = 98.350%
    Epoch 3, batch 1: loss = 0.04
    Epoch 3, batch 101: loss = 0.04
    Epoch 3, batch 201: loss = 0.04
    Epoch 3, batch 301: loss = 0.04
    Epoch 3, batch 401: loss = 0.04
    Epoch 3, batch 501: loss = 0.04
    Epoch 3, batch 601: loss = 0.05
    Epoch 3, batch 701: loss = 0.05
    Epoch 3, batch 801: loss = 0.05
    Epoch 3, batch 901: loss = 0.05
    Epoch 3 done. Average train loss = 0.05, average train accuracy = 98.548%
    Epoch 3 done. Average test loss = 0.04, average test accuracy = 98.580%
    Epoch 4, batch 1: loss = 0.05
    Epoch 4, batch 101: loss = 0.03
    Epoch 4, batch 201: loss = 0.04
    Epoch 4, batch 301: loss = 0.03
    Epoch 4, batch 401: loss = 0.04
    Epoch 4, batch 501: loss = 0.04
    Epoch 4, batch 601: loss = 0.04
    Epoch 4, batch 701: loss = 0.04
    Epoch 4, batch 801: loss = 0.04
    Epoch 4, batch 901: loss = 0.04
    Epoch 4 done. Average train loss = 0.04, average train accuracy = 98.790%
    Epoch 4 done. Average test loss = 0.05, average test accuracy = 98.410%
    Epoch 5, batch 1: loss = 0.00
    Epoch 5, batch 101: loss = 0.02
    Epoch 5, batch 201: loss = 0.03
    Epoch 5, batch 301: loss = 0.03
    Epoch 5, batch 401: loss = 0.03
    Epoch 5, batch 501: loss = 0.03
    Epoch 5, batch 601: loss = 0.03
    Epoch 5, batch 701: loss = 0.03
    Epoch 5, batch 801: loss = 0.03
    Epoch 5, batch 901: loss = 0.03
    Epoch 5 done. Average train loss = 0.03, average train accuracy = 99.008%
    Epoch 5 done. Average test loss = 0.05, average test accuracy = 98.660%


# 4. Compare the model results


```python
# If on Google Colab, uncomment and run this
 #!pip install torchinfo
```


```python
# TODO: Run this cell to import torchinfo.summary
from torchinfo import summary
```


```python
# TODO: Run this cell to instantiate a MLP model
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 100)
        self.output = nn.Linear(100, 10)
    
    def forward(self, x):
        """
        Forward pass implementation for the network
        
        :param x: torch.Tensor of shape (batch, 28, 28), input images

        :returns: torch.Tensor of shape (batch, 10), output logits
        """
        x = torch.flatten(x, 1)  # shape (batch, 28*28)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        return x

mlp = MultiLayerPerceptron()
```


```python
# TODO: Run this cell to view a summary of each model and note the number of parameters
print(summary(mlp))
print()
print(summary(cnn_mp))
print()
print(summary(cnn_bn))
```

    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    MultiLayerPerceptron                     --
    ├─Linear: 1-1                            78,500
    ├─Linear: 1-2                            1,010
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    =================================================================
    
    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    ConvMaxPoolNetwork                       --
    ├─ReLU: 1-1                              --
    ├─Conv2d: 1-2                            320
    ├─MaxPool2d: 1-3                         --
    ├─Conv2d: 1-4                            18,496
    ├─MaxPool2d: 1-5                         --
    ├─Conv2d: 1-6                            57,700
    ├─MaxPool2d: 1-7                         --
    ├─Linear: 1-8                            9,010
    =================================================================
    Total params: 85,526
    Trainable params: 85,526
    Non-trainable params: 0
    =================================================================
    
    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    ConvBatchNormNetwork                     --
    ├─ReLU: 1-1                              --
    ├─Conv2d: 1-2                            320
    ├─BatchNorm2d: 1-3                       64
    ├─BatchNorm2d: 1-4                       128
    ├─Conv2d: 1-5                            18,496
    ├─BatchNorm2d: 1-6                       200
    ├─Conv2d: 1-7                            57,700
    ├─AdaptiveAvgPool2d: 1-8                 --
    ├─Linear: 1-9                            1,010
    =================================================================
    Total params: 77,918
    Trainable params: 77,918
    Non-trainable params: 0
    =================================================================



```python
# TODO: Fill in the results in this cell. Then, run the cell to view a table for easy comparison.
import pandas as pd
results = [
    {
        "model": "MLP",
        "epoch_1_train_loss": 0.28,
        "epoch_5_train_loss": 0.16,
        "epoch_5_test_accuracy": 95.38,
        "parameters": None
    },
    {
        "model": "CNN MaxPool",
        "epoch_1_train_loss": None,
        "epoch_5_train_loss": None,
        "epoch_5_test_accuracy": None,
        "parameters": None
    },
    {
        "model": "CNN BatchNorm",
        "epoch_1_train_loss": None,
        "epoch_5_train_loss": None,
        "epoch_5_test_accuracy": None,
        "parameters": None
    }
]
df = pd.DataFrame(results)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>epoch_1_train_loss</th>
      <th>epoch_5_train_loss</th>
      <th>epoch_5_test_accuracy</th>
      <th>parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MLP</td>
      <td>0.28</td>
      <td>0.16</td>
      <td>95.38</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CNN MaxPool</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN BatchNorm</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



Some notes:
- All the models had roughly the same number of parameters. In fact, the final linear layer of the MLP and CNN BatchNorm is exactly the same. However, with the CNN, we have a deep feature extractor based on convolution, while with the MLP, we have a single linear layer and ReLU as the feature extractor. In deep learning, we tend to prefer deeper, more complex, and parameter efficient models that are well suited to processing the particular data, rather than a deep models with only fully connected layers.
- Both CNNs train faster (lower epoch 1 loss) and converge to a better final loss (lower epoch 5 loss) than the MLP. Both CNNs also have a better test accuracy than the MLP.
- CNN BatchNorm performs much better than CNN MaxPool. Indeed, it is a newer architecture that forms the back bone of a lot of modern CNN networks today and empirically has been shown to have better performance on many other tasks than the original CNN formulation.
- It would be best to run multiple trials to ensure the statistical validity of these results.
