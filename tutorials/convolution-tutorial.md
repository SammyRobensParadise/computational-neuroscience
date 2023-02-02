# Convolution activity

The convolution layer in PyTorch has several arguments that controls its behaviour, such as `in_channels`, `out_channels`, `kernel_size`, `stride`, and `padding`. When working with the convolution layer, it is important to understand how these arguments affect the shapes of your inputs. We can control the sizes of the output feature (channel) dimension as well as the spatial (height and width) dimensions. One of the most common issues when building your own neural network is getting these input and output shapes to be compatible when composing multiple layers.

When working with image data for deep learning, we represent it as a 4-dimensional tensor. In PyTorch, the convention is to have the dimensions in the order of (batch, channels, height, width) = (BCHW). 

You could think of an image as an array of pixels, that have a spatial shape given by H and W. This image has C channels, each with spatial shape H $\times$ W. We can think of all the pixels in a single channel as a "feature map". Feature maps are like RGB channels in typical images, although channels tend to lose a concrete meaning as we process images through a neural network. It can also be useful to think of images as a collection of C-dimensional feature vectors at every pixel location. 

The batch dimension represents a collection of many such of these CHW images. In practice, we batch process our inputs when running neural networks, since modern hardware such as graphical processing units (GPUs) are very efficient at doing batched tensor operations like convolution and linear layers.

For this activity, you will create convolution layers to perform the desired operations. Note that all inputs and outputs have shape (batch, channels, height, width).

Relevant documentation:
- [PyTorch nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)


```python
# Imports
import torch
from torch import nn
```

## 1. Channels

Create a convolution layer with a 3x3 kernel that transforms the 3-channel input to a 16-channel output. Ensure that the output  spatial dimensions (H and W) are the same as the input spatial dimensions.



```python
x = torch.randn(4, 3, 16, 16)
expected_size = torch.Size((4, 16, 16, 16))
# TODO: Create a convolution layer with the right arguments
conv = None
#####
y = conv(x)
assert y.shape == expected_size, f"Expected size {expected_size}, got size {y.shape}"
```

## 2. Padding

Create a convolution layer with a 5x5 kernel that has the same number of input channels as output channels, with "valid" padding. What is the output size after processing the input? Explain why the spatial dimensions changed.

Answer:



```python
x = torch.randn(4, 16, 16, 16)
# TODO: Create a convolution layer with the right arguments
conv1 = None
#####
y = conv1(x)
y.shape
```

## 3. Stride

Create a strided convolution layer with a 5x5 kernel that has the same number of input channels as output channels, that exactly halves the spatial dimensions. If we used a 3x3 kernel size instead, how much padding should be added to ensure that this layer exactly halves the output spatial dimensions?

Answer:


```python
x = torch.randn(4, 3, 64, 64)
expected_size = torch.Size((4, 3, 32, 32))
# TODO: Create the convolution layer.
conv = None
#####
y = conv(x)
assert y.shape == expected_size, f"Expected size {expected_size}, got size {y.shape}"
```

## 4. Combination

Using combinations of what you used above, create a convolution layer with a 3x3 kernel that transforms the input to the expected output shape.


```python
x = torch.randn(4, 3, 32, 32)
expected_size = torch.Size((4, 16, 16, 16))
# TODO: Create the convlution layer.
conv = None
#####
y = conv(x)
assert y.shape == expected_size, f"Expected size {expected_size}, got size {y.shape}"
```
