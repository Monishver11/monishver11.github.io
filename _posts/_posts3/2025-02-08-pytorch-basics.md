---
layout: post
title: PyTorch Basics for ML/DL
date: 2025-02-08 19:27:00-0400
featured: false
description: These are my notes from a YouTube tutorial on PyTorch basics. Please refer to the references section at the end for the tutorial link and the Colab notebook. I hope this serves as a solid starting point to build upon.
tags: ML Code
categories: RL-NYU
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---


PyTorch is a powerful deep learning framework that provides flexibility and ease of use. This post covers key PyTorch functionalities, including automatic differentiation, model training, data loading, and saving/loading models.

##### **Autograd: Automatic Differentiation**

The `torch.autograd` package provides automatic differentiation for all tensor operations. It acts as an engine for computing vector-Jacobian products and applies the chain rule for differentiation.

##### **Computing Gradients with Backpropagation**

Once we finish our computations, we can call `.backward()` to compute gradients automatically.

```python
import torch

# Create a tensor with requires_grad=True to track computation
x = torch.tensor(2.0, requires_grad=True)

# Compute a function of x
z = x ** 2 + 3 * x

# Compute gradients
z.backward()

# The gradient dz/dx is stored in x.grad
print(x.grad)  # Output: tensor(7.)
```
**Important Note:** `.backward()` accumulates gradients into the `.grad` attribute. If we call `.backward()` multiple times, the gradients are accumulated instead of overwritten. To avoid issues during optimization, we must use `optimizer.zero_grad()` before each backward pass.

##### **Stopping a Tensor from Tracking History**

Certain operations, like weight updates during training or evaluations, should not be part of the computation graph. We can prevent tracking history using:
```python
# Option 1: Using requires_grad_(False)
x.requires_grad_(False)

# Option 2: Detaching the tensor
x = x.detach()

# Option 3: Using no_grad() context manager
with torch.no_grad():
    w -= learning_rate * w.grad
```

##### **Building a Linear Regression Model**

PyTorch models should implement the forward pass inside a class that extends `nn.Module`.
```python
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

# Define model
input_size, output_size = 1, 1
model = LinearRegression(input_size, output_size)
```
##### **Understanding `nn.Module` and `super().__init__()`**

The call `super(LinearRegression, self).__init__()` (or simply `super().__init__()`) ensures that the parent class (`nn.Module`) is properly initialized. Without this call, the PyTorch module will not correctly register the layers, and key functionalities like `parameters()` will not work.

Think of `super().__init__()` as calling the constructor of `nn.Module`, ensuring that our model inherits all the capabilities of a standard PyTorch module.

**Do both `__init__` and `forward` need to be implemented?**

Yes, both methods are essential:

- **`__init__`**: Defines the model architecture (i.e., layers).
- **`forward`**: Implements the data flow through the network.

The `forward` method is automatically called when we pass data through the model
```python
`output = model(input_data)  # Calls model.forward(input_data)`
```
If `forward` is missing, the model won’t know how to process input data.

---

##### **Defining Loss Function and Optimizer**
```python
import torch.optim as optim

learning_rate = 0.01
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```
**How `model.parameters()` Works**

`model.parameters()` returns all learnable parameters (weights and biases) in the model.

**How `optimizer.step()` Updates Weights**

1. Computes gradients via `loss.backward()`
2. Updates each parameter using gradient descent:  
    $$w = w - \alpha \cdot \frac{\partial L}{\partial w}$$

**How `optimizer.zero_grad()` Works**

Clears gradients from previous steps to prevent accumulation across iterations.

##### **Training Loop**
```python
n_epochs = 100

for epoch in range(n_epochs):
    # Forward pass
    y_pred = model(X)

    # Compute loss
    loss = loss_fn(Y, y_pred)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()
    
    # Zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
```

##### **Data Loading with `DataLoader`**

Efficient data handling is crucial for training deep learning models. PyTorch provides `torch.utils.data.DataLoader` to load datasets in batches.

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
```

**How `DataLoader` Works:**

- **Automatic batching**: Instead of processing one sample at a time.
- **Shuffling**: Ensures better generalization by avoiding order bias.
- **Parallel loading**: Uses multiprocessing to speed up loading.

```python
# Fetch a batch of test data
examples = iter(test_loader)
example_data, example_targets = next(examples)
```
- `iter(test_loader)`: Converts `DataLoader` into an iterator.
- `next(examples)`: Fetches the next batch of data.

---

##### **Training a Neural Network**
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item():.4f}')
```

##### **Model Evaluation**
During testing, we do not compute gradients.
```python
with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

    acc = n_correct / n_samples
    print(f'Accuracy on test set: {100 * acc:.2f}%')
```

---

PyTorch provides `torchvision` for handling image data.
```python
import torchvision.transforms as transforms

# Normalize images to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```
**What Does This Transformation Do?**

- `transforms.ToTensor()`: Converts images to PyTorch tensors.
- `transforms.Normalize(mean, std)`: Normalizes pixel values.

##### **Saving and Loading Models**
```python
# Save model state
torch.save(model.state_dict(), "model.pth")

# Load model state
model.load_state_dict(torch.load("model.pth"))
```

`state_dict` is a dictionary containing all model parameters (weights & biases):

**Why use `state_dict()` instead of saving the whole model?**
-  It is lightweight and framework-independent.
- Provides flexibility (we can load into a modified model architecture).

---

##### **References**
- [PyTorch Crash Course - Getting Started with Deep Learning](https://www.youtube.com/watch?v=OIenNRt2bjg&t=157s)
- [PyTorch-CrashCourse.ipynb](https://colab.research.google.com/drive/1eiUBpmQ4m7Lbxqi2xth1jBaL61XTKdxp?usp=sharing)