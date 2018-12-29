import torch
import numpy as np
import matplotlib.pyplot as plt

a = torch.tensor([0,1,2,3,4])

print(a.dtype)
a = a.type(torch.FloatTensor)

print(a.type())
print(a.size())
print(a.ndimension())

print(a)

# put a tensor into a column
a_col = a.view(-1,1)

print(a_col)

numpyArray = np.array([0.0,1.0,2.0,3.0,4.0])

print(numpyArray)

# cast a torch tensor from numpy array
torchTensor = torch.from_numpy(numpyArray)
print(torchTensor)

# create a sequence of numbers (the larger the step size, the lower the increment)
x = torch.linspace(-2,2,steps=5)
print(x)

x = torch.linspace(-2,2,steps=9)
print(x)

x = torch.linspace(0, 2*np.pi, 100)
y = torch.sin(x)

import matplotlib.pyplot as plt

# cast x and y to a numpy array
plt.plot(x.numpy(), y.numpy())