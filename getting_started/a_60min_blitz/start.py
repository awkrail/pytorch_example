import torch


# Tensors
x = torch.empty(5, 3)
print(x)

# randn
x = torch.rand(5, 3)
print(x)

# zeros
x = torch.zeros(5, 3)
print(x)

# tensor
x = torch.tensor([5.5, 3])
print(x)

# randn_like
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

# get size
print(x.size())


# Operations
y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y) # => this operation mutates y's tensor.

# Resize(Reshape)
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8) # -1 is inferred from other dimension
print(x.size(), y.size(), z.size())

# item()
x = torch.rand(1)
print(x)
print(x.item())

# convert torch tensor to numpy array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# convert numpy array to tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)