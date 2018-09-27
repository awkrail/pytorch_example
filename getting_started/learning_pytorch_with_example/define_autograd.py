import torch


class MyReLU(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
    """
    ctx.save_for_backward(input)
    return input.clamp(min=0)
  
  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input

dtype = torch.float
device = torch.device("cpu")

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(500):
  relu = MyReLU.apply
  y_pred = relu(x.mm(w1)).mm(w2)

  # Compute and print loss
  loss = (y_pred - y).pow(2).sum()
  print(t, loss.item())

  # Backprop to compute gradients
  loss.backward()

  # Update Weights
  with torch.no_grad():
    w1 -= learning_rate * w1.grad
    w2 -= learning_rate * w2.grad

    w1.grad.zero_()
    w2.grad.zero_()