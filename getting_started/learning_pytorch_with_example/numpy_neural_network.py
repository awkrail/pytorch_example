import numpy as np

# N : batch_size
# D_in : input_dimension
# H : hidden_size
# D_out : output_dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create Random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

for t in range(500):
  h = x.dot(w1)
  h_relu = np.maximum(h, 0)
  y_pred = h_relu.dot(w2)

  # Compute and print loss
  loss = np.square(y_pred - y).sum()
  print(t, loss)

  # BackProp to compute gradients of W1 and W2
  grad_y_pred = 2.0 * (y_pred - y)
  grad_w2 = h_relu.T.dot(grad_y_pred)
  grad_h_relu = grad_y_pred.dot(w2.T)
  grad_h = grad_h_relu.copy()
  grad_h[h < 0] = 0
  grad_w1 = x.T.dot(grad_h)

  # Update weights
  w1 -= learning_rate * grad_w1
  w2 -= learning_rate * grad_w2