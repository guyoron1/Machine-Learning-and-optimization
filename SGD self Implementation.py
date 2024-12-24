import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_data():
  data = torch.rand(1000, 2)
  label = ((data[:,0]+0.3*data[:,1]) > 0.5).to(torch.int)
  return data[:,0], label

# this is a basic implementation of the SGD, with mini batching.
# there implementation is very simple, following the SGD algorithm using tensor operations alone.

input, label = generate_data()

# Make minibatches.
inputs = torch.split(input, 32)
labels = torch.split(label, 32)

# Define the two variables to optimize
b1 = torch.autograd.Variable(torch.tensor([0.01]), requires_grad=True)
b2 = torch.autograd.Variable(torch.tensor([0.01]), requires_grad=True)

alpha = 0.1 # we fix a learning rate - not too large and not too small, to make our optimization process accurate
for epoch in range(15):
  total_loss = 0
  for x, y in zip(inputs,labels):
        # Calculate p(x) = 1 / (1 + exp(-(b1 + b2 * x)))
        p_x = 1 / (1 + torch.exp(-(b1 + b2 * x)))

        # Calculate the negative log likelihood loss - this is a cross-entropy loss :)
        loss = -torch.mean(y * torch.log(p_x + 1e-8) + (1 - y) * torch.log(1 - p_x + 1e-8))

        # Calculate the gradient of the loss w.r.t. the parameters
        loss.backward()

        # Update parameters using SGD formula
        with torch.no_grad():
            b1 -= alpha * b1.grad
            b2 -= alpha * b2.grad

            # Zerofy the gradients after updating, as expected from the algorithm
            # since we run epoch's / batches
            b1.grad.zero_()
            b2.grad.zero_()

        # we sum the loss for each epoch, and will print at the end the average loss
        total_loss += loss.item()

  print(f"Epoch {epoch + 1}, Loss: {total_loss / len(inputs)}")
# Plotting results
x_vals = torch.linspace(0, 1, 100)
y_vals = 1 / (1 + torch.exp(-(b1.detach() + b2.detach() * x_vals)))

plt.figure(figsize=(8, 6))
plt.scatter(input.numpy(), label.numpy(), c=label.numpy(), cmap='coolwarm', label='True Data')
plt.plot(x_vals.numpy(), y_vals.numpy(), color='blue', label='Logistic Decision Boundary')
plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Logistic Regression with SGD")
plt.legend()
plt.show()
