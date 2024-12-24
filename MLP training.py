import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    '''Model to regress 2d time series values given scalar input.'''
    def __init__(self):
      # the builder with all member fields required
      # we take time and predict the values of x,y at that time
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # first fully connected layer
        self.fc2 = nn.Linear(64, 128) # second layer
        self.fc3 = nn.Linear(128, 64) #third layer
        self.fc4 = nn.Linear(64, 2)  #final layer which of course has to output 2 values
        self.relu = nn.ReLU()        # as learned from ML courses, we have to use a non-linear function on each layer (there's a lot more options)

    def forward(self, x):   # this is literally the implementation of the self.relu explanation above
                            # we activate the relu function between each layer when forwarding the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class TimeSeriesDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.t = data['t'].values.reshape(-1, 1).astype(np.float32)  # Time values
        self.x = data['x'].replace('-', np.nan).astype(np.float32)  # X values (replace '-' with NaN)
        self.y = data['y'].replace('-', np.nan).astype(np.float32)  # Y values (replace '-' with NaN)
        self.x = self.x.interpolate(limit_direction='both')  # Interpolate missing x values - 'smooth it' by making it continous
        self.y = self.y.interpolate(limit_direction='both')  # Interpolate missing y values - 'smooth it' by making it continous
        self.labels = np.stack([self.x, self.y], axis=1)  # Combine x and y

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        t = self.t[idx]  # Input: time
        label = self.labels[idx]  # Output: (x, y)
        return torch.tensor(t, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

import pandas as pd
csv_url = 'https://raw.githubusercontent.com/guyoron1/data/main/data.csv'
# data = pd.read_csv(csv_url)

dataset = TimeSeriesDataset(csv_url)
# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

def loss_fn(outputs, labels):
    return F.mse_loss(outputs, labels)  # Mean Squared Error for regression

# Prepare Dataset and DataLoader
dataset = TimeSeriesDataset(csv_url)
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Initialize the model, optimizer, and loss function
net = Net()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

for epoch in range(300):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:  # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')


# Plot results to validate training
# Plot Results
net.eval()
t_values = dataset.t
predicted = net(torch.tensor(t_values, dtype=torch.float32)).detach().numpy()
true_labels = dataset.labels

plt.figure(figsize=(10, 6))
plt.plot(t_values, true_labels[:, 0], label='True x(t)', alpha=0.6)
plt.plot(t_values, true_labels[:, 1], label='True y(t)', alpha=0.6)
plt.plot(t_values, predicted[:, 0], label='Predicted x(t)', linestyle='dashed')
plt.plot(t_values, predicted[:, 1], label='Predicted y(t)', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.title('Bivariate Time Series Regression')
plt.show()