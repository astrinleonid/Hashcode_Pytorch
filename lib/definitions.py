import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

class Conv1DModel(nn.Module):
    def __init__(self, input_shape):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[0], 64, kernel_size=64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * ((input_shape[1] - 63) // 2),
                             64)  # Update the value in Linear layer based on input_shape
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def read_dataset(file_x, file_y, train = True):
    X = pd.read_csv(file_x)
    y = pd.read_csv(file_y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    del X
    del y

    if train:

        X_train_np = X_train.to_numpy(dtype=np.float32)  # Convert DataFrame to NumPy array
        X_train_tensor = torch.tensor(X_train_np)  # Convert NumPy array to PyTorch tensor
        X_train_tensor = X_train_tensor.unsqueeze(1)
        y_train_np = y_train.to_numpy(dtype=np.float32)  # Convert to NumPy array
        y_train_tensor = torch.tensor(y_train_np)  # Convert NumPy array to PyTorch tensor

        del y_train
        del y_train_np
        del X_train
        del X_train_np

    X_test_np = X_test.to_numpy(dtype=np.float32)  # Convert DataFrame to NumPy array
    X_test_tensor = torch.tensor(X_test_np)  # Convert NumPy array to PyTorch tensor
    X_test_tensor = X_test_tensor.unsqueeze(1)

    del X_test
    del X_test_np

    y_test_np = y_test.to_numpy(dtype=np.float32)  # Convert to NumPy array
    y_test_tensor = torch.tensor(y_test_np)  # Convert NumPy array to PyTorch tensor

    if train:
        return (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
    else:
        return (X_test_tensor, y_test_tensor)
