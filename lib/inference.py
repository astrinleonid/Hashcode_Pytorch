import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from definitions import Conv1DModel, read_dataset
from torch.utils.data import TensorDataset, DataLoader

def train_model():
    file_x = 'X_toy.csv'
    file_y = 'y_toy.csv'

    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = read_dataset(file_x, file_y, train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    del X_train_tensor

    X_test_tensor.to(device)
    y_test_tensor.to(device)

    print(device)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    input_shape = (1, 128)  # Note the change in order of dimensions for PyTorch
    model = Conv1DModel(input_shape)
    model.load_state_dict(torch.load("best_model.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test_tensor = X_test_tensor.to(device)
    model.to(device)