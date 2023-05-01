import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.utils.data import TensorDataset, DataLoader
from definitions import Conv1DModel, read_dataset
from tqdm import tqdm

file_x = 'X_toy.csv'
file_y = 'y_toy.csv'

def train_model():

    file_x = 'X_toy.csv'
    file_y = 'y_toy.csv'
    
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor  =   read_dataset(file_x, file_y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor.to(device)
    X_test_tensor.to(device)
    y_train_tensor.to(device)
    y_test_tensor.to(device)

    print(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    input_shape = (1, 128)  # Note the change in order of dimensions for PyTorch
    model = Conv1DModel(input_shape)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = nn.BCELoss()

    num_epochs = 10
    patience = 3
    best_val_loss = float('inf')
    counter = 0

    model.to(device)

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_losses = []
        for inputs, targets in tqdm(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            clip_grad_value_(model.parameters(), clip_value=0.5)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_losses.append(loss.item())

        # Calculate average losses and print results
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == '__main__':
    train_model()