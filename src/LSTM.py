import copy
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Set device
use_cuda = 1
device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
print(device)


# Data transformation functions
def transform_type(x, device, is_train=True):
    tensor = torch.Tensor(x.astype(float)).to(device)
    return tensor if is_train else tensor.to(torch.int64)


def split_data(stock, lookback, interval, y):
    data_raw = np.array(stock)
    n_time = len(data_raw)
    data, targets = [], []
    for index in range(0, n_time - lookback, interval):
        data.append(data_raw[index : index + lookback, :])
        targets.append(y.iloc[index + lookback])

    data = np.array(data)
    targets = np.array(targets)
    print("Total data samples:{}".format(data.shape))

    # Split training and testing data
    x_train, x_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.2, shuffle=True, random_state=42
    )
    return x_train, y_train, x_test, y_test


def train(
    model, num_epochs, train_dl, valid_dl, device, criterion, optimizer, scheduler
):
    loss_hist_train = []
    accuracy_hist_train = []
    loss_hist_valid = []
    accuracy_hist_valid = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        batch_num = 0

        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            epoch_accuracy += is_correct.sum().item()
            batch_num += 1

        epoch_loss /= len(train_dl.dataset)
        epoch_accuracy /= len(train_dl.dataset)
        loss_hist_train.append(epoch_loss)
        accuracy_hist_train.append(epoch_accuracy)

        scheduler.step()
        model.eval()
        valid_loss = 0.0
        valid_accuracy = 0.0

        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                valid_loss += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                valid_accuracy += is_correct.sum().item()

        valid_loss /= len(valid_dl.dataset)
        valid_accuracy /= len(valid_dl.dataset)
        loss_hist_valid.append(valid_loss)
        accuracy_hist_valid.append(valid_accuracy)

        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}:   Train accuracy: {epoch_accuracy:.4f}    Validation accuracy: {valid_accuracy:.4f} "
            )
            print(
                f"Epoch {epoch}:   Train loss: {epoch_loss:.4f}    Validation loss: {valid_loss:.4f} "
            )

    model.load_state_dict(best_model_wts)

    history = {
        "loss_hist_train": loss_hist_train,
        "loss_hist_valid": loss_hist_valid,
        "accuracy_hist_train": accuracy_hist_train,
        "accuracy_hist_valid": accuracy_hist_valid,
    }
    return model, history


def show_accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        x_pred = model(x).detach().cpu().numpy()
    is_correct = (np.argmax(x_pred, axis=1) == y.cpu().numpy()).sum() / len(x_pred)
    return is_correct


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 20),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(20, output_dim),
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path, input_dim, hidden_dim, num_layers, output_dim):
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    model.load_state_dict(torch.load(path))
    model.to(device)
    print(f"Model loaded from {path}")
    return model


def prepare_data(file_paths, lookback, interval, period):
    x_train_all = None
    y_train_all = None
    x_test_all = None
    y_test_all = None
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        scaler = StandardScaler()
        data_ = scaler.fit_transform(data.iloc[:, 1:].values)

        ten_day_change = data["Close"].pct_change(periods=period) * 100
        fixed_bins = [-float("inf"), -10, -2, 2, 10, float("inf")]
        fixed_labels = [0, 1, 2, 3, 4]
        ten_day_change_fixed_discrete = pd.cut(
            ten_day_change, bins=fixed_bins, labels=fixed_labels
        )

        x_train, y_train, x_test, y_test = split_data(
            data_, lookback, interval, ten_day_change_fixed_discrete
        )
        print("x_train.shape =", x_train.shape)
        print("y_train.shape =", y_train.shape)
        print("x_test.shape =", x_test.shape)
        print("y_test.shape =", y_test.shape)
        if x_train_all is None:
            x_train_all = np.copy(x_train)
            y_train_all = np.copy(y_train)
            x_test_all = np.copy(x_test)
            y_test_all = np.copy(y_test)
        else:
            x_train_all = np.concatenate((x_train_all, x_train), axis=0)
            y_train_all = np.concatenate((y_train_all, y_train), axis=0)
            x_test_all = np.concatenate((x_test_all, x_test), axis=0)
            y_test_all = np.concatenate((y_test_all, y_test), axis=0)

    x_train_ = transform_type(x_train_all, device)
    x_test_ = transform_type(x_test_all, device)
    y_train_ = transform_type(y_train_all, device, is_train=False)
    y_test_ = transform_type(y_test_all, device, is_train=False)

    return x_train_, y_train_, x_test_, y_test_


def main():
    # Parameters
    input_dim = 71  # Number of features
    hidden_dim = 100
    num_layers = 7
    output_dim = 5
    batch_size = 8
    num_epochs = 100
    lr = 0.0001
    lookback = 60  # Sequence length
    interval = 10  # sample days difference
    period = 10  # predicted days after
    sector = "Finance"
    clusterID = "0"
    file_paths = glob.glob(os.path.join("stock_data_all", sector, clusterID, "*.csv"))
    file_path = f"stock_data_all/{sector}/{clusterID}/ABCB.csv"
    model_save_path = "model/best_lstm_model.pth"

    # Prepare data
    x_train_, y_train_, x_test_, y_test_ = prepare_data(
        file_paths, lookback, interval, period
    )

    # Data loaders
    train_ds = TensorDataset(x_train_, y_train_)
    test_ds = TensorDataset(x_test_, y_test_)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = LSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Train model
    start_time = time.time()
    best_model, hist = train(
        model, num_epochs, train_dl, test_dl, device, criterion, optimizer, scheduler
    )
    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))
    print(
        "Training accuracy: {:.4f}".format(
            show_accuracy(best_model, x_train_, y_train_)
        )
    )
    print(
        "Testing accuracy: {:.4f}".format(show_accuracy(best_model, x_test_, y_test_))
    )

    # Save model
    save_model(best_model, model_save_path)
    load_model(model_save_path, input_dim, hidden_dim, num_layers, output_dim)


if __name__ == "__main__":
    main()
