import copy
import glob
import logging
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from pydantic.v1 import BaseModel, Field, conint, root_validator
from typing import Literal
import pydantic_argparse

from my_utils.models import LSTM


class Config(BaseModel):
    input_dim: int = Field(71, description="Number of features")
    hidden_dim: int = Field(100, description="Hidden layer dimension")
    num_layers: conint(ge=1, le=5) = Field(  # type: ignore
        2, description="Number of LSTM layers")
    output_dim: int = Field(5, description="Output dimension")
    batch_size: int = Field(64, description="Batch size")
    num_epochs: int = Field(200, description="Number of epochs")
    lr: float = Field(0.0001, description="Learning rate")
    patience: int = Field(20, description="Early stopping patience")
    step_size: int = Field(20, description="Learning rate scheduler step size")
    gamma: float = Field(
        0.8, description="Learning rate scheduler weight decay")

    sector: Literal['Finance', 'Technology'] = Field(
        description="The name must be 'Finance' or 'Technology'")
    lookback: int = Field(None, description="Sequence length")
    interval: int = Field(None, description="Sample days difference")
    period: int = Field(None, description="Predicted days after")

    @root_validator(pre=True)
    def set_defaults(cls, values):
        name = values.get('sector')
        if name == 'Finance':
            values['lookback'] = 60
            values['interval'] = 10
            values['period'] = 10
        elif name == 'Technology':
            values['lookback'] = 20
            values['interval'] = 5
            values['period'] = 5
        return values
    clusterID: conint(ge=0, le=4) = Field(  # type: ignore
        description="Cluster ID")  # type: ignore


# Set device
use_cuda = 1
device = torch.device("cuda" if (
    torch.cuda.is_available() & use_cuda) else "cpu")
logging.info(f"Device: {device}")

# Data transformation functions


def transform_type(x, device, is_train=True):
    tensor = torch.Tensor(x.astype(float)).to(device)
    return tensor if is_train else tensor.to(torch.int64)


def split_data(stock, lookback, interval, y):
    data_raw = np.array(stock)
    n_time = len(data_raw)
    data, targets = [], []
    for index in range(0, n_time - lookback, interval):
        data.append(data_raw[index: index + lookback, :])
        targets.append(y.iloc[index + lookback])

    data = np.array(data)
    targets = np.array(targets)
    logging.info(f"Total data samples: {data.shape}")

    # Split training and testing data
    x_train, x_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.2, shuffle=True, random_state=42
    )
    return x_train, y_train, x_test, y_test


def discretization(pred):
    # 定义类别中心点
    class_centers = torch.tensor([0, 1, 2, 3, 4]).to(device)

    # 计算预测值与每个类别中心的距离
    distances = torch.abs(pred - class_centers.unsqueeze(0))

    # 找到距离最小的类别索引
    classes = torch.argmin(distances, dim=1)
    return classes


def train(
    model,
    num_epochs,
    patience,
    train_dl,
    valid_dl,
    device,
    criterion,
    optimizer,
    scheduler,
):
    loss_hist_train = []
    accuracy_hist_train = []
    loss_hist_valid = []
    accuracy_hist_valid = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_valid_loss = np.inf
    counter = 0

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
            # print(
            #     f"pred.shape = {pred.shape}, discretization(pred).shape = {discretization(pred).shape}, y_batch.shape = {y_batch.shape}"
            # )
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

        logging.info(
            f"Epoch {epoch}:   Train accuracy: {epoch_accuracy:.4f}    Validation accuracy: {valid_accuracy:.4f}"
        )
        logging.info(
            f"Epoch {epoch}:   Train loss: {epoch_loss:.4f}    Validation loss: {valid_loss:.4f}"
        )

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    model.load_state_dict(best_model_wts)

    history = {
        "loss_hist_train": loss_hist_train,
        "loss_hist_valid": loss_hist_valid,
        "accuracy_hist_train": accuracy_hist_train,
        "accuracy_hist_valid": accuracy_hist_valid,
    }
    return model, history


def show_accuracy(model, dl):
    model.eval()
    correct = 0
    size = len(dl.dataset)
    with torch.no_grad():
        for batch_x, batch_y in dl:
            pred_y = model(batch_x)
            correct += (
                (torch.argmax(pred_y, dim=1) == batch_y).type(
                    torch.float).sum().item()
            )
    return correct / size


def save_confusion_matrix(model, train_dl, test_dl, path):
    model.eval()
    correct = 0
    size = len(train_dl.dataset)
    all_batch_y = []
    all_batch_y_pred = []
    with torch.no_grad():
        for dl in [train_dl, test_dl]:
            for batch_x, batch_y in dl:
                pred_y = model(batch_x)
                correct += (
                    (torch.argmax(pred_y, dim=1) == batch_y)
                    .type(torch.float)
                    .sum()
                    .item()
                )
                all_batch_y.append(batch_y.cpu().numpy())
                all_batch_y_pred.append(
                    torch.argmax(pred_y, dim=1).cpu().numpy())

            if dl == train_dl:
                y_train = np.concatenate(all_batch_y)
                y_train_pred = np.concatenate(all_batch_y_pred)
            else:
                y_test = np.concatenate(all_batch_y)
                y_test_pred = np.concatenate(all_batch_y_pred)

    logging.info(f"train classification results")
    logging.info(f"{classification_report(y_train, y_train_pred)}")
    logging.info(f"test classification results")
    logging.info(f"{classification_report(y_test, y_test_pred)}")
    # Calculate confusion matrix
    train_cm = confusion_matrix(y_train, y_train_pred)
    train_cm = train_cm.astype("float") / train_cm.sum(axis=1)[:, np.newaxis]
    test_cm = confusion_matrix(y_test, y_test_pred)
    test_cm = test_cm.astype("float") / test_cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(train_cm, annot=True, fmt=".2f", cmap="Blues", ax=ax[0])
    ax[0].set_title("Train Confusion Matrix")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")

    sns.heatmap(test_cm, annot=True, fmt=".2f", cmap="Blues", ax=ax[1])
    ax[1].set_title("Test Confusion Matrix")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(path)


def save_model(model, path):
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")


def load_model(path, input_dim, hidden_dim, num_layers, output_dim, seq_length):
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim, seq_length)
    model.load_state_dict(torch.load(path))
    model.to(device)
    logging.info(f"Model loaded from {path}")
    return model


def prepare_data(file_paths, lookback, interval, period):
    x_train_all = None
    y_train_all = None
    x_test_all = None
    y_test_all = None
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        scaler = MinMaxScaler()
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
        logging.info(f"x_train.shape = {x_train.shape}")
        logging.info(f"y_train.shape = {y_train.shape}")
        logging.info(f"x_test.shape = {x_test.shape}")
        logging.info(f"y_test.shape = {y_test.shape}")
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

    logging.info(f"x_train_.shape = {x_train_.shape}")
    logging.info(f"y_train_.shape = {y_train_.shape}")
    logging.info(f"x_test_.shape = {x_test_.shape}")
    logging.info(f"y_test_.shape = {y_test_.shape}")
    return x_train_, y_train_, x_test_, y_test_


def main(config: Config) -> None:

    clusterID = str(config.clusterID)
    file_paths = glob.glob(os.path.join(
        "stock_data_all", config.sector, clusterID, "*.csv"))
    model_path = "model/"
    figure_path = "figure/"
    model_save_path = f"model/best_lstm_model_{config.sector}_{clusterID}.pth"
    figure_save_path = f"figure/confusion_matrix_best_lstm_{config.sector}_{clusterID}.jpg"

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    # Prepare data
    x_train_, y_train_, x_test_, y_test_ = prepare_data(
        file_paths, config.lookback, config.interval, config.period
    )

    # Data loaders
    train_ds = TensorDataset(x_train_, y_train_)
    test_ds = TensorDataset(x_test_, y_test_)
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    model = LSTM(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_layers=config.num_layers,
        seq_length=config.lookback,
    ).to(device)
    logging.info(summary(model, (config.lookback, config.input_dim)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.step_size, gamma=config.gamma)

    # Train model
    start_time = time.time()
    best_model, hist = train(
        model,
        config.num_epochs,
        config.patience,
        train_dl,
        test_dl,
        device,
        criterion,
        optimizer,
        scheduler,
    )
    training_time = time.time() - start_time
    logging.info(f"Training time: {training_time}")
    logging.info(
        f"Training accuracy: {show_accuracy(best_model, train_dl):.4f}")
    logging.info(f"Testing accuracy: {show_accuracy(best_model, test_dl):.4f}")

    # Save model
    save_model(best_model, model_save_path)
    load_model(model_save_path, config.input_dim, config.hidden_dim,
               config.num_layers, config.output_dim, config.lookback)
    save_confusion_matrix(model, train_dl, test_dl, figure_save_path)


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=Config,
        prog="LSTM Training Program",
        description="Train LSTM model with given configuration",
    )
    config = parser.parse_typed_args()

    # Get current time for log file name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, f"log_LSTM_{current_time}_{config.sector}_{config.clusterID}.txt")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(log_filename),  # Log to file]
        ],
    )
    main(config)
