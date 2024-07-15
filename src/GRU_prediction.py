import glob
import logging
import os
from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic_argparse
import seaborn as sns
import torch
import torch.nn as nn
from pydantic.v1 import BaseModel, Field, conint, root_validator
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from my_utils.data_processing import prepare_prediction_data
from my_utils.models import GRU


class Config(BaseModel):
    input_dim: int = Field(71, description="Number of features")
    hidden_dim: int = Field(100, description="Hidden layer dimension")
    num_layers: conint(ge=1, le=5) = Field(  # type: ignore
        2, description="Number of GRU layers"
    )
    output_dim: int = Field(5, description="Output dimension")
    batch_size: int = Field(64, description="Batch size")
    num_epochs: int = Field(200, description="Number of epochs")
    lr: float = Field(0.0001, description="Learning rate")
    patience: int = Field(30, description="Early stopping patience")
    step_size: int = Field(20, description="Learning rate scheduler step size")
    gamma: float = Field(
        0.8, description="Learning rate scheduler weight decay")

    sector: Literal["Finance", "Technology"] = Field(
        description="The name must be 'Finance' or 'Technology'"
    )
    lookback: int = Field(None, description="Sequence length")
    interval: int = Field(None, description="Sample days difference")
    period: int = Field(None, description="Predicted days after")

    @root_validator(pre=True)
    def set_defaults(cls, values):
        name = values.get("sector")
        if name == "Finance":
            values["lookback"] = 60
            values["interval"] = 10
            values["period"] = 10
        elif name == "Technology":
            values["lookback"] = 20
            values["interval"] = 5
            values["period"] = 5
        return values

    clusterID: conint(ge=0, le=4) = Field(  # type: ignore
        description="Cluster ID"
    )  # type: ignore


# Set device
use_cuda = 1
device = torch.device("cuda" if (
    torch.cuda.is_available() & use_cuda) else "cpu")
logging.info(f"Device: {device}")


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


def load_model(path, input_dim, hidden_dim, num_layers, output_dim):
    model = GRU(input_dim, hidden_dim, num_layers, output_dim)
    model.load_state_dict(torch.load(path))
    model.to(device)
    logging.info(f"Model loaded from {path}")
    return model


def main(config: Config) -> None:

    clusterID = str(config.clusterID)
    file_paths = glob.glob(
        os.path.join("stock_data_all_new", config.sector, clusterID, "*.csv")
    )
    model_path = "model/"
    figure_path = "figure/"
    model_save_path = f"model/best_gru_model_{config.sector}_{clusterID}.pth"
    figure_save_path = (
        f"figure/confusion_matrix_best_gru_{config.sector}_{clusterID}.jpg"
    )
    output_path = f"output/{config.sector}/{clusterID}/"

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Prepare data
    file_names, x_train_ = prepare_prediction_data(
        file_paths,
        config.lookback,
        config.input_dim,
        device,
    )

    model = load_model(
        model_save_path,
        config.input_dim,
        config.hidden_dim,
        config.num_layers,
        config.output_dim,
    )
    logging.info(summary(model, (config.lookback, config.input_dim)))

    model.eval()
    with torch.no_grad():
        pred_y = model(x_train_)
        pred_y = torch.argmax(pred_y, dim=1).cpu().numpy()
    output = np.vstack((file_names, pred_y)).T
    # 将二维数组转换为 DataFrame
    df = pd.DataFrame(output, columns=['stockID', 'Prediction'])

    # 写入 CSV 文件
    df.to_csv(os.path.join(output_path, 'prediction.csv'), index=False)


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=Config,
        prog="GRU Training Program",
        description="Train GRU model with given configuration",
    )
    config = parser.parse_typed_args()

    # Get current time for log file name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_datetime = datetime.now().isoformat()
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, f"log_GRU_{current_time}_{config.sector}_{config.clusterID}.txt"
    )

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
