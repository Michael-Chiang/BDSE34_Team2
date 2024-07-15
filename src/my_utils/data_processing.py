import logging

import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Data transformation functions


def transform_type(x, device, is_train=True):
    tensor = torch.Tensor(x.astype(float)).to(device)
    return tensor if is_train else tensor.to(torch.int64)


def split_data(stock, lookback, interval, y, input_dim):
    data_raw = np.array(stock)
    n_time = len(data_raw)
    data, targets = [], []
    for index in range(0, n_time - lookback, interval):
        data.append(data_raw[index: index + lookback, :input_dim])
        targets.append(y.iloc[index + lookback])

    data = np.array(data)
    targets = np.array(targets)
    logging.info(f"Total data samples: {data.shape}")

    # Split training and testing data
    x_train, x_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.2, shuffle=True, random_state=42
    )
    return x_train, y_train, x_test, y_test


def prepare_data(file_paths, lookback, interval, period, input_dim, device):
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
            data_, lookback, interval, ten_day_change_fixed_discrete, input_dim
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


def prepare_prediction_data(file_paths, lookback, input_dim, device):
    data_x = []
    file_names = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]

        data = pd.read_csv(file_path)
        scaler = MinMaxScaler()
        data_ = scaler.fit_transform(data.iloc[:, 1:].values)

        data_raw = np.array(data_)
        n_time = len(data_raw)

        file_names.append(file_name_without_ext)
        data_x.append(data_raw[n_time - lookback: n_time, :input_dim])

    file_names = np.array(file_names)
    data_x = np.array(data_x)
    x_train_ = transform_type(data_x, device)

    logging.info(f"x_train_.shape = {x_train_.shape}")
    return file_names, x_train_
