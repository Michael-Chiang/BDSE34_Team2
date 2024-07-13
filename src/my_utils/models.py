# LSTM model
import torch
import torch.nn as nn

# Set device
use_cuda = 1
device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, seq_length):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_length
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 50),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=50),
            nn.Dropout(p=0.5),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=10),
            nn.Dropout(p=0.5),
            nn.Linear(10, output_dim),
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        # out = out.contiguous().view(x.size(0), -1)
        out = self.fc(out[:, -1, :])
        return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        pool_kernel_size=2,
        dropout_rate=0.5,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        self.bn = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout1d(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ResConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        pool_kernel_size=2,
        dropout_rate=0.5,
    ):
        super(ResConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        self.bn = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout1d(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x_skip + x
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels, output_dim, num_features, dropout_rate):
        super(CNN, self).__init__()
        self.block1 = ConvBlock(
            in_channels=in_channels, out_channels=128, dropout_rate=dropout_rate
        )  # 第一层卷积模块
        self.block2 = ConvBlock(
            in_channels=128, out_channels=256, dropout_rate=dropout_rate
        )  # 第二层卷积模块
        self.block3 = ConvBlock(
            in_channels=256, out_channels=512, dropout_rate=dropout_rate
        )  # 第三层卷积模块
        self.final_length = num_features // (2**3)
        self.fc = nn.Linear(
            in_features=512 * self.final_length, out_features=output_dim
        )  # 全连接层

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x
