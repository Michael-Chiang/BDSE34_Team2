# LSTM model
import torch
import torch.nn as nn

# Set device
use_cuda = 1
device = torch.device("cuda" if (
    torch.cuda.is_available() & use_cuda) else "cpu")


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
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        # out = out.contiguous().view(x.size(0), -1)
        out = self.fc(out[:, -1, :])
        return out


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=0.5, bidirectional=True)
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
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(device)

        # Forward propagate the GRU
        out, _ = self.gru(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class GRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU_Attention, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=0.5, bidirectional=True)
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

    def attention_net(self, gru_output, final_state):
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat(
            (final_state[0], final_state[1]), dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(gru_output, hidden).squeeze(2)
        # attn_weights : [batch_size,n_step]
        soft_attn_weights = nn.functional.softmax(attn_weights, 1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(gru_output.transpose(
            1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context, soft_attn_weights

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(device)

        # Forward propagate the GRU
        out, final_hidden_state = self.gru(x, h0)

        # Decode the hidden state of the last time step
        attn_output, attention = self.attention_net(out, final_hidden_state)
        out = self.fc(attn_output)
        return out, attention


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        pool_kernel_size=2,
        dropout_rate=0.5,
        pool=False,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.max_pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        self.bn = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout1d(p=dropout_rate)
        self.pool = pool
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        if self.pool:
            x = self.max_pool(x)
        x = self.dropout(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels, output_dim, num_features, dropout_rate):
        super(CNN, self).__init__()
        self.block1 = ConvBlock(
            in_channels=in_channels, out_channels=128, dropout_rate=dropout_rate, pool=True
        )
        self.block2 = self._make_layers(128, 256, 3)
        self.block3 = self._make_layers(256, 512, 3)
        self.final_length = num_features // (2**3)
        self.fc = nn.Linear(
            in_features=512 * self.final_length, out_features=output_dim
        )  # 全连接层

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(ConvBlock(in_channels, out_channels))
        for _ in range(1, block_num - 1):
            layers.append(ConvBlock(out_channels, out_channels))
        layers.append(ConvBlock(out_channels, out_channels, pool=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x
