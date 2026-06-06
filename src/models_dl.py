import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Converts 2D (samples, features) arrays into 3D overlapping windows.
    Required for LSTM and 1D-CNN models.
    """

    def __init__(self, X, y, window_size):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window_size = int(window_size)

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        x_window = self.X[idx: idx + self.window_size]
        y_label = self.y[idx + self.window_size - 1]
        return x_window, y_label


class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]

        out = self.dropout(last_step_out)
        out = self.fc(out)

        return self.sigmoid(out).view(-1)


class CNN1DAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu(x)

        x, _ = torch.max(x, dim=2)

        x = self.dropout(x)
        x = self.fc(x)

        return self.sigmoid(x).view(-1)