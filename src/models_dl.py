import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. TIME-SERIES SLIDING WINDOW DATASET
# ==========================================
class TimeSeriesDataset(Dataset):
    """
    Converts 2D (Samples, Features) arrays into 3D overlapping windows.
    Required for LSTM and 1D-CNN models.
    """
    def __init__(self, X, y, window_size):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        # Extract a window of data
        x_window = self.X[idx : idx + self.window_size]
        # The label is the target value at the end of the window
        y_label = self.y[idx + self.window_size - 1] 
        return x_window, y_label


# ==========================================
# 2. LONG SHORT-TERM MEMORY (LSTM) MODEL
# ==========================================
class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(LSTMAnomalyDetector, self).__init__()
        
        # batch_first=True means inputs are (Batch, Seq_len, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # We only care about the prediction at the final time step of the window
        last_step_out = lstm_out[:, -1, :] 
        
        out = self.dropout(last_step_out)
        out = self.fc(out)
        return self.sigmoid(out).squeeze()


# ==========================================
# 3. 1D CONVOLUTIONAL NEURAL NETWORK
# ==========================================
class CNN1DAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(CNN1DAnomalyDetector, self).__init__()
        
        # In PyTorch CNNs, the channels are the features. 
        # Expected input shape for Conv1d: (Batch, Channels/Features, Seq_len)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Permute from (Batch, Seq_len, Features) -> (Batch, Features, Seq_len)
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu(x)
        
        # Global Max Pooling across the time dimension
        x, _ = torch.max(x, dim=2) 
        
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x).squeeze()