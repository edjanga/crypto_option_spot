import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List
import pytorch_lightning as p_light
import torch.nn.functional as F


class TimeSeriesDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return self.X[index:index + self.seq_len], self.y[index + self.seq_len - 1]


class LSTMModel(p_light.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, learning_rate: float,
                 feature_names: List[str], beta: float = None, output_size: int = 1, positiveness: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.beta = beta
        self.positiveness = positiveness
        self.feature_names = feature_names
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.train_loss_curve = []
        self.valid_loss_curve = []

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        if self.positiveness:
            out = F.softplus(out, beta=self.beta)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        self.train_loss_curve.append(loss.detach().cpu().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('valid_loss', loss)
        self.valid_loss_curve.append(loss.detach().cpu().numpy())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
