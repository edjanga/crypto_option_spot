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
                 feature_names: List[str], dropout: float, beta: float = None, output_size: int = 1,
                 positiveness: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.beta = beta
        self.positiveness = positiveness
        self.feature_names = feature_names
        self.dropout = dropout
        self.input_size = input_size
        lstm_output_size = hidden_size
        if torch.cuda.is_available():
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, dropout=self.dropout if num_layers > 1 else 0.0,
                                proj_size=hidden_size // 2)
            lstm_output_size /= 2
        else:
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, dropout=self.dropout if num_layers > 1 else 0.0)
        self.layer_norm = nn.LayerNorm([lstm_output_size])
        self.fc = nn.Linear(lstm_output_size, output_size)
        self.criterion = nn.MSELoss(reduction='mean')
        self.train_loss_curve = []
        self.valid_loss_curve = []
        self.save_hyperparameters()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.fc(out)
        if self.positiveness:
            out = F.softplus(out, beta=self.beta)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx % 10 == 0:
            self.train_loss_curve.append(loss.detach().cpu().item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx % 10 == 0:
            self.valid_loss_curve.append(loss.detach().cpu().item())
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'valid_loss', 'interval': 'epoch', 'frequency': 1}}
