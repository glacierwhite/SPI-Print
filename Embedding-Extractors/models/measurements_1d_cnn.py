import torch
import torch.nn as nn
import torch.nn.functional as F

class MeasurementsCNN(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),

            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),

            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),

            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
        )

        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):   # x: (B,1,256)
        x = self.net(x)
        x = x.mean(dim=2)   # global average pooling
        x = self.fc(x)
        return F.normalize(x)