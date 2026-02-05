import torch.nn as nn
import torch.nn.functional as F

class MLPBackbone(nn.Module):
    def __init__(self, input_dim=64, embedding_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, embedding_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization for ArcFace
        return x