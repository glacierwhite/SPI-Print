import torch.nn as nn
import torch.nn.functional as F

class CapacitiveMLP(nn.Module):
    def __init__(self, input_dim=64, emb_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, emb_dim)
        
    def forward(self, x):
        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization for ArcFace
        return x