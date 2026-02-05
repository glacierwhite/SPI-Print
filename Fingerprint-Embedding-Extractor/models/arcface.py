import torch
import torch.nn as nn

class ArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.W)
        self.s = s
        self.m = m

    def forward(self, emb, labels):
        W = nn.functional.normalize(self.W)
        cos = torch.matmul(emb, W.t())

        theta = torch.acos(torch.clamp(cos, -1+1e-7, 1-1e-7))
        cos_m = torch.cos(theta + self.m)

        onehot = torch.zeros_like(cos)
        onehot.scatter_(1, labels.view(-1,1), 1.0)

        logits = self.s * (onehot * cos_m + (1 - onehot) * cos)
        return logits