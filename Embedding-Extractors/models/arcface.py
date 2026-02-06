import torch
import torch.nn as nn

class ArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, s=30.0, m=0.5, eps=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.W)
        
        self.s = s
        self.m = m
        self.eps = eps

    def forward(self, embeddings, labels):
        # normalize
        x = nn.functional.normalize(embeddings)
        W = nn.functional.normalize(self.W)

        # cosine
        logits = torch.matmul(x, W.t())  # [B, C]
        theta = torch.acos(torch.clamp(logits, -1+1e-7, 1-1e-7))

        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # # label smoothing
        # one_hot = (1 - self.eps) * one_hot + self.eps / logits.size(1)

        output = self.s * (one_hot * target_logits + (1 - one_hot) * logits)

        return output