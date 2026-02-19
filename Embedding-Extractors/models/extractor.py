from .fingerprint_vit import FingerViT
from .measurements_1d_cnn import MeasurementsCNN
from .capacitive_mlp import CapacitiveMLP
from .arcface import ArcFace

import torch.nn as nn

class Extractor(nn.Module):
    def __init__(self, modality, num_classes):
        super().__init__()
        if modality == "fingerprint":
            EMB_DIM = 64
            self.backbone = FingerViT(emb_dim=EMB_DIM)
        elif modality == "measurements":
            EMB_DIM = 128
            self.backbone = MeasurementsCNN(emb_dim=EMB_DIM)
        elif modality == "capacitive":
            EMB_DIM = 32
            self.backbone = CapacitiveMLP(emb_dim=EMB_DIM)
        self.arcface = ArcFace(EMB_DIM, num_classes)
    
    def forward(self, x, labels=None):
        embeddings = self.backbone(x)
        if labels is not None:
            # Training: return ArcFace logits
            return self.arcface(embeddings, labels)
        # Inference: return raw embeddings
        return embeddings