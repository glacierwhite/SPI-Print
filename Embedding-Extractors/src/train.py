import sys
if './' not in sys.path:
	sys.path.append('./')

import torch
import torch.nn as nn
from models.fingerprint_vit import FingerViT
from models.arcface import ArcFace
from dataset import FingerprintDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

## Hyperparmeters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 64
EPOCHS = 50
LR = 3e-4
EMB_DIM = 512

## Build data
train_set = FingerprintDataset("./data/train", train=True)
val_set   = FingerprintDataset("./data/val", train=False)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_set, batch_size=64, shuffle=False)

NUM_CLASSES = len(train_set.id_map)

## Build model
model = FingerViT(emb_dim=EMB_DIM).to(DEVICE)
arcface = ArcFace(EMB_DIM, NUM_CLASSES).to(DEVICE)

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(arcface.parameters()),
    lr=LR, weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss()

## Training Loop
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    arcface.train() if train else arcface.eval()

    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        if train:
            optimizer.zero_grad()

        emb = model(imgs)
        logits = arcface(emb, labels)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

## Train
for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)

    print(f"Epoch {epoch+1:02d} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "finger_vit.pth")

# ## Verification
# model.eval()

# img1, img2 = ...   # load two fingerprints
# e1 = model(img1.unsqueeze(0).to(DEVICE))
# e2 = model(img2.unsqueeze(0).to(DEVICE))

# sim = torch.cosine_similarity(e1, e2)
# print("Similarity:", sim.item())