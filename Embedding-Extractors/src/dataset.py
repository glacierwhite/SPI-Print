import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class FingerprintDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.samples = []
        self.id_map = {}

        for idx, person in enumerate(sorted(os.listdir(root))):
            pdir = os.path.join(root, person)
            if not os.path.isdir(pdir):
                continue
            self.id_map[person] = idx
            for img in os.listdir(pdir):
                self.samples.append((os.path.join(pdir, img), idx))

        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(64, scale=(0.9, 1.0)),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor()
        ])

        self.norm = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))

        if self.train:
            img = self.aug(img)
        else:
            img = torch.tensor(img/255., dtype=torch.float32).unsqueeze(0)

        img = self.norm(img)
        return img, label
    
class MeasurementsDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.samples = []
        self.id_map = {}

        for idx, pid in enumerate(sorted(os.listdir(root))):
            self.id_map[pid] = idx
            pdir = os.path.join(root, pid)
            for f in os.listdir(pdir):
                self.samples.append((os.path.join(pdir,f), idx))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        x = np.load(path).astype(np.float32)
        x = torch.tensor(x).unsqueeze(0)  # (1,256)
        return x, label
    
class CapacitiveDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.class_to_idx = {}

        # scan directories
        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    self.samples.append(os.path.join(cls_dir, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = np.load(self.samples[idx]).astype(np.float32)  # 8x8 array
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        # add channel dim for PyTorch (1, 8, 8)
        x = torch.tensor(x).unsqueeze(0)  
        y = torch.tensor(y, dtype=torch.long)
        return x, y