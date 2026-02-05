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