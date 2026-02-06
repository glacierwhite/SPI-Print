import sys
if './' not in sys.path:
	sys.path.append('./')
     
import logging
import time
import os

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_dataset
from torch.utils.data import DataLoader
from running import setup
from options import Options
from models.arcface import ArcFace
from models.fingerprint_vit import FingerViT
from models.measurements_1d_cnn import MeasurementsCNN
import utils

def main(config):
    total_epoch_time = 0

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device.type == "cuda":
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    EMB_DIM = 64

    # Load dataset
    logger.info("Loading and preprocessing data ...")
    dataset_class = get_dataset(config['modality'])
    train_set = dataset_class(root=config['data_dir']+"/"+config['modality']+"/train", train=True)
    val_set   = dataset_class(root=config['data_dir']+"/"+config['modality']+"/val", train=False)

    logger.info("{} samples may be used for training".format(len(train_set)))
    logger.info("{} samples will be used for validation".format(len(val_set)))

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)

    NUM_CLASSES = len(train_set.id_map)

    # Create model
    logger.info("Creating model ...")
    model = FingerViT(emb_dim=EMB_DIM).to(device)
    arcface = ArcFace(EMB_DIM, NUM_CLASSES).to(device)

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
         list(model.parameters()) + list(arcface.parameters()),
         lr=config['lr'], weight_decay=config['weight_decay']
    )

    criterion = nn.CrossEntropyLoss()

    ## Training Loop
    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        arcface.train() if train else arcface.eval()

        total_loss, correct, total = 0, 0, 0

        for data, labels in tqdm(loader):
            data, labels = data.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            emb = model(data)
            logits = arcface(emb, labels)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    # Train
    best_acc = 0
    logger.info('Starting training...')
    for epoch in range(config["epochs"]):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)

        logger.info(f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/"+model.__class__.__name__+".pth")

    # ## Verification
    # model.eval()

    # img1, img2 = ...   # load two fingerprints
    # e1 = model(img1.unsqueeze(0).to(DEVICE))
    # e2 = model(img2.unsqueeze(0).to(DEVICE))

    # sim = torch.cosine_similarity(e1, e2)
    # print("Similarity:", sim.item())

if  __name__ == "__main__":
     args = Options().parse()
     config = setup(args)
     main(config)