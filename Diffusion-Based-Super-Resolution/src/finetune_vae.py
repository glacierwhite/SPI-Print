from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from dataset import ImageDataset
from tqdm import tqdm
import torch

## Load the VAE

# Choose the Stable Diffusion version
vae_model_id = "runwayml/stable-diffusion-v1-5"  # or sd-2-1, sd-3-5, etc.

# Load only the VAE
vae = AutoencoderKL.from_pretrained(vae_model_id, subfolder="vae")
vae = vae.to("cuda:3")  # move to GPU if available

## Prepare the dataset
dataset = ImageDataset("./data/train/HR_Fingerprints")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

## Define optimizer and loss
optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

## Fine-tuning
vae.train()
for epoch in range(10):  # adjust epochs
    for batch in dataloader:
        batch = batch.to("cuda:3")
        
        # Encode and decode
        latents = vae.encode(batch).latent_dist.sample()  # shape [B, C, H/8, W/8]
        recon = vae.decode(latents).sample  # shape [B, 3, H, W]
        
        # Compute loss
        loss = criterion(recon, batch)
        print("Training Epoch",epoch," Loss:",loss.item())
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch} Loss: {loss.item():.4f}")

## Save the fine-tuned VAE
vae.save_pretrained("fine_tuned_vae")