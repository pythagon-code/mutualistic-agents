import sys
import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fnn import FNN
from src.encoder_cnn import EncoderCNN
from src.utils import RangedTanh, device

# --- input ---
num_epochs = int(input("epochs: "))

image_dir = os.path.join("temp", "images", "gan")
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
os.makedirs(image_dir)

# --- data ---
transform = transforms.Compose([
    transforms.Resize((10, 10)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

dataset    = datasets.MNIST(root=os.path.join("temp", "data"), download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

# --- hyperparams ---
batch_size        = 128
image_size        = 100  # 10x10
latent_dim        = 128
print_interval    = 10
save_interval     = 1

# --- networks ---
gen = nn.Sequential(
    EncoderCNN(
        input_size        = latent_dim,
        hidden_size       = 64 * 25,
        num_hidden_layers = 2,
        num_channels      = 64,
        num_conv_t_3s     = 1,
        num_upscales      = 1,
        num_conv_5s       = 1,
        num_conv_3s       = 1,
        dropout_rate      = 0.2,
    ),
    nn.Conv2d(64, 1, kernel_size=1),
    RangedTanh(0, 1),
).to(device)

critic = nn.Sequential(
    # --- conv backbone ---
    nn.utils.spectral_norm(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)),
    nn.LeakyReLU(0.2, inplace=True),

    nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),  # 10x10 -> 5x5
    nn.LeakyReLU(0.2, inplace=True),
    
    nn.utils.spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)), # 5x5 -> 3x3
    nn.LeakyReLU(0.2, inplace=True),

    nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)), # 3x3 -> 3x3
    nn.LeakyReLU(0.2, inplace=True),
    
    nn.Flatten(),  # flatten for fully connected layers
    
    # --- fully connected layers ---
    nn.utils.spectral_norm(nn.Linear(128 * 3 * 3, 256)),
    nn.LeakyReLU(0.2, inplace=True),
    
    nn.utils.spectral_norm(nn.Linear(256, 256)),
    nn.LeakyReLU(0.2, inplace=True),
    
    nn.utils.spectral_norm(nn.Linear(256, 1))  # output single score
).to(device)

gen_optimizer    = optim.Adam(gen.parameters(),    lr=3e-4, betas=(0.0, 0.9))
critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4, betas=(0.0, 0.9))

# --- training ---
i = 0
gen.train()
critic.train()
for epoch in trange(num_epochs):
    for real_images, _ in dataloader:
        real_images = real_images.to(device)

        # critic update
        critic_optimizer.zero_grad(set_to_none=True)
        for _ in range(3):
            with torch.no_grad():
                latent_vectors = torch.randn(batch_size, latent_dim, device=device)
                fake_images    = gen(latent_vectors)
            real_score  = critic(real_images).mean()
            fake_score  = critic(fake_images).mean()
            critic_loss = torch.relu(1 - real_score).mean() + torch.relu(1 + fake_score).mean()
            critic_loss.backward()
            critic_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True)

        # generator update
        gen_optimizer.zero_grad(set_to_none=True)
        latent_vectors = torch.randn(batch_size, latent_dim, device=device)
        fake_images    = gen(latent_vectors)
        diversity      = (fake_images[::2] - fake_images[1::2]).norm(p=2, dim=1).mean() / image_size
        gen_loss       = -critic(fake_images).mean()
        gen_loss.backward()
        gen_optimizer.step()
        gen_optimizer.zero_grad(set_to_none=True)

        i += 1
        if i % print_interval == 0:
            print(f"{i}, g: {gen_loss.item():.6f}, c: {critic_loss.item():.6f}, d: {diversity.item():.6f}")

    if (epoch + 1) % save_interval == 0:
        gen.eval()
        with torch.no_grad():
            sample_latents = torch.randn(16, latent_dim, device=device)
            samples        = gen(sample_latents).cpu().numpy().reshape(16, 10, 10)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for ax, img in zip(axes.flatten(), samples):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f"samples_epoch_{epoch + 1}.png"))
        plt.close()
        gen.train()

        print(f"Check {image_dir} for generated images.")

# --- final samples ---
gen.eval()
with torch.no_grad():
    sample_latents = torch.randn(16, latent_dim, device=device)
    samples        = gen(sample_latents).cpu().numpy().reshape(16, 10, 10)
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for ax, img in zip(axes.flatten(), samples):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(image_dir, "samples_final.png"))
plt.close()