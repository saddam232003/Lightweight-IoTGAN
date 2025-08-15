# =============================================
# 🚀 IoT-Friendly FastGAN-Lite (48×48) + INT8 PTQ
# FULL CIFAR-10 Dataset Mode
# Author: Dr.Muhammad Saddam Khokhar
# =============================================

!pip install --quiet torch torchvision gradio==4.44.0 matplotlib tqdm datasets pillow

import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as tq
from torchvision import datasets as tv_datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------
# Config
# ------------------
IMG_SIZE = 48
Z_DIM = 128
BASE_CH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP32_GEN_PATH = "generator_fp32.pth"
FP32_DISC_PATH = "discriminator_fp32.pth"
INT8_GEN_PATH = "generator_int8.pth"
INT8_DISC_PATH = "discriminator_int8.pth"
torch.backends.quantized.engine = "qnnpack"

# ------------------
# Models
# ------------------
class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, base=BASE_CH, out_ch=3):
        super().__init__()
        self.fc = nn.Linear(z_dim, base*8*3*3)
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base*8, base*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base*4, base*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True)
        )
        self.block3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base*2, base, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )
        self.block4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base, base, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )
        self.to_rgb = nn.Conv2d(base, out_ch, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        if z.dim() == 4:
            z_flat = z.view(z.size(0), -1)[:, :Z_DIM]
        else:
            z_flat = z
        x = self.fc(z_flat).view(-1, BASE_CH*8, 3, 3)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.tanh(self.to_rgb(x))

class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base=BASE_CH):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(base*4, base*8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_conv = nn.Conv2d(base*8, 1, 3, 1, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.final_conv(x)

# ------------------
# Dataset Loader
# ------------------
def load_full_cifar10(img_size=IMG_SIZE):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return tv_datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# ------------------
# Plot Loss
# ------------------
def save_loss_plot(gen_losses, disc_losses, out="loss_curve.png"):
    plt.figure(figsize=(6,3))
    plt.plot(gen_losses, label="Gen Loss")
    plt.plot(disc_losses, label="Disc Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

# ------------------
# Quantization
# ------------------
def quantize_models(fp32_gen_path=FP32_GEN_PATH, fp32_disc_path=FP32_DISC_PATH, dataset=None):
    gen = Generator().cpu()
    disc = Discriminator().cpu()
    gen.load_state_dict(torch.load(fp32_gen_path, map_location="cpu"))
    disc.load_state_dict(torch.load(fp32_disc_path, map_location="cpu"))

    gen.eval()
    disc.eval()

    tq.fuse_modules(gen, [["block1.1", "block1.2", "block1.3"],
                          ["block2.1", "block2.2", "block2.3"],
                          ["block3.1", "block3.2", "block3.3"],
                          ["block4.1", "block4.2", "block4.3"]], inplace=True)

    tq.fuse_modules(disc, [["block1.0", "block1.1"],
                           ["block2.0", "block2.1"],
                           ["block3.0", "block3.1"],
                           ["block4.0", "block4.1"]], inplace=True)

    qconfig = tq.get_default_qconfig("qnnpack")
    gen.qconfig = qconfig
    disc.qconfig = qconfig

    tq.prepare(gen, inplace=True)
    tq.prepare(disc, inplace=True)

    with torch.no_grad():
        for _ in range(10):
            gen(torch.randn(1, Z_DIM))
        if dataset:
            loader = DataLoader(dataset, batch_size=2)
            for imgs, _ in loader:
                disc(imgs)
                break

    tq.convert(gen, inplace=True)
    tq.convert(disc, inplace=True)

    torch.save(gen, INT8_GEN_PATH)
    torch.save(disc, INT8_DISC_PATH)

# ------------------
# Training
# ------------------
def train_gan(dataset, epochs=20, batch_size=64, z_dim=Z_DIM, lr=2e-4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    gen = Generator(z_dim).to(DEVICE)
    disc = Discriminator().to(DEVICE)

    opt_g = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    gen_losses, disc_losses = [], []

    for epoch in range(epochs):
        for real, _ in tqdm(loader):
            real = real.to(DEVICE)
            bs = real.size(0)
            z = torch.randn(bs, z_dim, device=DEVICE)
            fake = gen(z)

            # Discriminator
            lossD = 0.5 * (criterion(disc(real).view(-1), torch.ones(bs, device=DEVICE)) +
                           criterion(disc(fake.detach()).view(-1), torch.zeros(bs, device=DEVICE)))
            opt_d.zero_grad()
            lossD.backward()
            opt_d.step()

            # Generator
            lossG = criterion(disc(fake).view(-1), torch.ones(bs, device=DEVICE))
            opt_g.zero_grad()
            lossG.backward()
            opt_g.step()

            gen_losses.append(lossG.item())
            disc_losses.append(lossD.item())

    torch.save(gen.state_dict(), FP32_GEN_PATH)
    torch.save(disc.state_dict(), FP32_DISC_PATH)
    save_loss_plot(gen_losses, disc_losses)

    quantize_models(fp32_gen_path=FP32_GEN_PATH, fp32_disc_path=FP32_DISC_PATH, dataset=dataset)

# ------------------
# Run FULL Training
# ------------------
dataset = load_full_cifar10()
train_gan(dataset, epochs=20, batch_size=64)

print("✅ Full Training + INT8 Quantization Completed! Models saved.")
