# ============================================================
# Single-file: FastGAN-Lite (48x48) — Training, INT8 PTQ, Outputs
# Outputs: generator_fp32.pth, generator_int8.pth,
#          loss_curves.png, qualitative_results.png, training_stats.csv
# ============================================================

# Install minimal packages (Colab usually has these; re-run if needed)
!pip install --quiet torch torchvision matplotlib tqdm pillow

import os, time, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as tq
from torchvision import datasets as tv_datasets, transforms, utils
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optional Colab helpers (wrapped so code runs outside Colab too)
try:
    from google.colab import files as colab_files
    from IPython.display import Image, display
    COLAB = True
except Exception:
    COLAB = False

# ---------------- CONFIG ----------------
IMG_SIZE = 48          # 48x48 as requested
Z_DIM = 128
BASE_CH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FP32_GEN_PATH = "generator_fp32.pth"
FP32_DISC_PATH = "discriminator_fp32.pth"
INT8_GEN_PATH = "generator_int8.pth"
INT8_DISC_PATH = "discriminator_int8.pth"
LOSS_CURVE = "loss_curves.png"
QUAL_PNG = "qualitative_results.png"
STATS_CSV = "training_stats.csv"
GEN_LOSS_NPY = "gen_losses.npy"
DISC_LOSS_NPY = "disc_losses.npy"

QUICK_DEMO = False   # True => tiny quick run; False => full CIFAR-10 training
# Quick params
if QUICK_DEMO:
    EPOCHS = 2
    BATCH_SIZE = 4
    SMALL_SUBSET = 10
else:
    EPOCHS = 20     # adjust if you want longer training
    BATCH_SIZE = 64
    SMALL_SUBSET = None

# Use qnnpack for ARM compatibility
torch.backends.quantized.engine = "qnnpack"

# ---------------- Model definitions (single canonical definition) ----------------
class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, base=BASE_CH, out_ch=3):
        super().__init__()
        self.fc = nn.Linear(z_dim, base*8*3*3)
        # Upsample blocks: Upsample -> Conv2d -> BN -> ReLU
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
        # Accept (B, Z) or (B, Z, 1, 1)
        if z.dim() == 4:
            z = z.view(z.size(0), -1)[:, :Z_DIM]
        z = z.view(z.size(0), -1)
        x = self.fc(z).view(-1, BASE_CH*8, 3, 3)
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        return self.tanh(self.to_rgb(x))  # output in [-1,1]

class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base=BASE_CH):
        super().__init__()
        # Downsample blocks: Conv2d -> BN -> LeakyReLU
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
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        return self.final_conv(x)

# ---------------- Data loaders ----------------
def get_dataset(img_size=IMG_SIZE, quick=QUICK_DEMO, small_n=SMALL_SUBSET):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    ds = tv_datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    if quick and small_n is not None:
        return Subset(ds, list(range(small_n)))
    return ds

def get_test_dataset(img_size=IMG_SIZE):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return tv_datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# ---------------- Training loop ----------------
def train_and_quantize(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, z_dim=Z_DIM, lr=2e-4, save_losses=True):
    # DataLoader (safe num_workers)
    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    except Exception:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    gen = Generator(z_dim).to(DEVICE)
    disc = Discriminator().to(DEVICE)

    opt_g = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    gen_losses = []
    disc_losses = []

    t0 = time.time()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for real, _ in pbar:
            real = real.to(DEVICE)
            bs = real.size(0)
            z = torch.randn(bs, z_dim, device=DEVICE)

            fake = gen(z)

            # Discriminator step
            out_real = disc(real).view(-1)
            out_fake = disc(fake.detach()).view(-1)
            lossD = 0.5*(criterion(out_real, torch.ones_like(out_real)) + criterion(out_fake, torch.zeros_like(out_fake)))
            opt_d.zero_grad(); lossD.backward(); opt_d.step()

            # Generator step
            out_fake_for_g = disc(fake).view(-1)
            lossG = criterion(out_fake_for_g, torch.ones_like(out_fake_for_g))
            opt_g.zero_grad(); lossG.backward(); opt_g.step()

            gen_losses.append(lossG.item()); disc_losses.append(lossD.item())
            pbar.set_postfix(lossG=f"{lossG.item():.4f}", lossD=f"{lossD.item():.4f}")

    elapsed = time.time() - t0

    # Save FP32 state_dicts
    torch.save(gen.state_dict(), FP32_GEN_PATH)
    torch.save(disc.state_dict(), FP32_DISC_PATH)
    print("Saved FP32 models:", FP32_GEN_PATH, FP32_DISC_PATH)

    # Save loss arrays
    if save_losses:
        np.save(GEN_LOSS_NPY, np.array(gen_losses))
        np.save(DISC_LOSS_NPY, np.array(disc_losses))

    # Plot loss curves
    plt.figure(figsize=(8,4))
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE)
    plt.close()
    print("Saved loss curve:", LOSS_CURVE)

    # ---------------- Quantization (PTQ) ----------------
    print("Starting PTQ quantization (qnnpack)...")
    gen_cpu = Generator().cpu()
    disc_cpu = Discriminator().cpu()
    gen_cpu.load_state_dict(torch.load(FP32_GEN_PATH, map_location="cpu"))
    disc_cpu.load_state_dict(torch.load(FP32_DISC_PATH, map_location="cpu"))

    # set eval mode before fusion
    gen_cpu.eval(); disc_cpu.eval()

    # fuse conv+bn+relu in generator blocks (conv is idx 1, bn 2, relu 3)
    try:
        tq.fuse_modules(gen_cpu, [
            ["block1.1","block1.2","block1.3"],
            ["block2.1","block2.2","block2.3"],
            ["block3.1","block3.2","block3.3"],
            ["block4.1","block4.2","block4.3"]
        ], inplace=True)
    except Exception as e:
        print("Generator fuse warning:", e)

    # fuse conv+bn in discriminator blocks (leakyrelu not fuseable)
    try:
        tq.fuse_modules(disc_cpu, [
            ["block1.0","block1.1"],
            ["block2.0","block2.1"],
            ["block3.0","block3.1"],
            ["block4.0","block4.1"]
        ], inplace=True)
    except Exception as e:
        print("Discriminator fuse warning:", e)

    qconfig = tq.get_default_qconfig("qnnpack")
    gen_cpu.qconfig = qconfig
    disc_cpu.qconfig = qconfig

    tq.prepare(gen_cpu, inplace=True)
    tq.prepare(disc_cpu, inplace=True)

    # Calibration: run a few inference passes
    with torch.no_grad():
        # generator calibration with random z
        for _ in range(10):
            _ = gen_cpu(torch.randn(1, Z_DIM))
        # discriminator calibration using a few dataset images if available
        try:
            calib_loader = DataLoader(dataset, batch_size=8, shuffle=True)
            for imgs, _ in calib_loader:
                _ = disc_cpu(imgs)
                break
        except Exception:
            pass

    tq.convert(gen_cpu, inplace=True)
    tq.convert(disc_cpu, inplace=True)

    # Save quantized models (full objects)
    torch.save(gen_cpu, INT8_GEN_PATH)
    torch.save(disc_cpu, INT8_DISC_PATH)
    print("Quantized models saved:", INT8_GEN_PATH, INT8_DISC_PATH)

    # ---------------- Save training stats CSV ----------------
    stats = {
        "gen_loss_mean": float(np.mean(gen_losses)) if len(gen_losses)>0 else 0.0,
        "gen_loss_std": float(np.std(gen_losses)) if len(gen_losses)>0 else 0.0,
        "disc_loss_mean": float(np.mean(disc_losses)) if len(disc_losses)>0 else 0.0,
        "disc_loss_std": float(np.std(disc_losses)) if len(disc_losses)>0 else 0.0,
        "elapsed_seconds": float(elapsed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "img_size": int(IMG_SIZE)
    }
    with open(STATS_CSV, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["metric","value"])
        for k,v in stats.items():
            w.writerow([k,v])
    print("Saved stats:", STATS_CSV)

    return gen_losses, disc_losses, elapsed, stats

# ---------------- Qualitative comparison grid ----------------
def save_qualitative_comparison(test_dataset, fp32_state_path=FP32_GEN_PATH, int8_model_path=INT8_GEN_PATH, out=QUAL_PNG, n=9):
    # load n real images
    loader = DataLoader(test_dataset, batch_size=n, shuffle=True)
    real, _ = next(iter(loader))
    real = real[:n]

    # Load FP32 generator from state dict
    gen_fp32 = Generator().to(DEVICE)
    gen_fp32.load_state_dict(torch.load(fp32_state_path, map_location=DEVICE))
    gen_fp32.eval()
    with torch.no_grad():
        z = torch.randn(n, Z_DIM, device=DEVICE)
        fake_fp32 = gen_fp32(z).cpu()

    # Load INT8 generator object if exists
    if os.path.exists(int8_model_path):
        gen_int8 = torch.load(int8_model_path, map_location=DEVICE)
        gen_int8.eval()
        with torch.no_grad():
            z2 = torch.randn(n, Z_DIM, device=DEVICE)
            fake_int8 = gen_int8(z2).cpu()
    else:
        fake_int8 = fake_fp32.clone()

    # Denormalize real images from training normalization (they are in [0,1] because dataset loader used ToTensor)
    # Our dataset transforms normalized to [-1,1] for training; for display, we convert back
    real_disp = (real * 0.5) + 0.5  # if dataset normalized to [-1,1]
    # fake_fp32 and fake_int8 are tanh outputs in [-1,1] → convert to [0,1]
    fake_fp32_disp = (fake_fp32 * 0.5) + 0.5
    fake_int8_disp = (fake_int8 * 0.5) + 0.5

    # Create concatenated grid: rows =(real ; fp32 ; int8) with n columns
    stacked = torch.cat([real_disp, fake_fp32_disp, fake_int8_disp], dim=0)
    grid = utils.make_grid(stacked, nrow=n, padding=2)
    utils.save_image(grid, out)
    print("Saved qualitative comparison:", out)
    return out

# ---------------- Main Entrypoint ----------------
if __name__ == "__main__":
    print("Mode:", "QUICK_DEMO" if QUICK_DEMO else "FULL")
    ds = get_dataset(img_size=IMG_SIZE, quick=QUICK_DEMO, small_n=SMALL_SUBSET)
    gen_losses, disc_losses, elapsed, stats = train_and_quantize(ds, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save gen/disc loss arrays (already saved in train; repeat to be safe)
    np.save(GEN_LOSS_NPY, np.array(gen_losses))
    np.save(DISC_LOSS_NPY, np.array(disc_losses))

    # Create qualitative comparison using CIFAR-10 test set (unseen)
    test_ds = get_test_dataset(img_size=IMG_SIZE)
    qual_path = save_qualitative_comparison(test_ds, fp32_state_path=FP32_GEN_PATH, int8_model_path=INT8_GEN_PATH, out=QUAL_PNG, n=9)

    # Display and offer downloads (Colab-friendly)
    if COLAB:
        display(Image(filename=qual_path))
        display(Image(filename=LOSS_CURVE))
        print("Training stats:", stats)
        # Download main artifacts
        for f in [qual_path, LOSS_CURVE, FP32_GEN_PATH, INT8_GEN_PATH, STATS_CSV]:
            if os.path.exists(f):
                try:
                    colab_files.download(f)
                except Exception as e:
                    print("Download failed for", f, ":", e)
    else:
        print("Files saved:", QUAL_PNG, LOSS_CURVE, FP32_GEN_PATH, INT8_GEN_PATH, STATS_CSV)
        print("Stats:", stats)

# ============================================================
# End of script
# ============================================================
