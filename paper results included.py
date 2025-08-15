# Paste this entire block into a single Colab cell and run.
# Colab-ready: produces loss curves + graphs for the two tables you provided.
# Author: ChatGPT
# Date: 2025-08-11

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Settings
np.random.seed(42)
NUM_EPOCHS = 300
OUT_DIR = "fastgan_plots_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1) Synthetic GAN loss generator (plausible dynamics & no NameError)
def synth_gan_losses(num_epochs=NUM_EPOCHS, seed=42):
    np.random.seed(seed)
    t = np.arange(num_epochs)

    # Discriminator: quick initial drop (learning real/fake), then oscillations as GAN adversarial balance forms
    disc_base = 0.9 * np.exp(-0.02 * t) + 0.12
    disc_osc = 0.06 * np.sin(0.25 * t) + 0.03 * np.sin(0.07 * t * 3.14)
    disc_noise = 0.02 * np.random.randn(num_epochs)
    disc_losses = np.clip(disc_base + disc_osc + disc_noise, 0.02, None)

    # Generator: starts higher, slowly decreases overall but with adversarial spikes
    gen_base = 1.2 * (1.0 / (1.0 + 0.015 * t)) + 0.05
    gen_osc = 0.08 * np.sin(0.18 * t + 0.9) + 0.04 * np.sin(0.045 * t)
    gen_noise = 0.03 * np.random.randn(num_epochs)
    gen_losses = np.clip(gen_base + gen_osc + gen_noise, 0.02, None)

    # Add a couple of simulated "instability" spikes to show realistic GAN training issues
    spikes_idx = [int(num_epochs*0.25), int(num_epochs*0.6)]
    for idx in spikes_idx:
        gen_losses[idx:idx+5] += 0.25 * np.exp(-0.8 * np.arange(5))  # generator spike
        disc_losses[idx:idx+5] -= 0.08 * np.exp(-0.6 * np.arange(5))  # discriminator dips a bit

    return gen_losses, disc_losses

gen_losses, disc_losses = synth_gan_losses()

# Plot generator + discriminator losses
plt.figure(figsize=(10,5))
plt.plot(gen_losses, label="Generator Loss", linewidth=2)
plt.plot(disc_losses, label="Discriminator Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Synthetic GAN Training Losses (Generator vs Discriminator)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gan_losses.png"), dpi=150)
plt.show()

# Save losses to CSV so you can inspect or reuse
loss_df = pd.DataFrame({
    "epoch": np.arange(len(gen_losses)),
    "gen_loss": gen_losses,
    "disc_loss": disc_losses
})
loss_csv_path = os.path.join(OUT_DIR, "gan_losses.csv")
loss_df.to_csv(loss_csv_path, index=False)
print(f"Saved synthetic loss CSV to: {loss_csv_path}")

# --- 2) Quantized GAN Comparison (table 1) -> DataFrame + grouped bar plot
data_quantized = {
    "Model": ["DCGAN", "StyleGAN", "FastGAN", "LadaGAN", "Ours"],
    "Params_M": [12.5, 25.0, 10.0, 8.2, 6.8],
    "FLOPs_G": [2.8, 5.5, 1.5, 1.2, 0.85],
    "Time_ms": [1200, 2500, 800, 700, 290],
    "Mem_MB": [750, 1200, 850, 700, 340]
}
df_quant = pd.DataFrame(data_quantized).set_index("Model")
print("\nQuantized GAN Comparison (table 1):")
display(df_quant)

# Grouped bar plot (all metrics) - wide format
ax = df_quant.plot(kind="bar", figsize=(12,6), rot=0)
ax.set_ylabel("Value (units vary per metric)")
ax.set_title("Quantized GAN Comparison: Params (M), FLOPs (G), Time (ms), Mem (MB)")
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "quantized_gan_comparison_grouped.png"), dpi=150)
plt.show()

# Also plot Time and Mem separately to emphasize runtime & memory advantage
fig, axes = plt.subplots(1,2,figsize=(12,4))
df_quant["Time_ms"].plot(kind="bar", ax=axes[0], rot=0)
axes[0].set_title("Inference Time (ms)")
axes[0].set_ylabel("ms")
axes[0].grid(axis='y', alpha=0.2)
df_quant["Mem_MB"].plot(kind="bar", ax=axes[1], rot=0)
axes[1].set_title("Memory (MB)")
axes[1].set_ylabel("MB")
axes[1].grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "time_mem_separate.png"), dpi=150)
plt.show()

# Save quant table CSV
quant_csv_path = os.path.join(OUT_DIR, "quantized_gan_table.csv")
df_quant.to_csv(quant_csv_path)
print(f"Saved quantized comparison CSV to: {quant_csv_path}")

# --- 3) Statistical Comparison (table 2) -> DataFrame + individual metric bar charts
data_stats = {
    "Method": [
        "DCGAN (FP32)",
        "LightGAN (FP32)",
        "MobileGAN (FP32)",
        "FastGAN-L (FP32)",
        "FastGAN-L (INT8)"
    ],
    "FID": [32.45, 29.87, 27.52, 25.34, 25.89],
    "IS": [3.12, 3.45, 3.56, 3.79, 3.77],
    "PSNR": [22.18, 23.01, 23.45, 24.28, 24.12],
    "LPIPS": [0.215, 0.198, 0.191, 0.176, 0.179],
    "Size_MB": [48.3, 32.7, 28.5, 12.4, 3.2],
    "Time_ms": [38.6, 31.4, 27.8, 14.6, 5.9]
}
df_stats = pd.DataFrame(data_stats).set_index("Method")
print("\nStatistical Comparison (table 2):")
display(df_stats)

# Plot each metric separately in a 2x3 grid for readability
metrics = ["FID", "IS", "PSNR", "LPIPS", "Size_MB", "Time_ms"]
fig, axes = plt.subplots(2, 3, figsize=(15,8))
axes = axes.flatten()
for ax, metric in zip(axes, metrics):
    df_stats[metric].plot(kind="bar", ax=ax, rot=20)
    # annotate bars with values
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, rotation=0)
    ax.set_title(metric)
    ax.grid(axis='y', alpha=0.15)
plt.suptitle("Statistical Comparison across metrics (lower is better for FID & LPIPS; higher better for IS & PSNR)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "statistical_comparison_metrics.png"), dpi=150)
plt.show()

# Save stats table CSV
stats_csv_path = os.path.join(OUT_DIR, "statistical_comparison_table.csv")
df_stats.to_csv(stats_csv_path)
print(f"Saved statistical comparison CSV to: {stats_csv_path}")

# --- 4) Optional: normalized radar-like overview (normalized 0..1 for visualization)
# Normalize metrics so bigger is always "better" for visualization purpose
def normalize_series(s, invert=False):
    arr = np.array(s, dtype=float)
    mn, mx = arr.min(), arr.max()
    norm = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
    return 1 - norm if invert else norm

# Create normalized overview DataFrame (make FID and LPIPS inverted because lower is better)
df_overview = pd.DataFrame({
    "FID_norm": normalize_series(df_stats["FID"], invert=True),
    "IS_norm": normalize_series(df_stats["IS"]),
    "PSNR_norm": normalize_series(df_stats["PSNR"]),
    "LPIPS_norm": normalize_series(df_stats["LPIPS"], invert=True),
    "Size_norm": normalize_series(df_stats["Size_MB"], invert=True),
    "Time_norm": normalize_series(df_stats["Time_ms"], invert=True)
}, index=df_stats.index)

# Plot heatmap-like overview (imshow)
plt.figure(figsize=(8,4.5))
plt.imshow(df_overview.values, aspect='auto', cmap='viridis', vmin=0, vmax=1)
plt.colorbar(label='Normalized (0 worst -> 1 best)')
plt.yticks(range(len(df_overview.index)), df_overview.index)
plt.xticks(range(len(df_overview.columns)), df_overview.columns, rotation=45)
plt.title("Normalized Overview (higher = better)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "normalized_overview.png"), dpi=150)
plt.show()

# Print output folder contents for convenience
print("\nSaved files in:", OUT_DIR)
for fname in sorted(os.listdir(OUT_DIR)):
    print(" -", fname)

print("\nAll done. The CSVs and PNGs are in the Colab workspace under the folder:", OUT_DIR)
