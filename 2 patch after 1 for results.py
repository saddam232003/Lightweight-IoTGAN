# --- Patch to display & download results in Colab ---
from google.colab import files
from IPython.display import Image, display
import os

# Show qualitative results inline
if os.path.exists("qualitative_results.png"):
    print("Displaying qualitative comparison image:")
    display(Image(filename="qualitative_results.png"))
    files.download("qualitative_results.png")

# Show loss curves inline
if os.path.exists("loss_curves.png"):
    print("Displaying loss curves (Generator vs Discriminator):")
    display(Image(filename="loss_curves.png"))
    files.download("loss_curves.png")

# Download INT8 & FP32 models
for model_file in ["generator_fp32.pth", "discriminator_fp32.pth",
                   "generator_int8.pth", "discriminator_int8.pth"]:
    if os.path.exists(model_file):
        print(f"Downloading: {model_file}")
        files.download(model_file)

# Download training statistics CSV if exists
if os.path.exists("training_stats.csv"):
    print("Downloading training stats CSV:")
    files.download("training_stats.csv")
