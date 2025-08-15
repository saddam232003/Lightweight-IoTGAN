#!/usr/bin/env python3
# ONNX export and virtualized execution example

import torch
from model import Generator

# Initialize the generator (adjust parameters as needed)
g = Generator(...)

# Load pretrained FP32 weights
g.load_state_dict(torch.load('iotgan_fp32.pth', map_location='cpu'))
g.eval()

# Create a dummy latent vector
dummy = torch.randn(1, 128)

# Export to ONNX format with opset version 13
torch.onnx.export(
    g, dummy, "iotgan.onnx", opset_version=13,
    input_names=['z'], output_names=['img'],
    dynamic_axes={'z': [0], 'img': [0]}
)

print("ONNX model exported successfully to iotgan.onnx")
