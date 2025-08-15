Uses entire CIFAR-10 training set.

Retains the no-error quantization fix (.eval() before fusion).

Keeps INT8 PTQ for ARM IoT devices (qnnpack).

Keeps model lightweight so it’s Colab-friendly even with full data.

Saves both FP32 and INT8 models.
This FULL dataset mode will:

Train on all 50k CIFAR-10 images resized to 48×48.

Run INT8 quantization after training.

Save generator_fp32.pth, discriminator_fp32.pth, and their INT8 versions.

Produce loss_curve.png