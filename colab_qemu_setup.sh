#!/bin/bash
# Example commands executed in Colab (shell):

# Install QEMU user emulation and binfmt support
sudo apt-get update && sudo apt-get install -y qemu-user-static binfmt-support

# Download minimal ARM64 rootfs (example: Debian arm64) and chroot (or use proot)
# Note: Replace <rootfs_url> with the actual download link for the ARM64 rootfs
# wget <rootfs_url> -O rootfs.tar.gz
# mkdir arm64-rootfs && sudo tar -xzf rootfs.tar.gz -C arm64-rootfs

# Inside chroot, install python3, pip, and onnxruntime-aarch64 for sanity tests
# sudo chroot arm64-rootfs /bin/bash -c "apt-get update && apt-get install -y python3 python3-pip && pip install onnxruntime-aarch64"
