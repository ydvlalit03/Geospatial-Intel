"""Download pretrained model weights for inference."""

import os
import sys
from pathlib import Path

# Add project root to path so we can import src.*
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ultralytics import YOLO

WEIGHTS_DIR = Path(__file__).parent


def download_yolov8():
    """Download YOLOv8n pretrained weights."""
    output = WEIGHTS_DIR / "yolov8_satellite.pt"
    if output.exists():
        print(f"YOLOv8 weights already exist: {output}")
        return

    print("Downloading YOLOv8n pretrained weights...")
    model = YOLO("yolov8n.pt")
    # Copy the downloaded weights to our weights directory
    import shutil

    default_path = Path("yolov8n.pt")
    if default_path.exists():
        shutil.move(str(default_path), str(output))
        print(f"Saved to {output}")


def create_dummy_unet_weights():
    """Create dummy U-Net weights for testing (random initialization)."""
    output = WEIGHTS_DIR / "unet_sentinel2.pth"
    if output.exists():
        print(f"U-Net weights already exist: {output}")
        return

    print("Creating dummy U-Net weights for testing...")
    from src.models.unet import UNet

    model = UNet(in_channels=3, num_classes=6)
    torch.save(model.state_dict(), str(output))
    print(f"Saved to {output}")


if __name__ == "__main__":
    create_dummy_unet_weights()
    download_yolov8()
    print("Done!")
