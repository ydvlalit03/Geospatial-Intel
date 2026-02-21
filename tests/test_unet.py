"""Tests for U-Net model."""

import torch
import pytest

from src.models.unet import UNet, DiceLoss, CombinedLoss


def test_unet_vanilla_output_shape():
    model = UNet(in_channels=3, num_classes=6, use_resnet=False)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 6, 256, 256)


def test_unet_different_num_classes():
    model = UNet(in_channels=3, num_classes=10, use_resnet=False)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 10, 128, 128)


def test_unet_4channel_input():
    model = UNet(in_channels=4, num_classes=6, use_resnet=False)
    x = torch.randn(1, 4, 256, 256)
    out = model(x)
    assert out.shape == (1, 6, 256, 256)


def test_dice_loss():
    loss_fn = DiceLoss()
    pred = torch.randn(2, 6, 64, 64)
    target = torch.randint(0, 6, (2, 64, 64))
    loss = loss_fn(pred, target)
    assert loss.item() >= 0
    assert loss.item() <= 1


def test_combined_loss():
    loss_fn = CombinedLoss()
    pred = torch.randn(2, 6, 64, 64)
    target = torch.randint(0, 6, (2, 64, 64))
    loss = loss_fn(pred, target)
    assert loss.item() > 0


def test_unet_resnet_encoder():
    model = UNet(in_channels=3, num_classes=6, use_resnet=True)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 6, 256, 256)
