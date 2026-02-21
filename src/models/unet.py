"""U-Net architecture for satellite image segmentation."""

import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):
    """Two consecutive conv-batchnorm-relu blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net with optional ResNet34 encoder for satellite image segmentation.

    Args:
        in_channels: Number of input channels (e.g. 3 for RGB, 4 for RGBN).
        num_classes: Number of segmentation classes.
        use_resnet: If True, use a pretrained ResNet34 encoder.
    """

    def __init__(
        self, in_channels: int = 3, num_classes: int = 6, use_resnet: bool = False
    ):
        super().__init__()
        self.use_resnet = use_resnet

        if use_resnet:
            self._build_resnet_encoder(in_channels)
        else:
            self._build_vanilla_encoder(in_channels)

        self._build_decoder(num_classes)

    def _build_resnet_encoder(self, in_channels: int):
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Adapt first conv if input channels != 3
        if in_channels != 3:
            self.input_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.input_conv = resnet.conv1

        self.input_bn = resnet.bn1
        self.input_relu = resnet.relu
        self.input_pool = resnet.maxpool

        self.enc1 = resnet.layer1  # 64
        self.enc2 = resnet.layer2  # 128
        self.enc3 = resnet.layer3  # 256
        self.enc4 = resnet.layer4  # 512

        self.encoder_channels = [64, 64, 128, 256, 512]

    def _build_vanilla_encoder(self, in_channels: int):
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)

        self.encoder_channels = [64, 128, 256, 512, 1024]

    def _build_decoder(self, num_classes: int):
        ch = self.encoder_channels

        if self.use_resnet:
            bottleneck_ch = ch[4]
            self.up4 = nn.ConvTranspose2d(bottleneck_ch, 256, kernel_size=2, stride=2)
            self.dec4 = DoubleConv(256 + ch[3], 256)
            self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(128 + ch[2], 128)
            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(64 + ch[1], 64)
            self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(32 + ch[0], 32)
            self.up0 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
            self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        else:
            self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.dec4 = DoubleConv(1024, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(256, 128)
            self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(128, 64)
            self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_resnet:
            return self._forward_resnet(x)
        return self._forward_vanilla(x)

    def _forward_resnet(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.input_relu(self.input_bn(self.input_conv(x)))  # 64, H/2
        x_pool = self.input_pool(x0)  # 64, H/4
        e1 = self.enc1(x_pool)   # 64,  H/4
        e2 = self.enc2(e1)       # 128, H/8
        e3 = self.enc3(e2)       # 256, H/16
        e4 = self.enc4(e3)       # 512, H/32

        # Decoder
        d4 = self.up4(e4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))
        d0 = self.up0(d1)

        return self.final_conv(d0)

    def _forward_vanilla(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final_conv(d1)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Dice + CrossEntropy loss."""

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(pred, target) + self.ce_weight * self.ce(
            pred, target
        )
