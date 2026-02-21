"""PyTorch Dataset classes for satellite imagery."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class SatelliteSegmentationDataset(Dataset):
    """Dataset for satellite image segmentation (image + mask pairs).

    Expects directory structure:
        root/
        ├── images/
        │   ├── img_001.png
        │   └── ...
        └── masks/
            ├── img_001.png
            └── ...

    Args:
        root: Root directory containing images/ and masks/ subdirs.
        image_size: Resize images to this size (H, W).
        augment: Whether to apply data augmentation.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (256, 256),
        augment: bool = False,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.image_size = image_size

        self.image_paths = sorted(self.image_dir.glob("*.png")) + sorted(
            self.image_dir.glob("*.tif")
        )

        self.transform = self._build_transform(augment) if HAS_ALBUMENTATIONS else None

    def _build_transform(self, augment: bool) -> A.Compose:
        transforms = [A.Resize(*self.image_size)]

        if augment:
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(p=0.2),
                ]
            )

        return A.Compose(transforms)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = np.array(
                Image.open(img_path).convert("RGB").resize(
                    (self.image_size[1], self.image_size[0])
                )
            )
            mask = np.array(
                Image.open(mask_path).convert("L").resize(
                    (self.image_size[1], self.image_size[0]),
                    resample=Image.NEAREST,
                )
            )

        # Convert to tensors: image [C, H, W] float32, mask [H, W] long
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return image, mask
