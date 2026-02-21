"""U-Net segmentation inference pipeline."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.unet import UNet

CLASS_NAMES = {
    0: "background",
    1: "building",
    2: "woodland",
    3: "water",
    4: "road",
}


class SegmentationPipeline:
    """End-to-end segmentation inference pipeline.

    Args:
        weights_path: Path to trained U-Net weights.
        num_classes: Number of segmentation classes.
        device: Inference device.
        image_size: Input size for the model.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        num_classes: int = 5,
        device: str = "cpu",
        image_size: int = 256,
    ):
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.image_size = image_size
        self.class_names = {
            i: CLASS_NAMES.get(i, f"class_{i}") for i in range(num_classes)
        }

        self.model = UNet(in_channels=3, num_classes=num_classes)

        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(
                str(weights_path), map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: np.ndarray | str | Path) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[0] <= 4:  # (C, H, W) -> (H, W, C)
            image = np.transpose(image, (1, 2, 0))

        # Use only first 3 channels if more exist
        image = image[:, :, :3]

        pil_img = Image.fromarray(image.astype(np.uint8))
        pil_img = pil_img.resize((self.image_size, self.image_size))
        tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def predict(self, image: np.ndarray | str | Path) -> dict:
        """Run segmentation on an image.

        Returns:
            Dict with 'mask', 'class_distribution', 'class_names'.
        """
        tensor = self.preprocess(image).to(self.device)
        output = self.model(tensor)
        mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()

        # Compute class distribution
        total_pixels = mask.size
        distribution = {}
        for cls_id in range(self.num_classes):
            count = int(np.sum(mask == cls_id))
            distribution[self.class_names[cls_id]] = round(count / total_pixels, 4)

        return {
            "mask": mask,
            "class_distribution": distribution,
            "class_names": self.class_names,
            "image_size": [self.image_size, self.image_size],
        }

    def mask_to_geojson(
        self, mask: np.ndarray, target_class: int = 1
    ) -> dict:
        """Convert a binary mask to simple GeoJSON polygons (pixel coords).

        This is a simplified version — for real geo-referenced output,
        you'd use rasterio's shapes() with the image's transform.
        """
        from geojson import Feature, FeatureCollection, Polygon

        binary = (mask == target_class).astype(np.uint8)

        # Simple bounding box of the class region
        rows, cols = np.where(binary == 1)
        if len(rows) == 0:
            return FeatureCollection(features=[])

        min_r, max_r = int(rows.min()), int(rows.max())
        min_c, max_c = int(cols.min()), int(cols.max())

        coords = [
            [min_c, min_r],
            [max_c, min_r],
            [max_c, max_r],
            [min_c, max_r],
            [min_c, min_r],
        ]

        feature = Feature(
            geometry=Polygon([coords]),
            properties={
                "class_id": target_class,
                "class_name": self.class_names.get(target_class, "unknown"),
                "pixel_count": int(np.sum(binary)),
            },
        )
        return FeatureCollection(features=[feature])
