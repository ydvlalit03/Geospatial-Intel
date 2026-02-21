"""YOLOv8 detection inference pipeline."""

from pathlib import Path

import numpy as np
from PIL import Image

from src.models.detector import DetectionResult, SatelliteDetector


class DetectionPipeline:
    """End-to-end object detection inference pipeline.

    Args:
        weights_path: Path to YOLOv8 weights.
        confidence_threshold: Minimum detection confidence.
        device: Inference device.
    """

    def __init__(
        self,
        weights_path: str | Path = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
    ):
        self.detector = SatelliteDetector(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )

    def preprocess(self, image: np.ndarray | str | Path) -> np.ndarray:
        """Ensure image is in the right format for YOLOv8."""
        if isinstance(image, (str, Path)):
            return np.array(Image.open(image).convert("RGB"))

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))

        return image[:, :, :3].astype(np.uint8)

    def predict(
        self,
        image: np.ndarray | str | Path,
        confidence_threshold: float | None = None,
    ) -> dict:
        """Run detection and return structured results.

        Returns:
            Dict with detection results including bounding boxes, classes, counts.
        """
        if isinstance(image, np.ndarray):
            image = self.preprocess(image)

        result = self.detector.predict(image, confidence_threshold)

        # Summarize by class
        class_counts: dict[str, int] = {}
        for det in result.detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        output = result.to_dict()
        output["class_summary"] = class_counts
        return output
