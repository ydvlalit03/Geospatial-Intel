"""YOLOv8 wrapper for satellite object detection."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Single detection result."""

    bbox: list[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float


@dataclass
class DetectionResult:
    """Collection of detections for one image."""

    detections: list[Detection]
    image_shape: tuple[int, int]

    @property
    def count(self) -> int:
        return len(self.detections)

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return DetectionResult(detections=filtered, image_shape=self.image_shape)

    def to_dict(self) -> dict:
        return {
            "num_detections": self.count,
            "image_shape": list(self.image_shape),
            "detections": [
                {
                    "bbox": d.bbox,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                }
                for d in self.detections
            ],
        }


class SatelliteDetector:
    """YOLOv8 wrapper for satellite imagery object detection.

    Args:
        weights_path: Path to YOLOv8 weights (.pt file).
        confidence_threshold: Minimum confidence for detections.
        device: Device to run inference on ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        weights_path: str | Path = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
    ):
        self.model = YOLO(str(weights_path))
        self.confidence_threshold = confidence_threshold
        self.device = device

    def predict(
        self,
        image: np.ndarray | str | Path,
        confidence_threshold: float | None = None,
    ) -> DetectionResult:
        """Run detection on an image.

        Args:
            image: Image as numpy array (H, W, C) or path to image file.
            confidence_threshold: Override default confidence threshold.

        Returns:
            DetectionResult with all detections.
        """
        conf = confidence_threshold or self.confidence_threshold
        results = self.model.predict(
            source=image, conf=conf, device=self.device, verbose=False
        )

        result = results[0]
        names = result.names
        boxes = result.boxes

        detections = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls[0].cpu().numpy())
                conf_score = float(box.conf[0].cpu().numpy())
                detections.append(
                    Detection(
                        bbox=xyxy,
                        class_id=cls_id,
                        class_name=names.get(cls_id, f"class_{cls_id}"),
                        confidence=conf_score,
                    )
                )

        h, w = result.orig_shape
        return DetectionResult(detections=detections, image_shape=(h, w))
