"""ML tools that the LangGraph agent can call."""

from langchain_core.tools import tool


@tool
def run_segmentation(image_path: str) -> dict:
    """Run land cover segmentation on a satellite image.

    Args:
        image_path: Path to the satellite image file.

    Returns:
        Segmentation results with class distribution.
    """
    from src.inference.segmentation import SegmentationPipeline

    pipeline = SegmentationPipeline()
    result = pipeline.predict(image_path)
    # Return serializable result (no numpy array)
    return {
        "class_distribution": result["class_distribution"],
        "class_names": result["class_names"],
        "image_size": result["image_size"],
    }


@tool
def run_detection(image_path: str, confidence: float = 0.25) -> dict:
    """Run object detection on a satellite image.

    Args:
        image_path: Path to the satellite image file.
        confidence: Minimum confidence threshold for detections.

    Returns:
        Detection results with bounding boxes and class counts.
    """
    from src.inference.detection import DetectionPipeline

    pipeline = DetectionPipeline()
    return pipeline.predict(image_path, confidence_threshold=confidence)


@tool
def compute_vegetation_index(image_path: str) -> dict:
    """Compute NDVI (vegetation index) for a satellite image.

    Args:
        image_path: Path to a multi-band satellite image.

    Returns:
        NDVI statistics (mean, min, max).
    """
    import numpy as np

    from src.data.preprocessing import compute_ndvi, read_geotiff

    image, meta = read_geotiff(image_path)

    if image.shape[0] >= 4:
        red = image[2]  # Band 3 (red) for Sentinel-2
        nir = image[3]  # Band 4 (NIR) for Sentinel-2
    else:
        # Fallback: use first two bands as approximation
        red = image[0]
        nir = image[-1]

    ndvi = compute_ndvi(red, nir)

    return {
        "mean_ndvi": float(np.mean(ndvi)),
        "min_ndvi": float(np.min(ndvi)),
        "max_ndvi": float(np.max(ndvi)),
        "vegetation_coverage": float(np.mean(ndvi > 0.3)),
    }


ALL_TOOLS = [run_segmentation, run_detection, compute_vegetation_index]
