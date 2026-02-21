"""Segmentation endpoint."""

import io
import tempfile

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas import SegmentationResponse
from src.inference.segmentation import SegmentationPipeline

router = APIRouter()

# Will be set during app startup
pipeline: SegmentationPipeline | None = None


@router.post("/predict/segment", response_model=SegmentationResponse)
async def segment_image(file: UploadFile = File(...)):
    """Upload a satellite image and get land cover segmentation."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")

    contents = await file.read()

    # Save to temp file for processing
    suffix = "." + (file.filename or "image.png").split(".")[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = pipeline.predict(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return SegmentationResponse(
        class_distribution=result["class_distribution"],
        class_names=result["class_names"],
        image_size=result["image_size"],
        mask_shape=list(result["mask"].shape),
    )
