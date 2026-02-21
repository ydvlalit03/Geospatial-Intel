"""Detection endpoint."""

import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas import DetectionResponse
from src.inference.detection import DetectionPipeline

router = APIRouter()

# Will be set during app startup
pipeline: DetectionPipeline | None = None


@router.post("/predict/detect", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    """Upload a satellite image and get object detections."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Detection model not loaded")

    contents = await file.read()

    suffix = "." + (file.filename or "image.png").split(".")[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = pipeline.predict(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return DetectionResponse(**result)
