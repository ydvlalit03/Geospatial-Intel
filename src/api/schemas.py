"""Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


class SegmentationResponse(BaseModel):
    class_distribution: dict[str, float] = Field(
        ..., description="Percentage of each land cover class"
    )
    class_names: dict[int, str] = Field(..., description="Class ID to name mapping")
    image_size: list[int] = Field(..., description="[height, width] of the output mask")
    mask_shape: list[int] = Field(..., description="Shape of the segmentation mask")


class DetectionBox(BaseModel):
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2]")
    class_id: int
    class_name: str
    confidence: float


class DetectionResponse(BaseModel):
    num_detections: int
    image_shape: list[int]
    detections: list[DetectionBox]
    class_summary: dict[str, int] = Field(
        ..., description="Count of detections per class"
    )


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about satellite imagery")
    image_path: str | None = Field(
        None, description="Optional path to an image to analyze"
    )


class QueryResponse(BaseModel):
    query: str
    response: str
    analysis: dict | None = Field(None, description="Structured analysis results")


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
