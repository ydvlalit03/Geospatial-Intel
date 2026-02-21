"""FastAPI application for the Geospatial Intelligence Platform."""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from src.api.routes import detect, query, segment
from src.api.schemas import HealthResponse

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, clean up on shutdown."""
    device = os.getenv("MODEL_DEVICE", "cpu")

    # Load segmentation model
    unet_path = os.getenv("UNET_WEIGHTS_PATH", "weights/unet_sentinel2.pth")
    try:
        from src.inference.segmentation import SegmentationPipeline

        segment.pipeline = SegmentationPipeline(
            weights_path=unet_path, device=device
        )
        print(f"Segmentation model loaded (device={device})")
    except Exception as e:
        print(f"Warning: Could not load segmentation model: {e}")
        segment.pipeline = None

    # Load detection model
    yolo_path = os.getenv("YOLO_WEIGHTS_PATH", "weights/yolov8_satellite.pt")
    try:
        from src.inference.detection import DetectionPipeline

        detect.pipeline = DetectionPipeline(
            weights_path=yolo_path, device=device
        )
        print(f"Detection model loaded (device={device})")
    except Exception as e:
        print(f"Warning: Could not load detection model: {e}")
        detect.pipeline = None

    # Initialize LangGraph agent
    try:
        from src.agent.graph import build_agent_graph

        query.agent_graph = build_agent_graph()
        print("LangGraph agent initialized")
    except Exception as e:
        print(f"Warning: Could not initialize agent: {e}")
        query.agent_graph = None

    yield

    # Cleanup
    segment.pipeline = None
    detect.pipeline = None
    query.agent_graph = None


app = FastAPI(
    title="Geospatial Intelligence Platform",
    description="3D Geospatial Intelligence Platform — satellite image segmentation, "
    "object detection, and natural language querying via LangGraph agent.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(segment.router, tags=["Segmentation"])
app.include_router(detect.router, tags=["Detection"])
app.include_router(query.router, tags=["Agent"])


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded={
            "segmentation": segment.pipeline is not None,
            "detection": detect.pipeline is not None,
            "agent": query.agent_graph is not None,
        },
    )
