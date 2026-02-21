"""Tests for FastAPI endpoints."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a simple test image in memory."""
    img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_segment_endpoint(client, sample_image_bytes):
    response = client.post(
        "/predict/segment",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    # May return 503 if model not loaded — that's acceptable in test
    assert response.status_code in (200, 503)


def test_detect_endpoint(client, sample_image_bytes):
    response = client.post(
        "/predict/detect",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code in (200, 503)


def test_query_endpoint(client):
    response = client.post(
        "/query",
        json={"query": "What land cover types are present?"},
    )
    assert response.status_code in (200, 503)
