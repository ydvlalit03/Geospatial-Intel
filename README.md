# Geospatial Intelligence Platform

3D Geospatial Intelligence Platform that analyzes satellite imagery using U-Net (segmentation) and YOLOv8 (object detection), served via FastAPI, with a LangGraph AI agent for natural language querying.

## Architecture

- **U-Net** — Land cover segmentation (buildings, roads, vegetation, water, barren)
- **YOLOv8** — Object detection (vehicles, structures, ships)
- **FastAPI** — REST API with `/predict/segment`, `/predict/detect`, `/query` endpoints
- **LangGraph** — AI agent for natural language geospatial queries
- **Docker** — Containerized deployment

## Quick Start

```bash
# 1. Clone and setup
cd ~/projects/geospatial-intel
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Create .env from example
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 3. Download/create model weights
python weights/download_weights.py

# 4. Run the API
uvicorn src.api.main:app --reload

# 5. Open Swagger docs
open http://localhost:8000/docs
```

## Docker

```bash
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/predict/segment` | POST | Upload image → segmentation mask + class distribution |
| `/predict/detect` | POST | Upload image → bounding boxes + class counts |
| `/query` | POST | Natural language query → AI agent response |

## Training (Google Colab)

Training notebooks are in `notebooks/`:
1. `01_data_exploration.ipynb` — Visualize satellite bands, compute NDVI
2. `02_unet_training.ipynb` — Train U-Net segmentation model
3. `03_yolov8_training.ipynb` — Fine-tune YOLOv8 for satellite detection

## Project Structure

```
src/
├── models/          # U-Net architecture + YOLOv8 wrapper
├── data/            # Preprocessing, datasets, augmentation
├── inference/       # Segmentation + detection pipelines
├── agent/           # LangGraph agent (parse → execute → respond)
└── api/             # FastAPI application + routes
```

## Tests

```bash
pytest tests/ -v
```

## Tech Stack

PyTorch, Ultralytics YOLOv8, FastAPI, LangGraph, LangChain, rasterio, albumentations, Docker
