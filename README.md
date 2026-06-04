# Geospatial Intelligence Platform

A 3D geospatial intelligence platform that analyzes satellite imagery using deep learning — **U-Net** for land-cover segmentation and **YOLOv8** for object detection — served through a **FastAPI** REST API, with a **LangGraph** agent that answers natural-language questions about the imagery.

---

## Features

- **Land-cover segmentation** — U-Net classifies each pixel into `background`, `building`, `woodland`, `water`, `road`
- **Object detection** — YOLOv8 detects vehicles, structures, ships and 60+ classes (fine-tuned from COCO weights)
- **Natural-language querying** — a LangGraph agent (parse → execute → respond) lets you ask questions in plain English
- **REST API** — clean endpoints for segmentation, detection and querying, with Swagger docs
- **Containerized** — one-command Docker Compose deployment

---

## Tech Stack

- **Models**: PyTorch (U-Net), Ultralytics YOLOv8
- **API**: FastAPI + Uvicorn
- **Agent**: LangGraph, LangChain
- **Geo / imaging**: rasterio, albumentations
- **Deploy**: Docker, docker-compose

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/predict/segment` | POST | Image → segmentation mask + class distribution |
| `/predict/detect` | POST | Image → bounding boxes + class counts |
| `/query` | POST | Natural-language query → AI agent response |

---

## Quick Start

### Prerequisites

- Python 3.10+
- An LLM API key (for the query agent)

### Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # add your OPENAI_API_KEY
python weights/download_weights.py

uvicorn src.api.main:app --reload
open http://localhost:8000/docs
```

### Docker

```bash
docker-compose up --build
```

---

## Training

Training notebooks live in `notebooks/`:

1. `01_data_exploration.ipynb` — visualize satellite bands, compute NDVI
2. `02_unet_training.ipynb` — train U-Net on [LandCover.ai](https://landcover.ai/) (5-class)
3. `03_yolov8_training.ipynb` — fine-tune YOLOv8 on [xView](http://xviewdataset.org/) (60 classes)

---

## Project Structure

```
src/
├── models/       # U-Net architecture + YOLOv8 wrapper
├── data/         # preprocessing, datasets, augmentation
├── inference/    # segmentation + detection pipelines
├── agent/        # LangGraph agent (parse → execute → respond)
└── api/          # FastAPI application + routes
```
