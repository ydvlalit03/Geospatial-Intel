# Geospatial Intelligence Platform

3D Geospatial Intelligence Platform that analyzes satellite imagery using U-Net (segmentation) and YOLOv8 (object detection), served via FastAPI, with a LangGraph AI agent for natural language querying.

## Architecture

- **U-Net** — Land cover segmentation (background, building, woodland, water, road)
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

## Training Data

### U-Net — Land Cover Segmentation

The U-Net model is trained on **[LandCover.ai](https://landcover.ai/)** high-resolution aerial imagery for 5-class land cover segmentation:
`background`, `building`, `woodland`, `water`, `road`.

The dataset expects paired image + mask directories:
```
dataset/
├── images/    # RGB satellite patches (256x256 PNG/TIF)
└── masks/     # Single-channel label masks (pixel value = class ID)
```

**Recommended datasets:**
- [LandCover.ai](https://landcover.ai/) — High-resolution aerial imagery with building, woodland, water, and road annotations
- [DeepGlobe Land Cover](https://competitions.codalab.org/competitions/18468) — Satellite images with 7 land cover classes
- [Sentinel-2 Land Use/Land Cover](https://livingatlas.arcgis.com/landcoverexplorer/) — Global 10m resolution land cover derived from Sentinel-2

The training notebook (`02_unet_training.ipynb`) automatically downloads and prepares LandCover.ai for training.

### YOLOv8 — Object Detection

The YOLOv8 model starts from **COCO-pretrained weights** (`yolov8n.pt`) and is fine-tuned on satellite imagery for detecting objects like vehicles, structures, and ships.

The dataset must follow standard YOLO format:
```
dataset/
├── train/
│   ├── images/    # Satellite image patches
│   └── labels/    # YOLO .txt annotations (class x_center y_center width height)
├── val/
│   ├── images/
│   └── labels/
└── data.yaml      # Class names and paths
```

**Recommended datasets:**
- [DOTA](https://captain-whu.github.io/DOTA/) — Large-scale dataset for object detection in aerial images (15 categories)
- [xView](http://xviewdataset.org/) — One of the largest overhead imagery datasets (60 classes, 1M+ objects) — **used by default**
- [DIOR](https://gcheng-nwpu.github.io/#Datasets) — 20 object classes in optical remote sensing images

The training notebook (`03_yolov8_training.ipynb`) downloads xView via the Kaggle API, converts annotations to YOLO format, and fine-tunes YOLOv8 on 60 object classes.

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
