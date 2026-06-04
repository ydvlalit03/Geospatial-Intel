<div align="center">

# 🛰️ Geospatial Intelligence Platform

### Analyze satellite imagery with deep learning — segmentation, detection, and a natural-language AI agent

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

</div>

---

## 📖 Overview

The **Geospatial Intelligence Platform** turns raw satellite and aerial imagery into structured insight. It combines two complementary computer-vision models — a **U-Net** for pixel-level land-cover segmentation and a **YOLOv8** detector for objects — behind a clean **FastAPI** service. On top of that sits a **LangGraph agent** that lets you ask questions in plain English ("how much of this image is water?", "count the vehicles") and routes them to the right model.

It's a complete pipeline: data preparation and training notebooks, model inference, a REST API, an agentic natural-language layer, and a Docker deployment — the kind of system you'd actually put in front of an analyst.

---

## 📑 Table of Contents

- [Capabilities](#-capabilities)
- [Architecture](#-architecture)
- [The models](#-the-models)
- [API reference](#-api-reference)
- [Tech stack](#-tech-stack)
- [Installation](#-installation)
- [Training](#-training)
- [Project structure](#-project-structure)
- [Testing](#-testing)

---

## ✨ Capabilities

- **🗺️ Land-cover segmentation** — every pixel classified into one of 5 classes: `background`, `building`, `woodland`, `water`, `road`, returned as a mask plus a class-distribution summary
- **🎯 Object detection** — bounding boxes for vehicles, structures, ships and 60+ classes, with per-class counts
- **💬 Natural-language querying** — a LangGraph agent parses your question, executes the right model(s), and responds conversationally
- **🔌 Clean REST API** — typed FastAPI endpoints with auto-generated Swagger docs
- **📓 Reproducible training** — Colab-ready notebooks that download datasets, train, and export weights
- **🐳 One-command deployment** — Docker Compose for the full stack

---

## 🏗️ Architecture

```
                         ┌────────────────────────────────────┐
   image / question ───▶ │            FastAPI service          │
                         │                                      │
                         │   /predict/segment  ─▶  U-Net        │
                         │   /predict/detect   ─▶  YOLOv8       │
                         │   /query            ─▶  LangGraph    │
                         │                          agent       │
                         │                            │         │
                         │              parse ─▶ execute ─▶ respond
                         └────────────────────────────────────┘
```

The query agent follows a **parse → execute → respond** loop: it interprets the natural-language request, decides whether segmentation, detection, or both are needed, calls the underlying inference pipelines, and composes a human-readable answer.

---

## 🧠 The models

### U-Net — Land-Cover Segmentation
Trained on **[LandCover.ai](https://landcover.ai/)** high-resolution aerial imagery for 5-class semantic segmentation. Expects paired image/mask directories:

```
dataset/
├── images/    # RGB satellite patches (256×256 PNG/TIF)
└── masks/     # single-channel label masks (pixel value = class ID)
```

**Compatible datasets:** LandCover.ai, [DeepGlobe Land Cover](https://competitions.codalab.org/competitions/18468), Sentinel-2 LULC.

### YOLOv8 — Object Detection
Starts from **COCO-pretrained** `yolov8n.pt` and is fine-tuned on satellite imagery. Standard YOLO dataset layout:

```
dataset/
├── train/{images,labels}/   # labels: class x_center y_center width height
├── val/{images,labels}/
└── data.yaml                # class names + paths
```

**Compatible datasets:** [xView](http://xviewdataset.org/) (60 classes — used by default), [DOTA](https://captain-whu.github.io/DOTA/), [DIOR](https://gcheng-nwpu.github.io/).

---

## 🌐 API reference

| Endpoint | Method | Description | Returns |
|----------|--------|-------------|---------|
| `/health` | GET | Health check + model load status | service & model state |
| `/predict/segment` | POST | Upload an image | segmentation mask + class distribution |
| `/predict/detect` | POST | Upload an image | bounding boxes + per-class counts |
| `/query` | POST | Natural-language question | AI agent response |

Interactive Swagger UI is available at `/docs` once the server is running.

---

## 🛠️ Tech stack

| Concern | Technology |
|---------|------------|
| **Segmentation** | PyTorch (custom U-Net) |
| **Detection** | Ultralytics YOLOv8 |
| **API** | FastAPI + Uvicorn |
| **Agent** | LangGraph, LangChain |
| **Geo / imaging** | rasterio, albumentations |
| **Deployment** | Docker, docker-compose |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- An LLM API key (for the `/query` agent)

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

## 🎓 Training

Training notebooks live in `notebooks/` and are written to run on Google Colab:

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | visualize satellite bands, compute NDVI |
| `02_unet_training.ipynb` | download LandCover.ai, train the U-Net |
| `03_yolov8_training.ipynb` | pull xView via Kaggle API, convert to YOLO format, fine-tune YOLOv8 |

---

## 🗂️ Project structure

```
src/
├── models/        # U-Net architecture + YOLOv8 wrapper
├── data/          # preprocessing, datasets, augmentation
├── inference/     # segmentation + detection pipelines
├── agent/         # LangGraph agent (parse → execute → respond)
└── api/           # FastAPI application + routes
notebooks/         # data exploration + training
weights/           # model weights + download script
tests/
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```
