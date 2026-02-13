# 🕶️ Smart Glasses – AI Object Detection Web Application

## Documentation

### Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Features](#features)
6. [Data Flow](#data-flow)
7. [API Reference](#api-reference)
8. [Dataset Format Guide](#dataset-format-guide)
9. [Model Hot-Reload](#model-hot-reload)
10. [Configuration](#configuration)
11. [Troubleshooting](#troubleshooting)

---

## Overview

A full-stack web application for **real-time object detection** on Smart Glasses.  
Run a single command, open your browser, and you get:

- **Live webcam feed** with MobileNet-SSD object detection rendered in-browser
- **Drag-and-drop dataset upload** (image / voice / text)
- **Automatic training** – the system detects the dataset type and trains the right model
- **Live training dashboard** with loss/accuracy charts, progress bar, and epoch logs
- **Model hot-reload** – newly trained models are picked up within 3 seconds, no restart needed
- **Text classification playground** for testing trained text models
- **REST API** for programmatic access

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     BROWSER (UI)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │Live Video│  │Upload &  │  │ Models   │  │Playgnd │  │
│  │  Feed    │  │  Train   │  │Dashboard │  │        │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘  │
│       │              │             │             │       │
│  MJPEG stream   File upload    REST poll     REST API   │
│       │           + SSE           │             │       │
└───────┼──────────────┼─────────────┼─────────────┼───────┘
        │              │             │             │
┌───────┴──────────────┴─────────────┴─────────────┴───────┐
│                   FLASK SERVER  (app.py)                  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ VideoStream  │  │   Trainer    │  │ModelManager  │    │
│  │ (MJPEG pipe) │  │(bg thread)  │  │(hot-reload)  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │            │
│    Webcam (cv2)      PyTorch / sklearn    Disk watcher  │
│                           │                 │            │
│                     trained_models/  ←──────┘            │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.9+ 
- Webcam (for live detection)
- GPU (optional, speeds up training)

### Installation

```bash
cd webapp
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Project Structure

```
Akshay-object-detection-with-OpenCV/
│
├── webapp/                          ← NEW: Main web application
│   ├── app.py                       # Flask server – all routes & APIs
│   ├── video_stream.py              # Webcam → detection → MJPEG engine
│   ├── model_manager.py             # Model loading, hot-reload, inference
│   ├── trainer.py                   # Auto-training adapter (image/voice/text)
│   ├── requirements.txt             # Python dependencies
│   ├── templates/
│   │   └── index.html               # Full single-page dashboard
│   ├── static/                      # Static assets (CSS/JS if needed)
│   └── uploads/                     # Extracted uploaded datasets
│
├── trained_models/                  ← Auto-created after training
│   ├── image_model/
│   │   ├── model.pth                # PyTorch weights
│   │   └── meta.json                # Class names, accuracy, timestamp
│   ├── voice_model/
│   │   ├── model.pth
│   │   └── meta.json
│   └── text_model/
│       ├── model.pkl                # sklearn pipeline
│       └── meta.json
│
├── real-time-object-detection/      ← Original MobileNet-SSD files
│   ├── MobileNetSSD_deploy.prototxt.txt
│   ├── MobileNetSSD_deploy.caffemodel
│   └── akshay_raj_real_time_object_detection.py
│
├── yolo-coco/                       ← YOLO weights & config
├── adapter/                         ← Earlier adapter (standalone)
└── training/                        ← Earlier training scripts
```

---

## Features

### 1. Live Detection (Tab: 🎥 Live Detection)

| Feature | Detail |
|---------|--------|
| **MJPEG stream** | Webcam frames are captured, processed, and streamed to the `<img>` tag at ~30 fps |
| **MobileNet-SSD** | Always-on 20-class object detection (person, car, dog, etc.) |
| **Custom model overlay** | If you've trained an image classifier, its prediction shows as `[Custom] <label>` in the top-left |
| **Live detection list** | Right-side panel shows every detected object with color dot and confidence |
| **FPS counter** | Real-time throughput displayed on both the feed and the stats panel |

### 2. Upload & Train (Tab: 📦 Upload & Train)

| Feature | Detail |
|---------|--------|
| **Drag-and-drop upload** | Upload a `.zip` file; server extracts and auto-detects the dataset type |
| **Type detection** | Counts file extensions → votes on `image` / `voice` / `text` |
| **Configurable** | Epochs, batch size, learning rate, force-type override |
| **Background training** | Runs in a daemon thread; UI is never blocked |
| **SSE live updates** | Server-Sent Events push epoch progress, loss, accuracy to the browser in real-time |
| **Chart** | Chart.js line chart of loss + accuracy over epochs |
| **Training log** | Scrollable monospace log of every epoch |

### 3. Models Dashboard (Tab: 🤖 Models)

| Feature | Detail |
|---------|--------|
| **Models table** | Shows type, classes, accuracy, and trained-at timestamp |
| **Force hot-reload** | One-click button to make ModelManager re-scan disk |
| **Dataset format guide** | In-page reference for how to structure each dataset type |

### 4. Playground (Tab: 🧪 Playground)

| Feature | Detail |
|---------|--------|
| **Text classifier** | Type any sentence → get the predicted label + confidence |
| **API reference** | In-page table of all REST endpoints |

---

## Data Flow

### Upload → Train → Hot-Reload → Live Detection

```
1. User drags .zip onto the Upload zone
       │
       ▼
2. POST /api/upload
   - Saves & extracts zip → webapp/uploads/<name>/
   - Scans file extensions → returns dataset_type
       │
       ▼
3. User clicks "Start Training"
   POST /api/train  { dataset_path, epochs, … }
       │
       ▼
4. Background thread runs trainer.run_training()
   - ImageTrainer / VoiceTrainer / TextTrainer
   - StatusTracker updated each epoch
   - SSE stream pushes updates to browser
       │
       ▼
5. Model saved to trained_models/<type>_model/
   - model.pth (or .pkl)  +  meta.json
       │
       ▼
6. ModelManager background watcher (every 3s)
   - Detects file hash change → loads new weights
   - No server restart needed
       │
       ▼
7. VideoStream's next frame uses updated model
   - SSD detections + Custom model overlay
   - Browser sees new predictions immediately
```

---

## API Reference

### Video

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/video/start` | Start webcam capture thread |
| `POST` | `/api/video/stop` | Stop webcam capture |
| `GET` | `/api/video/feed` | MJPEG stream (use as `<img src>`) |
| `GET` | `/api/video/detections` | JSON: `{ fps, detections: [...] }` |

### Upload & Training

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/api/upload` | `multipart/form-data` with `file` | Upload .zip dataset |
| `POST` | `/api/train` | `{ dataset_path, type?, epochs?, batch_size?, lr? }` | Start training |
| `GET` | `/api/train/status` | — | Current training status JSON |
| `GET` | `/api/train/status/stream` | — | SSE stream of training events |

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/models` | All model metadata |
| `POST` | `/api/models/reload` | Force hot-reload from disk |

### Classification

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/api/classify/text` | `{ text: "..." }` | Classify text with trained model |

---

## Dataset Format Guide

### Image Dataset
```
my_images.zip
└── my_images/
    ├── cat/
    │   ├── img001.jpg
    │   └── img002.png
    ├── dog/
    │   └── img003.jpg
    └── bird/
        └── img004.jpeg
```
- **Minimum**: 2 class folders, each with at least a few images
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.webp`
- **Training**: MobileNetV2 fine-tuned (transfer learning)

### Voice / Audio Dataset
```
my_audio.zip
└── my_audio/
    ├── hello/
    │   ├── hello_01.wav
    │   └── hello_02.mp3
    └── stop/
        ├── stop_01.wav
        └── stop_02.flac
```
- **Supported formats**: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`
- **Training**: MFCC feature extraction → 1D-CNN classifier

### Text Dataset (folders)
```
my_text.zip
└── my_text/
    ├── positive/
    │   ├── doc1.txt
    │   └── doc2.txt
    └── negative/
        └── doc3.txt
```

### Text Dataset (CSV)
```
my_csv.zip
└── data.csv

# CSV format:
text,label
"This product is amazing!",positive
"Worst purchase ever.",negative
```
- **Training**: TF-IDF vectorizer → Multinomial Naive Bayes

---

## Model Hot-Reload

The `ModelManager` runs a background daemon thread that:

1. Checks `trained_models/<type>_model/model.pth` (or `.pkl`) every **3 seconds**
2. Computes an MD5 hash of the file
3. If the hash differs from the cached one → **reload the model into memory**
4. The `VideoStream` immediately uses the new model on the next frame

This means:
- **No server restart** after training
- **No manual reload** needed (though you can click "Force Hot-Reload")
- The live video feed will show custom model predictions as soon as training finishes

---

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 10 (image), 20 (voice), 1 (text) | Number of training epochs |
| `batch_size` | 16 (image), 32 (voice) | Mini-batch size |
| `lr` | 0.001 | Learning rate (Adam optimizer) |
| `type` | auto-detected | Force `image`, `voice`, or `text` |

### Server Settings

| Setting | Default | Location |
|---------|---------|----------|
| Port | 5000 | `app.py` – `app.run(port=5000)` |
| Max upload | 2 GB | `app.py` – `MAX_CONTENT_LENGTH` |
| MJPEG quality | 80% | `video_stream.py` – `IMWRITE_JPEG_QUALITY` |
| Detection threshold | 0.2 | `video_stream.py` – `confidence` param |
| Hot-reload interval | 3 sec | `model_manager.py` – `time.sleep(3)` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot open webcam" | Make sure no other app is using the camera. Try changing `src=0` to `src=1` in `app.py`. |
| MobileNet-SSD not found | Ensure `real-time-object-detection/MobileNetSSD_deploy.caffemodel` exists (check Git LFS). |
| Training fails with "need ≥ 2 sub-folders" | Your zip must contain class sub-folders (e.g., `cat/`, `dog/`), not loose files. |
| "PyTorch required" | Run `pip install torch torchvision`. |
| "librosa required" | Run `pip install librosa` (only needed for voice datasets). |
| Upload fails with "Invalid zip" | Make sure the file is a valid `.zip` archive, not `.rar` or `.7z`. |
| Port 5000 in use | Change the port in `app.py`: `app.run(port=8080)`. |
| Low FPS | Use a smaller webcam resolution or ensure GPU is available for custom models. |

---

## License

This project is for educational purposes as part of the Smart Glasses project by Akshay Raj.
