# SmartGlass AI – Object Detection with OpenCV

> Real-time object detection platform with YOLO, MobileNet-SSD, and custom model training — built for wearable AI smart glasses.

**By Akshay Raj**

---

## Features

- **Live Camera Detection** — YOLOv4-tiny (80 COCO classes) with real-time webcam streaming
- **MobileNet-SSD Fallback** — 20 VOC classes, ultra-lightweight
- **Custom Model Training** — Upload image/audio/text .zip datasets, auto-detect type, train & hot-reload
- **Adapter Switching** — Hot-swap between YOLO, MobileNet-SSD, and custom models from the UI
- **Elegant Web Dashboard** — Single-page app with live terminal, Chart.js training graphs, drag-and-drop upload
- **Image Detection** — Upload a single image for instant object detection
- **Server-Sent Events** — Real-time log streaming and training progress

---

## Quick Start

### 1. Clone & Setup (one command)

```bash
git clone https://github.com/AkshayRaj367/Akkiglasses.git
cd Akkiglasses
python setup.py
```

The setup script will:
- Create a `.venv` virtual environment
- Install all Python dependencies
- Download YOLOv4-tiny weights (~23 MB) if missing
- Create required directories
- Verify the installation

### 2. Run the App

**Windows (PowerShell):**
```powershell
& .venv\Scripts\Activate.ps1
cd webapp
python app.py
```

**macOS / Linux:**
```bash
source .venv/bin/activate
cd webapp
python app.py
```

### 3. Open in Browser

```
http://localhost:5000
```

---

## Project Structure

```
├── setup.py                     # Automated setup script
├── akshay_raj_README.md         # This file
├── webapp/
│   ├── app.py                   # Flask server (all API routes)
│   ├── model_manager.py         # Centralized model manager + hot-reload
│   ├── video_stream.py          # Webcam capture + detection + MJPEG
│   ├── trainer.py               # Auto-training adapter (image/voice/text)
│   ├── requirements.txt         # Python dependencies
│   ├── templates/index.html     # Frontend UI
│   ├── static/                  # Static assets
│   └── uploads/                 # User-uploaded datasets (gitignored)
├── yolo-coco/                   # YOLO config + weights
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights      # Downloaded by setup.py
│   └── coco.names
├── real-time-object-detection/  # MobileNet-SSD weights
│   ├── MobileNetSSD_deploy.prototxt.txt
│   └── MobileNetSSD_deploy.caffemodel
└── trained_models/              # Custom model checkpoints (gitignored)
    ├── image_model/
    ├── text_model/
    └── voice_model/
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web dashboard |
| `/api/video/start` | POST | Start webcam stream |
| `/api/video/stop` | POST | Stop webcam |
| `/api/video/feed` | GET | MJPEG video feed |
| `/api/video/detections` | GET | Current detection results + FPS |
| `/api/detect/image` | POST | Detect objects in uploaded image |
| `/api/upload` | POST | Upload .zip dataset |
| `/api/upload/clear` | POST | Clear uploaded dataset |
| `/api/train` | POST | Start model training |
| `/api/train/status/stream` | GET | SSE training progress |
| `/api/models` | GET | List all models |
| `/api/models/switch` | POST | Switch active adapter |
| `/api/models/history` | GET | Training history |
| `/api/models/reload` | POST | Force reload models |
| `/api/logs` | GET | Recent server logs |
| `/api/logs/stream` | GET | SSE server log stream |
| `/api/classify/text` | POST | Text classification |

---

## Requirements

- Python 3.10+
- Webcam (for live detection)
- ~500 MB disk space (dependencies + YOLO weights)

---

## How It Works

1. **Detection** — The app uses YOLOv4-tiny by default for fast 80-class object detection. If YOLO weights aren't available, it falls back to MobileNet-SSD (20 classes).

2. **Custom Training** — Upload a .zip file containing images (in class folders), audio files, or text CSVs. The trainer auto-detects the dataset type, builds the right model (MobileNetV2 for images, CNN for audio, TF-IDF+SVM for text), and saves it to `trained_models/`.

3. **Hot-Reload** — A background watcher thread polls `trained_models/` every 3 seconds. When a new checkpoint is found, it's loaded without restarting the server.

4. **Adapter Switching** — Switch between YOLO, SSD, and custom models from the top bar dropdown. Custom model predictions overlay on top of the base detector's bounding boxes.

---

*Originally forked from YOLO-object-detection-with-OpenCV. Rebuilt as a full-stack AI platform.*
