"""
==========================================================================
  model_manager.py  –  Centralized Model Manager with Hot-Reload
==========================================================================
Manages all trained models (image / voice / text) and the default
MobileNet-SSD.  Provides a single lock-free read path so the video
streamer never blocks, while a background watcher thread picks up
new checkpoints from disk.

Author : Akshay Raj
Project: Smart Glasses – Object Detection
==========================================================================
"""

import os
import json
import hashlib
import threading
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

logger = logging.getLogger("model_manager")

# ── paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── YOLOv4-tiny (80 COCO classes including "cell phone") ────────────────
_YOLO_DIR     = PROJECT_ROOT / "yolo-coco"
_YOLO_CFG     = _YOLO_DIR / "yolov4-tiny.cfg"
_YOLO_WEIGHTS = _YOLO_DIR / "yolov4-tiny.weights"
_YOLO_NAMES   = _YOLO_DIR / "coco.names"

# ── MobileNet-SSD fallback (20 VOC classes) ─────────────────────────────
_SSD_DIR       = PROJECT_ROOT / "real-time-object-detection"
_SSD_PROTO     = _SSD_DIR / "MobileNetSSD_deploy.prototxt.txt"
_SSD_WEIGHTS   = _SSD_DIR / "MobileNetSSD_deploy.caffemodel"

SSD_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]

# Load COCO names for YOLO
COCO_CLASSES = []
if _YOLO_NAMES.exists():
    COCO_CLASSES = [c.strip() for c in _YOLO_NAMES.read_text(encoding="utf-8-sig").splitlines() if c.strip()]
COCO_COLORS = np.random.uniform(0, 255, size=(max(len(COCO_CLASSES), 80), 3))
SSD_COLORS  = np.random.uniform(0, 255, size=(len(SSD_CLASSES), 3))


class ModelManager:
    """Thread-safe model store with automatic hot-reload from disk."""

    def __init__(self):
        self._lock = threading.Lock()

        # ── YOLO primary detector ──
        self.yolo_net = None
        self.yolo_output_layers = None
        self._load_yolo()

        # ── SSD fallback detector ──
        self.ssd_net = None
        if self.yolo_net is None:
            self._load_ssd()

        # ── custom trained models ──
        self._image_model = None   # (torch_model, classes, img_size, device)
        self._image_hash  = None

        self._text_model  = None   # {"vectorizer": …, "classifier": …}
        self._text_meta   = None
        self._text_hash   = None

        self._voice_model = None
        self._voice_meta  = None
        self._voice_hash  = None

        # pre-load anything already on disk (graceful -- torch may be broken)
        for loader in (self._try_load_image_model, self._try_load_text_model, self._try_load_voice_model):
            try:
                loader()
            except Exception as e:
                logger.warning(f"Skipping {loader.__name__}: {e}")

        # background watcher
        self._stop_event = threading.Event()
        self._watcher = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher.start()

    # ── YOLO ─────────────────────────────────────────────────────────────
    def _load_yolo(self):
        if _YOLO_CFG.exists() and _YOLO_WEIGHTS.exists() and _YOLO_WEIGHTS.stat().st_size > 1_000_000:
            try:
                net = cv2.dnn.readNetFromDarknet(str(_YOLO_CFG), str(_YOLO_WEIGHTS))
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                ln = net.getLayerNames()
                out_layers = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
                self.yolo_net = net
                self.yolo_output_layers = out_layers
                logger.info(f"YOLOv4-tiny loaded [OK] -- {len(COCO_CLASSES)} COCO classes (incl. cell phone)")
            except Exception as e:
                logger.error(f"Failed to load YOLOv4-tiny: {e}")
        else:
            logger.warning("YOLOv4-tiny files not found -- will try MobileNet-SSD fallback")

    # ── SSD (fallback) ───────────────────────────────────────────────────
    def _load_ssd(self):
        if _SSD_PROTO.exists() and _SSD_WEIGHTS.exists():
            try:
                self.ssd_net = cv2.dnn.readNetFromCaffe(
                    str(_SSD_PROTO), str(_SSD_WEIGHTS))
                logger.info("MobileNet-SSD loaded [OK] (fallback)")
            except Exception as e:
                logger.error(f"Failed to load MobileNet-SSD: {e}")
        else:
            logger.warning("MobileNet-SSD files not found -- SSD detection disabled")

    # ── Image Model (PyTorch MobileNetV2 or Embedding) ─────────────────────
    def _try_load_image_model(self):
        model_path = MODELS_DIR / "image_model" / "model.pth"
        meta_path  = MODELS_DIR / "image_model" / "meta.json"
        if not model_path.exists() or not meta_path.exists():
            return
        h = hashlib.md5(model_path.read_bytes()).hexdigest()
        if h == self._image_hash:
            return
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models

            meta     = json.loads(meta_path.read_text())
            classes  = meta["classes"]
            img_size = meta.get("img_size", 224)
            device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_type = meta.get("model_type", "classifier")

            if model_type == "embedding":
                # Embedding-distance model: load centroid + threshold
                emb_path = MODELS_DIR / "image_model" / "embedding.npz"
                data = np.load(str(emb_path))
                centroid  = data["centroid"]
                threshold = float(data["threshold"])
                target_class = meta.get("target_class", classes[-1])

                # Load pretrained feature extractor
                model = models.mobilenet_v2(pretrained=True)
                model.classifier = nn.Identity()
                model.to(device).eval()

                with self._lock:
                    self._image_model = {
                        "type": "embedding",
                        "model": model,
                        "centroid": centroid,
                        "threshold": threshold,
                        "target_class": target_class,
                        "classes": classes,
                        "img_size": img_size,
                        "device": device,
                    }
                    self._image_hash = h
                logger.info(f"[HOT-RELOAD] Embedding model loaded – target={target_class}, threshold={threshold:.4f}")

            else:
                # Standard classifier model
                model = models.mobilenet_v2(pretrained=False)
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(model.last_channel, len(classes)),
                )
                model.load_state_dict(torch.load(str(model_path), map_location=device))
                model.to(device).eval()

                with self._lock:
                    self._image_model = {
                        "type": "classifier",
                        "model": model,
                        "classes": classes,
                        "img_size": img_size,
                        "device": device,
                    }
                    self._image_hash = h
                logger.info(f"[HOT-RELOAD] Classifier model loaded – classes={classes}")

        except Exception as e:
            logger.error(f"Image model load failed: {e}")

    # ── Text Model (sklearn) ─────────────────────────────────────────────
    def _try_load_text_model(self):
        model_path = MODELS_DIR / "text_model" / "model.pkl"
        meta_path  = MODELS_DIR / "text_model" / "meta.json"
        if not model_path.exists():
            return
        h = hashlib.md5(model_path.read_bytes()).hexdigest()
        if h == self._text_hash:
            return
        try:
            import pickle
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            with self._lock:
                self._text_model = obj
                self._text_meta  = meta
                self._text_hash  = h
            logger.info("[HOT-RELOAD] Text model loaded [OK]")
        except Exception as e:
            logger.error(f"Text model load failed: {e}")

    # ── Voice Model (PyTorch CNN) ────────────────────────────────────────
    def _try_load_voice_model(self):
        model_path = MODELS_DIR / "voice_model" / "model.pth"
        meta_path  = MODELS_DIR / "voice_model" / "meta.json"
        if not model_path.exists():
            return
        h = hashlib.md5(model_path.read_bytes()).hexdigest()
        if h == self._voice_hash:
            return
        try:
            import torch, torch.nn as nn
            meta    = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            classes = meta.get("classes", [])
            device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            class AudioCNN(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv1d(40, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                        nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                    )
                    self.fc = nn.Linear(128, num_classes)
                def forward(self, x):
                    x = self.conv(x); x = x.squeeze(-1); return self.fc(x)

            model = AudioCNN(len(classes)).to(device)
            model.load_state_dict(torch.load(str(model_path), map_location=device))
            model.eval()
            with self._lock:
                self._voice_model = (model, classes, device)
                self._voice_meta  = meta
                self._voice_hash  = h
            logger.info(f"[HOT-RELOAD] Voice model loaded – classes={classes}")
        except Exception as e:
            logger.error(f"Voice model load failed: {e}")

    # ── Background watcher ───────────────────────────────────────────────
    def _watch_loop(self):
        """Poll disk every 3 seconds for new / changed model files."""
        import time
        while not self._stop_event.is_set():
            for loader in (self._try_load_image_model, self._try_load_text_model, self._try_load_voice_model):
                try:
                    loader()
                except Exception as e:
                    logger.debug(f"Watcher skip {loader.__name__}: {e}")
            time.sleep(3)

    def stop(self):
        self._stop_event.set()

    # ── Public inference helpers ─────────────────────────────────────────

    def detect_objects(self, frame, confidence_threshold=0.3, nms_threshold=0.4):
        """Run object detection. Uses YOLO if available, else SSD fallback.
        Returns list of {label, confidence, box, color}."""
        if self.yolo_net is not None:
            return self._detect_yolo(frame, confidence_threshold, nms_threshold)
        return self._detect_ssd(frame, confidence_threshold)

    # keep old name as alias for backward compat
    def detect_ssd(self, frame, confidence_threshold=0.3):
        return self.detect_objects(frame, confidence_threshold)

    def _detect_yolo(self, frame, conf_thresh=0.3, nms_thresh=0.4):
        """YOLOv4-tiny detection -- 80 COCO classes."""
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.yolo_net.setInput(blob)
        layer_outs = self.yolo_net.forward(self.yolo_output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in layer_outs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                if conf > conf_thresh:
                    cx, cy, bw, bh = (detection[0:4] * np.array([W, H, W, H])).astype(int)
                    x = int(cx - bw / 2)
                    y = int(cy - bh / 2)
                    boxes.append([x, y, int(bw), int(bh)])
                    confidences.append(conf)
                    class_ids.append(class_id)

        # Non-maximum suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, bw, bh = boxes[i]
                cid = class_ids[i]
                label = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
                results.append({
                    "label": label,
                    "confidence": confidences[i],
                    "box": [max(0,x), max(0,y), x+bw, y+bh],
                    "color": COCO_COLORS[cid].tolist(),
                })
        return results

    def _detect_ssd(self, frame, confidence_threshold=0.2):
        """MobileNet-SSD fallback -- 20 VOC classes."""
        if self.ssd_net is None:
            return []
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.ssd_net.setInput(blob)
        detections = self.ssd_net.forward()
        results = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                results.append({
                    "label": SSD_CLASSES[idx],
                    "confidence": conf,
                    "box": box.astype("int").tolist(),
                    "color": SSD_COLORS[idx].tolist(),
                })
        return results

    def classify_image_custom(self, frame):
        """Run custom image classifier. Returns (label, confidence) or None.
        Supports both 'classifier' (softmax) and 'embedding' (cosine distance) models.
        """
        with self._lock:
            m = self._image_model
        if m is None:
            return None
        import torch

        model_type = m.get("type", "classifier") if isinstance(m, dict) else "classifier"

        if model_type == "embedding":
            model    = m["model"]
            centroid = m["centroid"]
            threshold = m["threshold"]
            target   = m["target_class"]
            img_size = m["img_size"]
            device   = m["device"]

            img = cv2.resize(frame, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            tensor = torch.tensor(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model(tensor).squeeze().cpu().numpy()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            similarity = float(feat @ centroid)

            if similarity >= threshold:
                # Scale confidence: threshold=0% → max_sim=100%
                # Map from [threshold, 1.0] to [0.5, 1.0] for display
                conf = 0.5 + 0.5 * (similarity - threshold) / max(1.0 - threshold, 0.01)
                return target, min(conf, 0.99)
            else:
                conf = 0.5 * (1.0 - similarity / max(threshold, 0.01))
                return "background", min(max(conf, 0.5), 0.99)

        else:
            # Standard classifier
            model   = m["model"]
            classes = m["classes"]
            img_size = m["img_size"]
            device  = m["device"]

            img = cv2.resize(frame, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            tensor = torch.tensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out   = model(tensor)
                probs = torch.softmax(out, dim=1)
                conf, idx = probs.max(dim=1)
            return classes[idx.item()], conf.item()

    def get_custom_image_classes(self):
        """Return the list of classes for the custom image model, or None."""
        with self._lock:
            m = self._image_model
        if m is None:
            return None
        if isinstance(m, dict):
            return m.get("classes")
        return m[1]  # legacy tuple format

    def classify_text(self, text: str):
        """Run text classifier. Returns (label, confidence) or None."""
        with self._lock:
            m = self._text_model
        if m is None:
            return None
        vec  = m["vectorizer"].transform([text])
        pred = m["classifier"].predict(vec)[0]
        prob = max(m["classifier"].predict_proba(vec)[0])
        return pred, prob

    def get_models_info(self):
        """Return dict of all available model metadata."""
        info = {}
        for mt in ("image", "voice", "text"):
            meta_file = MODELS_DIR / f"{mt}_model" / "meta.json"
            if meta_file.exists():
                info[mt] = json.loads(meta_file.read_text())
            else:
                info[mt] = None
        info["ssd"] = {
            "loaded": self.yolo_net is not None or self.ssd_net is not None,
            "detector": "YOLOv4-tiny" if self.yolo_net else ("MobileNet-SSD" if self.ssd_net else "None"),
            "classes": COCO_CLASSES if self.yolo_net else SSD_CLASSES,
        }
        return info

    def force_reload(self):
        """Force reload all models now."""
        self._image_hash = None
        self._text_hash  = None
        self._voice_hash = None
        self._try_load_image_model()
        self._try_load_text_model()
        self._try_load_voice_model()


# singleton
manager = ModelManager()
