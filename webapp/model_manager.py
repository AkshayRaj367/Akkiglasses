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
import shutil
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
# ── YOLOv11 (Ultralytics – 80 COCO classes, state-of-the-art) ──────────────
_YOLO11_DIR    = PROJECT_ROOT / "yolo-coco"
_YOLO11_PT     = _YOLO11_DIR / "yolo11n.pt"       # nano model (~6 MB)
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


# =====================================================================
#  MODEL REGISTRY  –  multi-model storage with versioning
# =====================================================================

class ModelRegistry:
    """
    Persistent registry of all trained models.

    Every trained model is copied into  trained_models/models/<unique_id>/
    alongside a registry.json index.  The 'active' model for each type is
    the one whose files live in  trained_models/<type>_model/  (unchanged
    from the original layout, so hot-reload keeps working).

    Switching the active model copies its files into the type directory
    and the ModelManager's watcher picks up the MD5 change.
    """

    REGISTRY_FILE = MODELS_DIR / "registry.json"
    MODELS_STORE  = MODELS_DIR / "models"

    def __init__(self):
        self.MODELS_STORE.mkdir(parents=True, exist_ok=True)
        self._data = self._load()
        # Auto-import any existing active model not yet in the registry
        self._auto_import_existing()

    def _load(self):
        if self.REGISTRY_FILE.exists():
            try:
                return json.loads(self.REGISTRY_FILE.read_text())
            except Exception:
                return {"models": []}
        return {"models": []}

    def _save(self):
        self.REGISTRY_FILE.write_text(
            json.dumps(self._data, indent=2, default=str))

    def _auto_import_existing(self):
        """If there's an active model on disk that isn't in the registry,
        import it so the UI shows it."""
        existing_ids = {m["id"] for m in self._data["models"]}
        for mt in ("image", "text", "voice", "yolo_detection"):
            model_dir = (MODELS_DIR / "yolo_custom" if mt == "yolo_detection"
                         else MODELS_DIR / f"{mt}_model")
            meta_f = model_dir / "meta.json"
            if not meta_f.exists():
                continue
            try:
                meta = json.loads(meta_f.read_text())
            except Exception:
                continue
            # Check if already registered
            already = False
            for m in self._data["models"]:
                if (m["type"] == mt
                        and m.get("meta", {}).get("trained_at")
                            == meta.get("trained_at")):
                    already = True
                    break
            if already:
                continue
            # Register it
            ts = meta.get("trained_at", datetime.now().isoformat())
            classes = meta.get("classes", [])
            name = ", ".join(classes[:3]) if classes else mt
            mid = f"{mt[:3]}_{ts.replace(':', '').replace('-', '')[:15]}_{len(classes)}cls"
            if mid in existing_ids:
                continue
            dst = self.MODELS_STORE / mid
            dst.mkdir(parents=True, exist_ok=True)
            for f in model_dir.iterdir():
                if f.is_file():
                    shutil.copy2(str(f), str(dst / f.name))
            entry = {
                "id": mid, "type": mt, "name": name,
                "classes": classes,
                "accuracy": meta.get("accuracy"),
                "trained_at": ts, "active": True,
                "path": str(dst), "meta": meta,
            }
            for m in self._data["models"]:
                if m["type"] == mt:
                    m["active"] = False
            self._data["models"].append(entry)
        self._save()

    # ── public API ───────────────────────────────────────────────────

    def register(self, model_type: str, name: str, classes: list,
                 accuracy: float, meta: dict) -> str:
        """Register a newly trained model. Returns model ID."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_cls = len(classes) if classes else 0
        clean = name.replace(" ", "_").lower()[:20]
        model_id = f"{model_type[:3]}_{ts}_{clean}_{n_cls}cls"

        # Source directory depends on model type
        if model_type == "yolo_detection":
            src_dir = MODELS_DIR / "yolo_custom"
        else:
            src_dir = MODELS_DIR / f"{model_type}_model"

        dst_dir = self.MODELS_STORE / model_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(dst_dir / f.name))

        entry = {
            "id": model_id, "type": model_type, "name": name,
            "classes": classes, "accuracy": accuracy,
            "trained_at": datetime.now().isoformat(),
            "active": True, "path": str(dst_dir), "meta": meta,
        }
        for m in self._data["models"]:
            if m["type"] == model_type:
                m["active"] = False
        self._data["models"].append(entry)
        self._save()
        logger.info(f"[REGISTRY] Registered model {model_id}")
        return model_id

    def list_models(self, model_type: str | None = None) -> list[dict]:
        models = self._data["models"]
        if model_type:
            models = [m for m in models if m["type"] == model_type]
        return models

    def activate(self, model_id: str) -> bool:
        """Copy stored model into the active type directory."""
        model = None
        for m in self._data["models"]:
            if m["id"] == model_id:
                model = m
                break
        if not model:
            return False

        src_dir = Path(model["path"])
        if not src_dir.exists():
            return False

        # Destination depends on model type
        if model["type"] == "yolo_detection":
            dst_dir = MODELS_DIR / "yolo_custom"
        else:
            dst_dir = MODELS_DIR / f"{model['type']}_model"

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Clear and copy
        for f in dst_dir.iterdir():
            if f.is_file():
                f.unlink()
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(dst_dir / f.name))

        for m in self._data["models"]:
            if m["type"] == model["type"]:
                m["active"] = (m["id"] == model_id)
        self._save()
        logger.info(f"[REGISTRY] Activated model {model_id}")
        return True

    def delete(self, model_id: str) -> bool:
        """Delete a stored model (cannot delete active model)."""
        model = None
        for m in self._data["models"]:
            if m["id"] == model_id:
                model = m
                break
        if not model or model.get("active"):
            return False

        p = Path(model["path"])
        if p.exists():
            shutil.rmtree(p)
        self._data["models"] = [
            m for m in self._data["models"] if m["id"] != model_id]
        self._save()
        logger.info(f"[REGISTRY] Deleted model {model_id}")
        return True

    def get_active(self, model_type: str):
        for m in self._data["models"]:
            if m["type"] == model_type and m.get("active"):
                return m
        return None


# singleton
model_registry = ModelRegistry()


class ModelManager:
    """Thread-safe model store with automatic hot-reload from disk."""

    def __init__(self):
        self._lock = threading.Lock()

        # ── YOLOv11 (Ultralytics) primary detector ──
        self.yolo11_model = None
        self._load_yolo11()

        # ── YOLOv4-tiny fallback detector ──
        self.yolo_net = None
        self.yolo_output_layers = None
        if self.yolo11_model is None:
            self._load_yolo()

        # ── SSD fallback detector ──
        self.ssd_net = None
        if self.yolo11_model is None and self.yolo_net is None:
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

        # ── custom YOLO detection model (trained via YOLODetectionTrainer) ──
        self._custom_yolo_model = None
        self._custom_yolo_meta  = None
        self._custom_yolo_hash  = None

        # pre-load anything already on disk (graceful -- torch may be broken)
        for loader in (self._try_load_image_model, self._try_load_text_model,
                       self._try_load_voice_model, self._try_load_custom_yolo):
            try:
                loader()
            except Exception as e:
                logger.warning(f"Skipping {loader.__name__}: {e}")

        # background watcher
        self._stop_event = threading.Event()
        self._watcher = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher.start()

    # ── YOLOv11 (Ultralytics) ──────────────────────────────────────────────
    def _load_yolo11(self):
        """Load YOLOv11 from the ultralytics package.  Auto-downloads
        the nano model (~6 MB) the first time if not on disk."""
        try:
            from ultralytics import YOLO as UltralyticsYOLO
            # Use on-disk weights if present, otherwise let ultralytics fetch them
            weight_path = str(_YOLO11_PT) if _YOLO11_PT.exists() else "yolo11n.pt"
            model = UltralyticsYOLO(weight_path)
            # Warm up with a tiny dummy frame so the first real frame isn't slow
            model.predict(np.zeros((64, 64, 3), dtype=np.uint8),
                          verbose=False, imgsz=64)
            self.yolo11_model = model
            # Cache the model to our yolo-coco dir for next time
            if not _YOLO11_PT.exists():
                import shutil as _sh
                local_pt = Path(model.ckpt_path)
                if local_pt.exists():
                    _YOLO11_DIR.mkdir(parents=True, exist_ok=True)
                    _sh.copy2(str(local_pt), str(_YOLO11_PT))
            logger.info(f"YOLOv11 (nano) loaded [OK] – {len(model.names)} classes")
        except ImportError:
            logger.warning("ultralytics package not installed – YOLOv11 unavailable")
        except Exception as e:
            logger.warning(f"Failed to load YOLOv11: {e}")

    # ── YOLOv4-tiny (OpenCV DNN) ─────────────────────────────────────────
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

    # ── Custom YOLO Detection Model ──────────────────────────────────────
    def _try_load_custom_yolo(self):
        model_path = MODELS_DIR / "yolo_custom" / "best.pt"
        meta_path  = MODELS_DIR / "yolo_custom" / "meta.json"
        if not model_path.exists():
            return
        h = hashlib.md5(model_path.read_bytes()).hexdigest()
        if h == self._custom_yolo_hash:
            return
        try:
            from ultralytics import YOLO as UltralyticsYOLO
            model = UltralyticsYOLO(str(model_path))
            # Warm up
            model.predict(np.zeros((64, 64, 3), dtype=np.uint8),
                          verbose=False, imgsz=64)
            meta = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            with self._lock:
                self._custom_yolo_model = model
                self._custom_yolo_meta  = meta
                self._custom_yolo_hash  = h
            cls_count = len(meta.get("classes", model.names or {}))
            logger.info(f"[HOT-RELOAD] Custom YOLO model loaded – {cls_count} classes")
        except Exception as e:
            logger.error(f"Custom YOLO model load failed: {e}")

    # ── Background watcher ───────────────────────────────────────────────
    def _watch_loop(self):
        """Poll disk every few seconds for new / changed model files."""
        import time
        inactive_count = 0
        loaders = {
            self._try_load_image_model:  "_image_hash",
            self._try_load_text_model:   "_text_hash",
            self._try_load_voice_model:  "_voice_hash",
            self._try_load_custom_yolo:  "_custom_yolo_hash",
        }
        while not self._stop_event.is_set():
            changes_detected = False
            for loader, hash_attr in loaders.items():
                try:
                    old_hash = getattr(self, hash_attr, None)
                    loader()
                    new_hash = getattr(self, hash_attr, None)
                    if old_hash != new_hash:
                        changes_detected = True
                except Exception as e:
                    logger.debug(f"Watcher skip {loader.__name__}: {e}")

            if changes_detected:
                inactive_count = 0
                time.sleep(3)
            else:
                inactive_count += 1
                sleep_time = min(3 + inactive_count * 0.5, 10)
                time.sleep(sleep_time)

    def stop(self):
        self._stop_event.set()

    # ── Public inference helpers ─────────────────────────────────────────

    def detect_objects(self, frame, confidence_threshold=0.3, nms_threshold=0.4):
        """Run object detection.  Priority: Custom YOLO > YOLOv11 > YOLOv4-tiny > SSD.
        Returns list of {label, confidence, box, color}."""
        if self._custom_yolo_model is not None:
            return self._detect_custom_yolo(frame, confidence_threshold)
        if self.yolo11_model is not None:
            return self._detect_yolo11(frame, confidence_threshold)
        if self.yolo_net is not None:
            return self._detect_yolo(frame, confidence_threshold, nms_threshold)
        return self._detect_ssd(frame, confidence_threshold)

    # keep old name as alias for backward compat
    def detect_ssd(self, frame, confidence_threshold=0.3):
        return self.detect_objects(frame, confidence_threshold)

    def _detect_yolo11(self, frame, conf_thresh=0.3):
        """YOLOv11 (Ultralytics) detection – 80 COCO classes. Best accuracy."""
        results = self.yolo11_model.predict(
            frame, conf=conf_thresh, verbose=False, imgsz=640)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.yolo11_model.names.get(cid, f"class_{cid}")
                dets.append({
                    "label": label,
                    "confidence": conf,
                    "box": [max(0, x1), max(0, y1), x2, y2],
                    "color": COCO_COLORS[cid % len(COCO_COLORS)].tolist(),
                })
        return dets

    def _detect_custom_yolo(self, frame, conf_thresh=0.3):
        """Custom-trained YOLO detection – user-defined classes."""
        meta = self._custom_yolo_meta or {}
        imgsz = meta.get("imgsz", 640)
        results = self._custom_yolo_model.predict(
            frame, conf=conf_thresh, verbose=False, imgsz=imgsz)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                label = self._custom_yolo_model.names.get(cid, f"class_{cid}")
                dets.append({
                    "label": label,
                    "confidence": conf,
                    "box": [max(0, x1), max(0, y1), x2, y2],
                    "color": COCO_COLORS[cid % len(COCO_COLORS)].tolist(),
                })
        return dets

    def _detect_yolo(self, frame, conf_thresh=0.3, nms_thresh=0.4):
        """YOLOv4-tiny detection -- 80 COCO classes. Optimized for speed."""
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        self.yolo_net.setInput(blob)
        layer_outs = self.yolo_net.forward(self.yolo_output_layers)

        # Pre-allocate lists with estimated capacity
        boxes, confidences, class_ids = [], [], []
        scale_w, scale_h = W, H
        
        for output in layer_outs:
            for detection in output:
                # Vectorized operations for speed
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                if conf > conf_thresh:
                    # Direct calculation without intermediate array
                    cx, cy, bw, bh = detection[0] * scale_w, detection[1] * scale_h, detection[2] * scale_w, detection[3] * scale_h
                    x, y = int(cx - bw * 0.5), int(cy - bh * 0.5)
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
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 0.007843, (300, 300), 127.5)
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
        Optimized for speed.
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

            # Optimized preprocessing pipeline
            img = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Faster normalization and transpose
            img = img.astype(np.float32, copy=False) * (1.0/255.0)
            img = np.transpose(img, (2, 0, 1))
            tensor = torch.from_numpy(img).unsqueeze(0).to(device, non_blocking=True)

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
            # Standard classifier - optimized
            model   = m["model"]
            classes = m["classes"]
            img_size = m["img_size"]
            device  = m["device"]

            # Faster preprocessing
            img = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32, copy=False) * (1.0/255.0)
            img = np.transpose(img, (2, 0, 1))
            tensor = torch.from_numpy(img).unsqueeze(0).to(device, non_blocking=True)
            
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

        # Custom YOLO detection model
        custom_yolo_meta = MODELS_DIR / "yolo_custom" / "meta.json"
        if custom_yolo_meta.exists():
            info["yolo_custom"] = json.loads(custom_yolo_meta.read_text())
        else:
            info["yolo_custom"] = None

        active_det = (
            "Custom YOLO" if self._custom_yolo_model else
            "YOLOv11" if self.yolo11_model else
            "YOLOv4-tiny" if self.yolo_net else
            "MobileNet-SSD" if self.ssd_net else "None"
        )
        det_classes = (
            list(self._custom_yolo_model.names.values()) if self._custom_yolo_model else
            list(self.yolo11_model.names.values()) if self.yolo11_model else
            COCO_CLASSES if self.yolo_net else SSD_CLASSES
        )
        info["ssd"] = {
            "loaded": (self._custom_yolo_model is not None
                       or self.yolo11_model is not None
                       or self.yolo_net is not None
                       or self.ssd_net is not None),
            "detector": active_det,
            "classes": det_classes,
        }
        info["registry"] = model_registry.list_models()
        return info

    def switch_adapter(self, target: str) -> dict:
        """Thread-safe adapter switching (yolo_custom / yolo11 / yolo / ssd)."""
        with self._lock:
            if target == "yolo_custom":
                self.yolo11_model = None
                self.yolo_net = None
                self.yolo_output_layers = None
                self.ssd_net = None
                if self._custom_yolo_model is None:
                    self._try_load_custom_yolo()
                if self._custom_yolo_model:
                    meta = self._custom_yolo_meta or {}
                    classes = meta.get("classes",
                                       list(self._custom_yolo_model.names.values()))
                    return {"ok": True, "active": "Custom YOLO",
                            "classes": classes}
                return {"ok": False,
                        "error": "No custom YOLO model trained yet"}
            elif target == "yolo11":
                self._custom_yolo_model = None
                self.yolo_net = None
                self.yolo_output_layers = None
                self.ssd_net = None
                if self.yolo11_model is None:
                    self._load_yolo11()
                if self.yolo11_model:
                    return {"ok": True, "active": "YOLOv11"}
                return {"ok": False, "error": "ultralytics not installed – pip install ultralytics"}
            elif target == "yolo":
                self._custom_yolo_model = None
                self.yolo11_model = None
                self.ssd_net = None
                if self.yolo_net is None:
                    self._load_yolo()
                if self.yolo_net:
                    return {"ok": True, "active": "YOLOv4-tiny"}
                return {"ok": False, "error": "YOLO weights not found"}
            elif target == "ssd":
                self._custom_yolo_model = None
                self.yolo11_model = None
                self.yolo_net = None
                self.yolo_output_layers = None
                self._load_ssd()
                if self.ssd_net:
                    return {"ok": True, "active": "MobileNet-SSD"}
                return {"ok": False, "error": "SSD weights not found"}
        return {"ok": False, "error": "Unknown model"}

    def force_reload(self):
        """Force reload all models now."""
        self._image_hash = None
        self._text_hash  = None
        self._voice_hash = None
        self._custom_yolo_hash = None
        self._try_load_image_model()
        self._try_load_text_model()
        self._try_load_voice_model()
        self._try_load_custom_yolo()


# singleton
manager = ModelManager()
