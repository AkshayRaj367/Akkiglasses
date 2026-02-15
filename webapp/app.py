"""
==========================================================================
  app.py  -  Smart Glasses Web Application
==========================================================================
Launch:   cd webapp && python app.py
Open:     http://localhost:5000

Author : Akshay Raj
Project: Smart Glasses - Object Detection
==========================================================================
"""

import os
import sys
import json
import shutil
import zipfile
import tarfile
import threading
import logging
import time
import math
import base64
import collections
from pathlib import Path

from flask import (
    Flask, Response, request, jsonify, render_template, send_from_directory, make_response,
)

# -- project paths --------------------------------------------------------
WEBAPP_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
UPLOAD_DIR   = WEBAPP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -- logging --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(WEBAPP_DIR / "app.log", encoding="utf-8"),
              logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False))],
)
logger = logging.getLogger("app")

# -- ring buffer for live CLI logs ----------------------------------------
_LOG_BUFFER_SIZE = 300
_log_buffer = collections.deque(maxlen=_LOG_BUFFER_SIZE)

class _BufferHandler(logging.Handler):
    """Captures log lines into a ring buffer for the in-browser CLI."""
    def emit(self, record):
        try:
            _log_buffer.append(self.format(record))
        except Exception:
            pass

_buf_handler = _BufferHandler()
_buf_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_buf_handler)

# -- local modules ---------------------------------------------------------
from model_manager import manager as model_mgr
from model_manager import model_registry
from video_stream  import VideoStream
from trainer       import run_training, status as train_status, detect_dataset_type
from announcer     import announcer as tts_announcer

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# -- Flask app -------------------------------------------------------------
app = Flask(__name__,
            template_folder=str(WEBAPP_DIR / "templates"),
            static_folder=str(WEBAPP_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB
app.config["TEMPLATES_AUTO_RELOAD"] = True

# -- video stream singleton ------------------------------------------------
video_stream = VideoStream(model_mgr, src=0, fps_mode="balanced")  # Default: balanced mode

# -- training thread handle ------------------------------------------------
_train_thread = None


# =====================================================================
#  PAGES
# =====================================================================

@app.route("/")
def index():
    response = make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# =====================================================================
#  VIDEO FEED API
# =====================================================================

@app.route("/api/video/start", methods=["POST"])
def api_video_start():
    video_stream.start()
    return jsonify({"ok": True, "message": "Video stream started"})


@app.route("/api/video/stop", methods=["POST"])
def api_video_stop():
    video_stream.stop()
    return jsonify({"ok": True, "message": "Video stream stopped"})


@app.route("/api/video/feed")
def api_video_feed():
    if not video_stream.is_running:
        video_stream.start()
    return Response(video_stream.generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/video/detections")
def api_video_detections():
    cam_info = video_stream.get_camera_info()
    return jsonify({
        "fps": round(video_stream.fps, 1),
        "detections": video_stream.get_detections(),
        "resolution": f"{cam_info['width']}x{cam_info['height']}",
    })


@app.route("/api/video/fps-mode", methods=["GET", "POST"])
def api_video_fps_mode():
    """Get or set video stream FPS optimization mode (in-place switch)."""
    if request.method == "GET":
        mode = video_stream._fps_mode
        return jsonify({
            "ok": True,
            "current_mode": mode,
            "fps": round(video_stream.fps, 1),
            "detection_interval": video_stream.detection_interval,
            "modes": {
                "ultra": {"fps": "Max", "description": "Max FPS, detect every 8th frame"},
                "high":  {"fps": "High", "description": "High FPS, detect every 4th frame"},
                "balanced": {"fps": "Balanced", "description": "Balanced, detect every 2nd frame"},
            },
        })

    data = request.get_json() or {}
    new_mode = data.get("mode", "").lower()

    if new_mode not in ("ultra", "high", "balanced"):
        return jsonify({"ok": False,
                        "error": "Invalid mode. Use: ultra, high, balanced"}), 400

    # In-place switch — no stream recreation needed
    video_stream.set_fps_mode(new_mode)

    return jsonify({
        "ok": True,
        "message": f"FPS mode switched to {new_mode}",
        "mode": new_mode,
        "detection_interval": video_stream.detection_interval,
        "estimated_fps": "Max" if new_mode == "ultra" else "High" if new_mode == "high" else "Balanced",
    })


# =====================================================================
#  LIVE LOGS (in-browser CLI)
# =====================================================================

@app.route("/api/logs")
def api_logs():
    """Return the last N log lines from the ring buffer."""
    return jsonify({"lines": list(_log_buffer)})


@app.route("/api/logs/stream")
def api_logs_stream():
    """SSE stream of log lines for the in-browser CLI."""
    def gen():
        sent = len(_log_buffer)
        while True:
            current = len(_log_buffer)
            if current > sent:
                new_lines = list(_log_buffer)[-(current - sent):]
                for line in new_lines:
                    yield f"data: {json.dumps(line)}\n\n"
                sent = current
            time.sleep(0.5)
    return Response(gen(), mimetype="text/event-stream")


# =====================================================================
#  DATASET UPLOAD API
# =====================================================================

def _infer_dataset_name(path: str, ds_type: str) -> str:
    """Infer a human-readable dataset name from the folder structure."""
    p = Path(path)
    # Gather class folder names
    class_names = []
    if ds_type in ("image", "audio"):
        for d in sorted(p.iterdir()):
            if d.is_dir() and d.name not in ("__MACOSX", ".DS_Store", "_prepared_crops"):
                # Check if subfolder itself has class subfolders
                subdirs = [s for s in d.iterdir() if s.is_dir() and s.name not in ("__MACOSX",)]
                if subdirs:
                    for s in sorted(subdirs):
                        if s.name.lower() not in ("background", "images", "labels", "annotations", "train", "test", "valid", "val"):
                            class_names.append(s.name.replace("_", " ").title())
                else:
                    if d.name.lower() not in ("background", "images", "labels", "annotations"):
                        class_names.append(d.name.replace("_", " ").title())
    elif ds_type == "text":
        csv_files = list(p.rglob("*.csv"))
        if csv_files:
            return csv_files[0].stem.replace("_", " ").title() + " (Text)"

    if not class_names:
        return p.name.replace("_", " ").replace("-", " ").title()

    # Limit display to first few classes
    if len(class_names) > 4:
        return f"{', '.join(class_names[:3])} + {len(class_names)-3} more"
    return ", ".join(class_names)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file in request"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename"}), 400
    fname = Path(f.filename)
    ext = fname.suffix.lower()
    allowed = {'.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.rar', '.7z'}
    is_tarball = fname.name.lower().endswith(('.tar.gz', '.tar.bz2', '.tar.xz'))
    if ext not in allowed and not is_tarball:
        return jsonify({"ok": False,
                        "error": f"Accepted formats: .zip, .tar.gz, .tar, .tgz, .rar, .7z"}), 400

    stem = fname.stem
    if stem.lower().endswith('.tar'):
        stem = stem[:-4]
    dest = UPLOAD_DIR / stem
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    archive_path = UPLOAD_DIR / fname.name
    f.save(str(archive_path))
    try:
        if ext == '.zip':
            with zipfile.ZipFile(str(archive_path), "r") as zf:
                zf.extractall(str(dest))
        elif is_tarball or ext in ('.tar', '.gz', '.tgz', '.bz2', '.xz'):
            with tarfile.open(str(archive_path), "r:*") as tf:
                tf.extractall(str(dest))
        else:
            return jsonify({"ok": False, "error": f"Cannot extract {ext} files. "
                            "Please use .zip or .tar.gz"}), 400
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        return jsonify({"ok": False, "error": f"Invalid archive: {e}"}), 400
    finally:
        archive_path.unlink(missing_ok=True)

    children = [c for c in dest.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        dest = children[0]

    try:
        ds_type = detect_dataset_type(str(dest))
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    recs = _analyze_dataset(str(dest), ds_type)
    ds_name = _infer_dataset_name(str(dest), ds_type)

    return jsonify({"ok": True, "dataset_path": str(dest),
                    "dataset_type": ds_type,
                    "dataset_name": ds_name,
                    "message": f"Uploaded - detected as {ds_type} dataset",
                    **recs})


@app.route("/api/upload/clear", methods=["POST"])
def api_upload_clear():
    """Clear the uploaded dataset so user can upload a new one."""
    d = request.get_json(force=True, silent=True) or {}
    dp = d.get("dataset_path")
    if dp and Path(dp).exists():
        try:
            shutil.rmtree(dp)
        except Exception:
            pass
    return jsonify({"ok": True, "message": "Dataset cleared"})


def _analyze_dataset(path: str, ds_type: str) -> dict:
    """Analyze an extracted dataset and return recommended hyperparams + ETA."""
    p = Path(path)
    num_files = 0
    num_classes = 0
    class_names = []

    if ds_type == "image":
        all_images = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        num_files = len(all_images)
        class_dirs = set()
        for img in all_images:
            rel = img.relative_to(p)
            parts = rel.parts
            if len(parts) >= 2:
                class_dirs.add(parts[-2])
        class_names = sorted(class_dirs - {"images", "labels", "annotations", "__MACOSX"})
        num_classes = max(len(class_names), 2)
    elif ds_type == "audio":
        all_audio = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
        num_files = len(all_audio)
        class_dirs = set()
        for af in all_audio:
            rel = af.relative_to(p)
            if len(rel.parts) >= 2:
                class_dirs.add(rel.parts[-2])
        class_names = sorted(class_dirs)
        num_classes = max(len(class_names), 2)
    elif ds_type == "text":
        csv_files = list(p.rglob("*.csv"))
        num_files = 0
        for csv_f in csv_files:
            try:
                num_files += sum(1 for _ in open(str(csv_f), encoding="utf-8", errors="ignore")) - 1
            except Exception:
                pass
        num_files = max(num_files, 1)
        num_classes = 2

    if num_files <= 100:
        rec_batch = 8
    elif num_files <= 500:
        rec_batch = 16
    elif num_files <= 2000:
        rec_batch = 32
    else:
        rec_batch = 64

    if num_files <= 50:
        rec_epochs = 20
    elif num_files <= 200:
        rec_epochs = 15
    elif num_files <= 1000:
        rec_epochs = 10
    elif num_files <= 5000:
        rec_epochs = 8
    else:
        rec_epochs = 5

    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    if ds_type == "text":
        per_sample_per_epoch = 0.001
    elif ds_type == "audio":
        per_sample_per_epoch = 0.08 if not has_gpu else 0.02
    else:
        per_sample_per_epoch = 0.05 if not has_gpu else 0.012

    eta_seconds = int(num_files * rec_epochs * per_sample_per_epoch)
    eta_seconds = max(eta_seconds, 5)

    if eta_seconds < 60:
        eta_str = f"{eta_seconds}s"
    elif eta_seconds < 3600:
        eta_str = f"{eta_seconds // 60}m {eta_seconds % 60}s"
    else:
        h = eta_seconds // 3600
        m = (eta_seconds % 3600) // 60
        eta_str = f"{h}h {m}m"

    return {
        "num_files": num_files,
        "num_classes": num_classes,
        "class_names": class_names[:20],
        "rec_batch_size": rec_batch,
        "rec_epochs": rec_epochs,
        "eta_seconds": eta_seconds,
        "eta_display": eta_str,
        "has_gpu": has_gpu,
    }


# =====================================================================
#  IMAGE DETECTION API
# =====================================================================

@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file in request"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename"}), 400

    file_bytes = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"ok": False, "error": "Could not decode image"}), 400

    det_results = model_mgr.detect_objects(frame, 0.3)

    annotated = frame.copy()
    for det in det_results:
        x1, y1, x2, y2 = det["box"]
        color = tuple(int(c) for c in det["color"])
        label_text = f"{det['label']}: {det['confidence']*100:.1f}%"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        y_pos = y1 - 12 if y1 - 12 > 12 else y1 + 20
        cv2.putText(annotated, label_text, (x1, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    custom_results = []
    custom_classes = model_mgr.get_custom_image_classes()
    if custom_classes:
        custom_info = model_mgr.classify_image_custom(frame)
        if custom_info:
            clabel, cconf = custom_info
            if clabel != "background" and cconf > 0.5:
                cv2.putText(annotated, f"[Custom] {clabel}: {cconf*100:.1f}%",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                custom_results.append({"label": f"[Custom] {clabel}", "confidence": round(cconf, 4)})

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "ok": True,
        "image": img_b64,
        "detections": det_results,
        "custom": custom_results,
        "total": len(det_results) + len(custom_results),
    })


# =====================================================================
#  TRAINING API
# =====================================================================

@app.route("/api/train", methods=["POST"])
def api_train():
    global _train_thread
    d = request.get_json(force=True, silent=True) or {}
    dp = d.get("dataset_path")
    if not dp or not Path(dp).exists():
        return jsonify({"ok": False, "error": "dataset_path missing / not found"}), 400
    cur = train_status.get()
    if cur["status"] == "training":
        return jsonify({"ok": False, "error": "Training already running"}), 409

    def _go():
        run_training(dp,
                     force_type=d.get("type") or None,
                     epochs=int(d["epochs"]) if d.get("epochs") else None,
                     batch_size=int(d["batch_size"]) if d.get("batch_size") else None,
                     lr=float(d["lr"]) if d.get("lr") else None)
        model_mgr.force_reload()

    _train_thread = threading.Thread(target=_go, daemon=True)
    _train_thread.start()
    return jsonify({"ok": True, "message": "Training started"})


@app.route("/api/train/status")
def api_train_status():
    return jsonify(train_status.get())


@app.route("/api/train/status/stream")
def api_train_stream():
    def gen():
        last = None
        while True:
            s = json.dumps(train_status.get(), default=str)
            if s != last:
                yield f"data: {s}\n\n"
                last = s
            time.sleep(0.5)
    return Response(gen(), mimetype="text/event-stream")


# =====================================================================
#  MODELS API
# =====================================================================

@app.route("/api/models")
def api_models():
    return jsonify(model_mgr.get_models_info())


@app.route("/api/models/reload", methods=["POST"])
def api_models_reload():
    model_mgr.force_reload()
    return jsonify({"ok": True, "message": "Models reloaded"})


@app.route("/api/models/history")
def api_models_history():
    """Return all models from the registry (not just active)."""
    history = model_registry.list_models()
    # Sort by trained_at descending
    history.sort(key=lambda x: x.get("trained_at", ""), reverse=True)
    return jsonify(history)


@app.route("/api/models/switch", methods=["POST"])
def api_models_switch():
    """Switch the active detection model (yolo11 / yolo / ssd) — thread-safe."""
    d = request.get_json(force=True, silent=True) or {}
    target = d.get("model", "yolo11")
    result = model_mgr.switch_adapter(target)
    if result["ok"]:
        return jsonify(result)
    return jsonify(result), 404


# =====================================================================
#  TEXT CLASSIFICATION PLAYGROUND
# =====================================================================

@app.route("/api/classify/text", methods=["POST"])
def api_classify_text():
    d = request.get_json(force=True, silent=True) or {}
    text = d.get("text", "")
    if not text:
        return jsonify({"ok": False, "error": "No text provided"}), 400
    result = model_mgr.classify_text(text)
    if result is None:
        return jsonify({"ok": False, "error": "No text model loaded"}), 404
    label, conf = result
    return jsonify({"ok": True, "label": label, "confidence": round(conf, 4)})


# =====================================================================
#  TTS ANNOUNCER API  (Smart Glasses Speaker)
# =====================================================================

@app.route("/api/tts", methods=["GET"])
def api_tts_status():
    """Get current TTS announcer settings."""
    return jsonify(tts_announcer.get_settings())


@app.route("/api/tts/toggle", methods=["POST"])
def api_tts_toggle():
    """Toggle TTS on/off, or set explicitly with {enable: true/false}."""
    d = request.get_json(force=True, silent=True) or {}
    if "enable" in d:
        tts_announcer.enabled = bool(d["enable"])
    else:
        tts_announcer.enabled = not tts_announcer.enabled
    state = "on" if tts_announcer.enabled else "off"
    return jsonify({"ok": True, "enabled": tts_announcer.enabled,
                    "message": f"Voice announcements {state}"})


@app.route("/api/tts/settings", methods=["POST"])
def api_tts_settings():
    """Update TTS settings: rate, volume, cooldown."""
    d = request.get_json(force=True, silent=True) or {}
    if "rate" in d:
        tts_announcer.rate = int(d["rate"])
    if "volume" in d:
        tts_announcer.volume = float(d["volume"])
    if "cooldown" in d:
        tts_announcer.cooldown = float(d["cooldown"])
    return jsonify({"ok": True, **tts_announcer.get_settings()})


# =====================================================================
#  MODEL REGISTRY API
# =====================================================================

@app.route("/api/models/registry")
def api_models_registry():
    """List all stored models in the registry."""
    model_type = request.args.get("type")
    return jsonify(model_registry.list_models(model_type))


@app.route("/api/models/registry/activate", methods=["POST"])
def api_models_registry_activate():
    """Activate a stored model by ID."""
    d = request.get_json(force=True, silent=True) or {}
    model_id = d.get("id")
    if not model_id:
        return jsonify({"ok": False, "error": "Missing model id"}), 400
    ok = model_registry.activate(model_id)
    if ok:
        model_mgr.force_reload()
        return jsonify({"ok": True, "message": f"Activated {model_id}"})
    return jsonify({"ok": False, "error": "Model not found or already active"}), 404


@app.route("/api/models/registry/<model_id>", methods=["DELETE"])
def api_models_registry_delete(model_id):
    """Delete a stored model by ID."""
    ok = model_registry.delete(model_id)
    if ok:
        return jsonify({"ok": True, "message": f"Deleted {model_id}"})
    return jsonify({"ok": False,
                    "error": "Not found or cannot delete active model"}), 400


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "127.0.0.1"

    print("=" * 56)
    print("  Smart Glasses - Object Detection Web App")
    print(f"  Local:   http://localhost:5000")
    print(f"  LAN:     http://{lan_ip}:5000")
    print("=" * 56)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
