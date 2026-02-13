"""
==========================================================================
  app.py  –  Smart Glasses Web Application  (main entry-point)
==========================================================================
Single command to launch:
    cd webapp
    python app.py

Then open  http://localhost:5000  in your browser.

Features:
  • Live webcam feed with MobileNet-SSD object detection  (MJPEG stream)
  • Dataset upload  (zip → auto-extract → auto-detect type)
  • Auto-training  (image / voice / text)  in background thread
  • Live training progress  (SSE stream + REST poll)
  • Model hot-reload  (background watcher picks up new weights every 3 s)
  • Trained-models dashboard
  • Text classification playground

Author : Akshay Raj
Project: Smart Glasses – Object Detection
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
from pathlib import Path

from flask import (
    Flask, Response, request, jsonify, render_template, send_from_directory,
)

# ── project paths ────────────────────────────────────────────────────────
WEBAPP_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
UPLOAD_DIR   = WEBAPP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(WEBAPP_DIR / "app.log", encoding="utf-8"),
              logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False))],  # force UTF-8 on Windows console
)
logger = logging.getLogger("app")

# ── local modules ────────────────────────────────────────────────────────
from model_manager import manager as model_mgr
from video_stream  import VideoStream
from trainer       import run_training, status as train_status, detect_dataset_type

# ── Flask app ────────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder=str(WEBAPP_DIR / "templates"),
            static_folder=str(WEBAPP_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

# ── video stream singleton ───────────────────────────────────────────────
video_stream = VideoStream(model_mgr, src=0)

# ── training thread handle ───────────────────────────────────────────────
_train_thread = None


# =====================================================================
#  PAGES
# =====================================================================

@app.route("/")
def index():
    return render_template("index.html")


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
    """MJPEG stream consumed by the <img> tag in the browser."""
    if not video_stream.is_running:
        video_stream.start()
    return Response(video_stream.generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/video/detections")
def api_video_detections():
    return jsonify({
        "fps": round(video_stream.fps, 1),
        "detections": video_stream.get_detections(),
    })


# =====================================================================
#  DATASET UPLOAD API
# =====================================================================

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
    # Also allow .tar.gz compound extension
    is_tarball = fname.name.lower().endswith(('.tar.gz', '.tar.bz2', '.tar.xz'))
    if ext not in allowed and not is_tarball:
        return jsonify({"ok": False,
                        "error": f"Accepted formats: .zip, .tar.gz, .tar, .tgz, .rar, .7z"}), 400

    # Clean stem (remove .tar from compound names like archive.tar.gz)
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

    # If the zip contained a single root folder, point to it
    children = [c for c in dest.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        dest = children[0]

    try:
        ds_type = detect_dataset_type(str(dest))
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "dataset_path": str(dest),
                    "dataset_type": ds_type,
                    "message": f"Uploaded & extracted – detected type: {ds_type}"})


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
        # force model manager to pick it up immediately
        model_mgr.force_reload()

    _train_thread = threading.Thread(target=_go, daemon=True)
    _train_thread.start()
    return jsonify({"ok": True, "message": "Training started"})


@app.route("/api/train/status")
def api_train_status():
    return jsonify(train_status.get())


@app.route("/api/train/status/stream")
def api_train_stream():
    """Server-Sent Events for live training progress."""
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
    """Return training history from meta.json files with timestamps."""
    from pathlib import Path as P
    models_dir = P(__file__).resolve().parent.parent / "trained_models"
    history = []
    for mt in ("image", "voice", "text"):
        meta_f = models_dir / f"{mt}_model" / "meta.json"
        if meta_f.exists():
            try:
                meta = json.loads(meta_f.read_text())
                meta["type"] = mt
                meta["model_file"] = str(models_dir / f"{mt}_model" / (
                    "model.pth" if mt in ("image","voice") else "model.pkl"))
                history.append(meta)
            except Exception:
                pass
    # Sort by trained_at descending
    history.sort(key=lambda x: x.get("trained_at",""), reverse=True)
    return jsonify(history)


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
#  MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("   Smart Glasses – Object Detection Web App")
    print("   Open  http://localhost:5000  in your browser")
    print("=" * 62)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
