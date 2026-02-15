"""
==========================================================================
  video_stream.py  –  Webcam Capture + Detection + MJPEG Streaming
==========================================================================
Two-thread pipeline architecture:
  Thread 1 (Capture) : Reads frames from the camera at native speed.
  Thread 2 (Process) : Detects objects, tracks them, annotates + encodes.

Key improvements over the single-thread version:
  • ObjectTracker with centroid matching and label lock-in → no flickering
  • Capture never blocks on inference → camera runs at native FPS
  • Dynamic camera resolution probing → uses camera's max resolution
  • In-place FPS mode switching → no stream recreation needed
  • Velocity-based bounding-box prediction on non-detection frames

Author : Akshay Raj
Project: Smart Glasses – Object Detection
==========================================================================
"""

import threading
import time
import logging

import cv2
import numpy as np

logger = logging.getLogger("video_stream")

# TTS announcer for smart glasses speaker output
try:
    from announcer import announcer as _announcer
except ImportError:
    _announcer = None


# =====================================================================
#  OBJECT TRACKER  –  centroid matching + label lock-in
# =====================================================================

class TrackedObject:
    """Single tracked object with label history for stability."""
    __slots__ = (
        "obj_id", "centroid", "bbox", "label", "confidence", "color",
        "label_history", "stable_label", "stable_conf",
        "disappeared", "velocity",
    )

    def __init__(self, obj_id, bbox, label, confidence, color):
        self.obj_id = obj_id
        x1, y1, x2, y2 = bbox
        self.centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.bbox = list(bbox)
        self.label = label
        self.confidence = confidence
        self.color = list(color)
        self.label_history = [(label, confidence)]
        self.stable_label = label
        self.stable_conf = confidence
        self.disappeared = 0
        self.velocity = (0.0, 0.0)


class ObjectTracker:
    """
    Lightweight centroid-based multi-object tracker.

    • Matches detections to existing tracked objects via nearest centroid.
    • Maintains a label history per object; locks in the label once it
      has been consistent for LOCK_IN_COUNT consecutive detection frames.
    • On non-detection frames, predicts bounding-box positions using
      the object's last known velocity (simple linear extrapolation).
    """

    LOCK_IN_COUNT    = 3     # consistent readings before label locks
    MAX_DISAPPEARED  = 12    # detection frames before dropping object
    DISTANCE_THRESH  = 120   # max centroid distance (px) for matching

    def __init__(self):
        self._next_id = 0
        self._objects: dict[int, TrackedObject] = {}

    # ── called on every detection frame ──────────────────────────────

    def update(self, detections: list[dict]) -> list[dict]:
        """Feed new raw detections, returns list of stabilised results."""

        # No detections → all existing objects get a disappeared tick
        if not detections:
            gone = []
            for oid, obj in self._objects.items():
                obj.disappeared += 1
                if obj.disappeared > self.MAX_DISAPPEARED:
                    gone.append(oid)
            for oid in gone:
                del self._objects[oid]
            return self._stable_results()

        # First frame with no tracked objects → register everything
        if not self._objects:
            for det in detections:
                self._register(det)
            return self._stable_results()

        # Compute centroid distances (greedy nearest-neighbour)
        obj_ids = list(self._objects.keys())
        obj_cx  = [(self._objects[oid].centroid) for oid in obj_ids]

        det_cx = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            det_cx.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        pairs = []
        for i, oc in enumerate(obj_cx):
            for j, dc in enumerate(det_cx):
                dist = ((oc[0] - dc[0]) ** 2 + (oc[1] - dc[1]) ** 2) ** 0.5
                pairs.append((dist, i, j))
        pairs.sort(key=lambda t: t[0])

        used_o, used_d = set(), set()
        for dist, oi, di in pairs:
            if oi in used_o or di in used_d:
                continue
            if dist > self.DISTANCE_THRESH:
                continue
            used_o.add(oi)
            used_d.add(di)

            oid = obj_ids[oi]
            det = detections[di]
            obj = self._objects[oid]

            # position + velocity
            old_cx, old_cy = obj.centroid
            x1, y1, x2, y2 = det["box"]
            new_cx = (x1 + x2) / 2.0
            new_cy = (y1 + y2) / 2.0
            obj.velocity = (new_cx - old_cx, new_cy - old_cy)
            obj.centroid = (new_cx, new_cy)
            obj.bbox = list(det["box"])
            obj.label = det["label"]
            obj.confidence = det["confidence"]
            obj.color = list(det.get("color", obj.color))
            obj.disappeared = 0

            # label history → lock-in
            obj.label_history.append((det["label"], det["confidence"]))
            if len(obj.label_history) > 12:
                obj.label_history = obj.label_history[-12:]

            recent = obj.label_history[-self.LOCK_IN_COUNT:]
            if len(recent) >= self.LOCK_IN_COUNT:
                labels = [r[0] for r in recent]
                if len(set(labels)) == 1:
                    obj.stable_label = labels[0]
                    obj.stable_conf  = max(r[1] for r in recent)

            # allow override when confidence drops very low
            if det["confidence"] < 0.25:
                obj.stable_label = det["label"]
                obj.stable_conf  = det["confidence"]

        # disappeared objects
        gone = []
        for i, oid in enumerate(obj_ids):
            if i not in used_o:
                self._objects[oid].disappeared += 1
                if self._objects[oid].disappeared > self.MAX_DISAPPEARED:
                    gone.append(oid)
        for oid in gone:
            del self._objects[oid]

        # new objects
        for j in range(len(detections)):
            if j not in used_d:
                self._register(detections[j])

        return self._stable_results()

    # ── called on non-detection frames ───────────────────────────────

    def predict(self) -> list[dict]:
        """
        Return predicted object positions for non-detection frames.
        Uses last velocity for linear extrapolation and damping.
        """
        results = []
        for obj in self._objects.values():
            if obj.disappeared > 3:
                continue
            dx, dy = obj.velocity
            cx, cy = obj.centroid
            # damped extrapolation
            cx += dx * 0.4
            cy += dy * 0.4
            x1, y1, x2, y2 = obj.bbox
            bw, bh = x2 - x1, y2 - y1
            new_x1 = int(cx - bw / 2)
            new_y1 = int(cy - bh / 2)
            obj.centroid = (cx, cy)
            obj.bbox = [new_x1, new_y1, new_x1 + bw, new_y1 + bh]

            results.append({
                "label": obj.stable_label,
                "confidence": obj.stable_conf,
                "box": list(obj.bbox),
                "color": list(obj.color),
            })
        return results

    # ── internals ────────────────────────────────────────────────────

    def _register(self, det):
        color = det.get("color", [0, 255, 0])
        o = TrackedObject(self._next_id, det["box"],
                          det["label"], det["confidence"], color)
        self._objects[self._next_id] = o
        self._next_id += 1

    def _stable_results(self):
        out = []
        for obj in self._objects.values():
            if obj.disappeared > 0:
                continue
            out.append({
                "label": obj.stable_label,
                "confidence": obj.stable_conf,
                "box": list(obj.bbox),
                "color": list(obj.color),
            })
        return out


# =====================================================================
#  VIDEO STREAM  –  two-thread pipeline
# =====================================================================

class VideoStream:
    """
    Thread-safe  webcam → detection → MJPEG  pipeline.

    Two daemon threads:
      1. *capture*  – reads camera frames into ``_latest_frame``
      2. *process*  – runs detection / tracking / annotation / JPEG encode

    The capture thread is **never** blocked by inference, so the camera
    always runs at its native frame-rate.  The process thread pulls the
    most-recent frame, runs YOLO + custom-model inference only on every
    Nth frame (configurable via ``fps_mode``), and uses the
    ``ObjectTracker`` for smooth bounding-box interpolation in between.
    """

    # Pre-defined FPS-mode presets
    _MODES = {
        "ultra": {
            "detection_interval": 8,
            "jpeg_quality": 65,
            "skip_custom": True,
            "stream_delay": 0.001,
        },
        "high": {
            "detection_interval": 4,
            "jpeg_quality": 75,
            "skip_custom": False,
            "stream_delay": 0.003,
        },
        "balanced": {
            "detection_interval": 2,
            "jpeg_quality": 80,
            "skip_custom": False,
            "stream_delay": 0.008,
        },
    }

    def __init__(self, model_manager, src=0, confidence=0.2,
                 fps_mode="balanced"):
        self.model_manager = model_manager
        self.src = src
        self.confidence = confidence
        self._fps_mode = fps_mode
        self._announcer = _announcer

        m = self._MODES.get(fps_mode, self._MODES["balanced"])
        self.detection_interval = m["detection_interval"]
        self.jpeg_quality       = m["jpeg_quality"]
        self.skip_custom        = m["skip_custom"]
        self.stream_delay       = m["stream_delay"]

        # shared state
        self._latest_frame = None
        self._frame_lock   = threading.Lock()
        self._jpeg         = None
        self._jpeg_lock    = threading.Lock()
        self._detections   = []
        self._det_lock     = threading.Lock()

        # camera info (populated after start)
        self.camera_width      = 0
        self.camera_height     = 0
        self.camera_fps_native = 0

        # lifecycle
        self.fps       = 0.0
        self._running  = False
        self._cap_thread  = None
        self._proc_thread = None
        self._cap         = None

        # tracker
        self._tracker = ObjectTracker()

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._tracker = ObjectTracker()
        self._cap_thread  = threading.Thread(target=self._capture_loop,
                                             daemon=True)
        self._proc_thread = threading.Thread(target=self._process_loop,
                                             daemon=True)
        self._cap_thread.start()
        self._proc_thread.start()
        # Start TTS announcer for smart glasses
        if self._announcer:
            self._announcer.start()
        logger.info("VideoStream started (2-thread pipeline)")

    def stop(self):
        self._running = False
        if self._cap_thread:
            self._cap_thread.join(timeout=5)
        if self._proc_thread:
            self._proc_thread.join(timeout=5)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        # Stop TTS announcer
        if self._announcer:
            self._announcer.stop()
        logger.info("VideoStream stopped")

    @property
    def is_running(self):
        return self._running

    # ── in-place FPS mode switch ─────────────────────────────────────

    def set_fps_mode(self, mode_name: str) -> bool:
        """Change FPS mode without recreating the stream object."""
        mode = self._MODES.get(mode_name)
        if not mode:
            return False
        self._fps_mode          = mode_name
        self.detection_interval = mode["detection_interval"]
        self.jpeg_quality       = mode["jpeg_quality"]
        self.skip_custom        = mode["skip_custom"]
        self.stream_delay       = mode["stream_delay"]
        logger.info(f"FPS mode → {mode_name}  "
                    f"(detect every {self.detection_interval} frames)")
        return True

    # ── camera resolution probing ────────────────────────────────────

    def _probe_max_resolution(self, cap):
        """Attempt highest resolution the camera supports."""
        # Request MJPG codec first — many USB cameras only give
        # high resolutions with MJPG, not raw YUV.
        cap.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"))

        targets = [
            (1920, 1080),
            (1280, 720),
            (960,  720),
            (800,  600),
            (640,  480),
        ]
        best_w, best_h = 640, 480
        for tw, th in targets:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, tw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, th)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if aw >= tw and ah >= th:
                best_w, best_h = aw, ah
                break
            if aw > best_w:
                best_w, best_h = aw, ah

        self.camera_width  = best_w
        self.camera_height = best_h

        cap.set(cv2.CAP_PROP_FPS, 60)
        self.camera_fps_native = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        logger.info(f"Camera: {best_w}×{best_h} @ "
                    f"{self.camera_fps_native:.0f} fps (requested)")

    # ── Thread 1 — capture ───────────────────────────────────────────

    def _capture_loop(self):
        """Read frames from the camera as fast as possible."""
        import sys
        # DirectShow is usually faster on Windows
        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.src)
        else:
            self._cap = cv2.VideoCapture(self.src)

        if not self._cap.isOpened():
            logger.error(
                f"Cannot open webcam (src={self.src}). "
                "Check camera permissions / availability.")
            self._running = False
            return

        self._probe_max_resolution(self._cap)

        # warm-up: discard first frames while camera auto-exposes
        logger.info("Camera warm-up …")
        for _ in range(20):
            self._cap.read()
        logger.info("Camera ready")

        reconnect_attempts = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                reconnect_attempts += 1
                if reconnect_attempts > 30:
                    logger.error("Camera disconnected — giving up after "
                                 "30 failed reads")
                    self._running = False
                    return
                time.sleep(0.05)
                continue
            reconnect_attempts = 0

            with self._frame_lock:
                self._latest_frame = frame

        try:
            self._cap.release()
        except Exception:
            pass

    # ── Thread 2 — process (detect + track + annotate + encode) ──────

    def _process_loop(self):                                      # noqa: C901
        """
        Main processing loop.
        Pulls the latest camera frame, optionally runs detection,
        updates the object tracker, draws annotations, and encodes
        the result as JPEG for the MJPEG stream.
        """
        # wait until capture provides the first frame
        while self._running:
            with self._frame_lock:
                if self._latest_frame is not None:
                    break
            time.sleep(0.01)

        frame_counter = 0
        prev_time     = time.time()
        frame_count   = 0
        last_ssd      = []   # last raw YOLO/SSD results (for custom model)

        while self._running:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.001)
                continue

            frame_counter += 1
            should_detect = (frame_counter % self.detection_interval) == 0

            # ── detection or prediction ──────────────────────────────
            if should_detect:
                ssd_results = self.model_manager.detect_objects(
                    frame, self.confidence)
                last_ssd = ssd_results

                # update tracker with fresh detections
                tracked = self._tracker.update(ssd_results)

                # custom model overlay
                if not self.skip_custom:
                    custom_dets = self._run_custom_model(frame, ssd_results)
                    det_results = tracked + custom_dets
                else:
                    det_results = tracked

                # ── TTS: announce detected objects ───────────────────
                if self._announcer:
                    self._announcer.announce(det_results)
            else:
                # non-detection frame: use tracker prediction
                det_results = self._tracker.predict()

            # ── annotate ─────────────────────────────────────────────
            annotated = frame.copy()

            for det in det_results:
                box = det.get("box")
                if box is None:
                    # scene-level custom label (no box)
                    lbl = f"{det['label']}: {det['confidence']*100:.1f}%"
                    cv2.putText(annotated, lbl, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2)
                    continue
                x1, y1, x2, y2 = box
                color = tuple(int(c) for c in det.get("color", (0, 255, 0)))
                lbl = f"{det['label']}: {det['confidence']*100:.1f}%"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                y_pos = y1 - 12 if y1 - 12 > 12 else y1 + 20
                cv2.putText(annotated, lbl, (x1, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # ── HUD overlays ─────────────────────────────────────────
            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                prev_time = now

            h, w = annotated.shape[:2]
            cv2.putText(annotated, f"FPS: {self.fps:.1f}",
                        (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated, f"{w}x{h}",
                        (w - 130, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # ── JPEG encode ──────────────────────────────────────────
            _, jpeg = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

            with self._jpeg_lock:
                self._jpeg = jpeg.tobytes()
            with self._det_lock:
                self._detections = det_results

            time.sleep(0.001)  # yield CPU slice

    # ── custom-model helpers ─────────────────────────────────────────

    def _run_custom_model(self, frame, ssd_results):
        """Run the custom image model(s). Returns extra detection dicts."""
        try:
            custom_classes = self.model_manager.get_custom_image_classes()
        except Exception:
            custom_classes = None
        if custom_classes is None:
            return []

        non_bg = [c for c in custom_classes if c != "background"]
        if len(custom_classes) <= 2 and non_bg:
            return self._binary_custom(frame, ssd_results)
        if len(custom_classes) >= 3:
            return self._multiclass_custom(frame, ssd_results,
                                           custom_classes)
        return []

    def _binary_custom(self, frame, ssd_results):
        """Binary (background / target) custom model with selectivity."""
        dets = []
        hf, wf = frame.shape[:2]
        crops = []

        for det in ssd_results:
            x1, y1, x2, y2 = det["box"]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(wf, x2), min(hf, y2)
            if x2c - x1c < 30 or y2c - y1c < 30:
                continue
            crop = frame[y1c:y2c, x1c:x2c]
            try:
                res = self.model_manager.classify_image_custom(crop)
            except Exception:
                res = None
            if res:
                crops.append((det, res[0], res[1], x1c, y1c, x2c, y2c))

        if crops:
            pos = [r for r in crops if r[1] != "background"]
            ratio = len(pos) / len(crops) if crops else 0
            selective = ratio < 0.6 or len(crops) <= 2
            if selective:
                for (det, cl, cc, x1c, y1c, x2c, y2c) in crops:
                    if cl != "background" and cc > 0.75:
                        dets.append({
                            "label": f"[Custom] {cl}",
                            "confidence": cc,
                            "box": [x1c, y1c, x2c, y2c],
                            "color": [0, 255, 255],
                        })

        # scene-level (simplified — no blur double-check)
        try:
            si = self.model_manager.classify_image_custom(frame)
        except Exception:
            si = None
        if si and si[0] != "background" and si[1] > 0.90:
            if not any(si[0] in d.get("label", "") for d in dets):
                dets.append({
                    "label": f"[Custom] {si[0]}",
                    "confidence": si[1],
                    "box": None,
                    "color": [0, 255, 255],
                })
        return dets

    def _multiclass_custom(self, frame, ssd_results, classes):
        """Multi-class custom model on full frame + YOLO crops."""
        dets = []
        n = len(classes)
        fthr = max(0.20, 0.7 - n * 0.006)
        cthr = max(0.25, 0.7 - n * 0.005)

        try:
            si = self.model_manager.classify_image_custom(frame)
        except Exception:
            si = None
        if si and si[0] != "background" and si[1] > fthr:
            dets.append({
                "label": f"[Custom] {si[0]}",
                "confidence": si[1],
                "box": None,
                "color": [0, 255, 255],
            })

        hf, wf = frame.shape[:2]
        for det in ssd_results:
            x1, y1, x2, y2 = det["box"]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(wf, x2), min(hf, y2)
            if x2c - x1c < 20 or y2c - y1c < 20:
                continue
            crop = frame[y1c:y2c, x1c:x2c]
            try:
                res = self.model_manager.classify_image_custom(crop)
            except Exception:
                res = None
            if res and res[0] != "background" and res[1] > cthr:
                dets.append({
                    "label": f"[Custom] {res[0]}",
                    "confidence": res[1],
                    "box": [x1c, y1c, x2c, y2c],
                    "color": [0, 255, 255],
                })
        return dets

    # ── public getters ───────────────────────────────────────────────

    def get_jpeg(self):
        with self._jpeg_lock:
            return self._jpeg

    def get_detections(self):
        with self._det_lock:
            return list(self._detections)

    def get_camera_info(self):
        return {
            "width": self.camera_width,
            "height": self.camera_height,
            "native_fps": self.camera_fps_native,
        }

    def generate(self):
        """Yield MJPEG frames for Flask streaming response."""
        while self._running:
            jpeg = self.get_jpeg()
            if jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + jpeg + b"\r\n")
            time.sleep(self.stream_delay)
