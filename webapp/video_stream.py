"""
==========================================================================
  video_stream.py  –  Webcam Capture + Detection + MJPEG Streaming
==========================================================================
Captures frames from the default webcam, runs MobileNet-SSD detection
(+ optional custom model overlay), encodes each annotated frame as JPEG
and yields it as an MJPEG stream that Flask serves to the browser.

Runs in its own daemon thread so the main Flask process stays responsive.

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


class VideoStream:
    """
    Thread-safe webcam → detection → MJPEG pipeline.

    Usage:
        vs = VideoStream(model_manager)
        vs.start()
        # In Flask route:
        return Response(vs.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    """

    def __init__(self, model_manager, src=0, confidence=0.2):
        self.model_manager = model_manager
        self.src = src
        self.confidence = confidence

        self._frame = None          # latest raw frame  (BGR)
        self._jpeg  = None          # latest annotated JPEG bytes
        self._lock  = threading.Lock()
        self._running = False
        self._thread  = None
        self._cap     = None

        # stats
        self.fps  = 0.0
        self._detections = []       # latest detection results

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("VideoStream started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        logger.info("VideoStream stopped")

    @property
    def is_running(self):
        return self._running

    # ── capture + detect loop ─────────────────────────────────────────────

    def _capture_loop(self):
        self._cap = cv2.VideoCapture(self.src)
        if not self._cap.isOpened():
            logger.error("Cannot open webcam")
            self._running = False
            return

        prev_time = time.time()
        frame_count = 0

        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # ── run detection ──
            annotated = frame.copy()
            det_results = []

            # 1) YOLO / SSD object detection
            ssd_results = self.model_manager.detect_objects(frame, self.confidence)
            for det in ssd_results:
                x1, y1, x2, y2 = det["box"]
                color = tuple(int(c) for c in det["color"])
                label_text = f"{det['label']}: {det['confidence']*100:.1f}%"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                y_pos = y1 - 12 if y1 - 12 > 12 else y1 + 20
                cv2.putText(annotated, label_text, (x1, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            det_results.extend(ssd_results)

            # 2) Custom image model
            custom_classes = self.model_manager.get_custom_image_classes()
            is_binary = custom_classes is not None and len(custom_classes) <= 2
            non_bg_classes = [c for c in (custom_classes or []) if c != "background"]

            if is_binary and non_bg_classes:
                # ── Binary model (e.g. background / light_bulb) ──
                # SELECTIVITY FILTER: classify ALL YOLO crops first,
                # then only overlay labels if the model is actually
                # discriminating (doesn't label everything the same).
                (h_frame, w_frame) = frame.shape[:2]
                crop_results = []  # (det, label, conf, x1c, y1c, x2c, y2c)

                for det in ssd_results:
                    x1, y1, x2, y2 = det["box"]
                    x1c = max(0, x1); y1c = max(0, y1)
                    x2c = min(w_frame, x2); y2c = min(h_frame, y2)
                    if x2c - x1c < 30 or y2c - y1c < 30:
                        continue
                    crop = frame[y1c:y2c, x1c:x2c]
                    result = self.model_manager.classify_image_custom(crop)
                    if result:
                        clabel, cconf = result
                        crop_results.append((det, clabel, cconf, x1c, y1c, x2c, y2c))

                # Selectivity check: if > 60% of crops got the SAME
                # non-background label, the model is not discriminating
                # and we suppress all custom labels for this frame.
                if crop_results:
                    positive_crops = [r for r in crop_results if r[1] != "background"]
                    total_crops = len(crop_results)
                    positive_ratio = len(positive_crops) / total_crops if total_crops > 0 else 0

                    # Only show labels when the model is selective
                    # (labels SOME crops as target but NOT most of them)
                    is_selective = positive_ratio < 0.6 or total_crops <= 2

                    if is_selective:
                        for (det, clabel, cconf, x1c, y1c, x2c, y2c) in crop_results:
                            x1, y1, x2, y2 = det["box"]
                            if clabel != "background" and cconf > 0.80:
                                color_custom = (0, 255, 255)
                                label_custom = f"[Custom] {clabel}: {cconf*100:.1f}%"
                                y_bottom = y2 + 18 if y2 + 18 < h_frame - 5 else y1 - 30
                                cv2.putText(annotated, label_custom,
                                            (x1, y_bottom), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.55, color_custom, 2)
                                det_results.append({
                                    "label": f"[Custom] {clabel}",
                                    "confidence": cconf,
                                    "box": [x1c, y1c, x2c, y2c],
                                    "color": list(color_custom),
                                })

                # Scene-level: only at very high confidence AND only
                # if the model proved selective on crops
                custom_info = self.model_manager.classify_image_custom(frame)
                if custom_info:
                    clabel, cconf = custom_info
                    # Require 95%+ AND the crop selectivity check passed
                    if clabel != "background" and cconf > 0.95:
                        # Double check: classify a blurred version too
                        blurred = cv2.GaussianBlur(frame, (51, 51), 0)
                        blur_result = self.model_manager.classify_image_custom(blurred)
                        blur_conf = blur_result[1] if blur_result and blur_result[0] == clabel else 0.0
                        # Only show if sharp image is significantly more confident than blurred
                        if cconf - blur_conf > 0.15:
                            cv2.putText(annotated,
                                        f"[Scene] {clabel}: {cconf*100:.1f}%",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 255, 255), 2)
                            already = any(clabel in d.get("label","") for d in det_results)
                            if not already:
                                det_results.append({
                                    "label": f"[Custom] {clabel}",
                                    "confidence": cconf, "box": None,
                                })

            elif custom_classes and len(custom_classes) >= 3:
                # ── Multi-class model: run on whole frame + each crop ──
                custom_info = self.model_manager.classify_image_custom(frame)
                if custom_info:
                    clabel, cconf = custom_info
                    if cconf > 0.7 and clabel != "background":
                        cv2.putText(annotated,
                                    f"[Custom] {clabel}: {cconf*100:.1f}%",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 255), 2)
                        det_results.append({
                            "label": f"[Custom] {clabel}",
                            "confidence": cconf, "box": None,
                        })

                (h_frame, w_frame) = frame.shape[:2]
                for det in ssd_results:
                    x1, y1, x2, y2 = det["box"]
                    x1c = max(0, x1); y1c = max(0, y1)
                    x2c = min(w_frame, x2); y2c = min(h_frame, y2)
                    if x2c - x1c < 20 or y2c - y1c < 20:
                        continue
                    crop = frame[y1c:y2c, x1c:x2c]
                    result = self.model_manager.classify_image_custom(crop)
                    if result:
                        clabel, cconf = result
                        if cconf > 0.7 and clabel != "background":
                            color_custom = (0, 255, 255)
                            label_custom = f"{clabel}: {cconf*100:.1f}%"
                            y_bottom = y2 + 18 if y2 + 18 < h_frame - 5 else y1 - 30
                            cv2.putText(annotated, label_custom,
                                        (x1, y_bottom), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55, color_custom, 2)
                            cv2.rectangle(annotated, (x1c, y1c), (x2c, y2c),
                                          color_custom, 2)
                            det_results.append({
                                "label": f"[Custom] {clabel}",
                                "confidence": cconf,
                                "box": [x1c, y1c, x2c, y2c],
                                "color": list(color_custom),
                            })

            # ── FPS counter ──
            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                prev_time = now

            cv2.putText(annotated, f"FPS: {self.fps:.1f}",
                        (10, annotated.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ── encode ──
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._frame = frame
                self._jpeg  = jpeg.tobytes()
                self._detections = det_results

        self._cap.release()

    # ── public getters ────────────────────────────────────────────────────

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_detections(self):
        with self._lock:
            return list(self._detections)

    def generate(self):
        """Yield MJPEG frames for Flask streaming response."""
        while self._running:
            jpeg = self.get_jpeg()
            if jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.03)   # ~30 fps cap
