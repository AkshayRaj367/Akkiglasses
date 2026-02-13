# USAGE
# python akshay_raj_real_time_object_detection.py
# Press 'q' to quit, 'r' to force-reload custom model

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os
import sys
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# HOT-RELOAD SUPPORT – check for adapter-trained models every N frames
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models" / "image_model"
_RELOAD_EVERY = 150   # check disk every N frames
_frame_count  = 0
_custom_model = None   # holds (torch_model, classes, img_size) or None
_custom_hash  = None

def _try_load_custom_model():
    """Load the adapter-trained image model if available."""
    global _custom_model, _custom_hash
    model_path = TRAINED_MODELS_DIR / "model.pth"
    meta_path  = TRAINED_MODELS_DIR / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        return

    import hashlib
    h = hashlib.md5(model_path.read_bytes()).hexdigest()
    if h == _custom_hash:
        return  # unchanged

    try:
        import torch, torchvision.models as models, torch.nn as nn
        meta = json.loads(meta_path.read_text())
        classes  = meta["classes"]
        img_size = meta.get("img_size", 224)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, len(classes))
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device).eval()

        _custom_model = (model, classes, img_size, device)
        _custom_hash = h
        print(f"[HOT-RELOAD] Custom image model loaded  classes={classes}")
    except Exception as e:
        print(f"[HOT-RELOAD] Failed to load custom model: {e}")

def _classify_frame_custom(frame):
    """Run the custom classifier on a frame. Returns (label, confidence) or None."""
    if _custom_model is None:
        return None
    import torch
    model, classes, img_size, device = _custom_model
    img = cv2.resize(frame, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = probs.max(dim=1)
    return classes[idx.item()], conf.item()

# ---------------------------------------------------------------------------
# DEFAULT MobileNet-SSD model (pre-trained, always available)
# ---------------------------------------------------------------------------
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Try to load custom model at startup
_try_load_custom_model()

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	if frame is None:
		continue

	# --- periodic hot-reload check ---
	_frame_count += 1
	if _frame_count % _RELOAD_EVERY == 0:
		_try_load_custom_model()

	# Keep original frame size for full screen display
	(h, w) = frame.shape[:2]

	# ===== DEFAULT MobileNet-SSD detection =====
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detections = net.forward()

	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.2:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# ===== CUSTOM MODEL overlay (top-left corner) =====
	custom_result = _classify_frame_custom(frame)
	if custom_result:
		clabel, cconf = custom_result
		text = f"[Custom] {clabel}: {cconf*100:.1f}%"
		cv2.putText(frame, text, (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

	# show the output frame
	cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
	cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 'q' to quit, 'r' to force-reload custom model
	if key == ord("q"):
		break
	elif key == ord("r"):
		print("[INFO] Force-reloading custom model...")
		_custom_hash = None
		_try_load_custom_model()

	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()