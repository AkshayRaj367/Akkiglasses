"""
==========================================================================
  SmartGlass AI – Automated Setup Script
==========================================================================
Run this once after cloning the repo to install everything and download
the YOLO model weights:

    python setup.py

It will:
  1. Create a Python virtual environment  (.venv/)
  2. Install all pip dependencies
  3. Download YOLOv4-tiny weights (~23 MB)  if missing
  4. Create required directories
  5. Verify the installation

Author : Akshay Raj
Project: Smart Glasses – Object Detection with OpenCV
==========================================================================
"""

import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQ  = ROOT / "webapp" / "requirements.txt"
YOLO_DIR     = ROOT / "yolo-coco"
YOLO_WEIGHTS = YOLO_DIR / "yolov4-tiny.weights"
YOLO_URL     = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"

# ── helpers ──────────────────────────────────────────────────────────────

def heading(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")

def run(cmd, **kw):
    """Run a command and stream output."""
    print(f"  > {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=True, **kw)
    if result.returncode != 0:
        print(f"  [ERROR] Command failed with exit code {result.returncode}")
        sys.exit(1)

def get_pip():
    """Return path to pip inside the venv."""
    if sys.platform == "win32":
        return str(VENV / "Scripts" / "pip.exe")
    return str(VENV / "bin" / "pip")

def get_python():
    """Return path to python inside the venv."""
    if sys.platform == "win32":
        return str(VENV / "Scripts" / "python.exe")
    return str(VENV / "bin" / "python")

def download(url, dest):
    """Download a file with progress."""
    print(f"  Downloading: {url}")
    print(f"  Saving to:   {dest}")
    try:
        urllib.request.urlretrieve(url, str(dest), _progress)
        print()  # newline after progress
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print("  You can manually download from:")
        print(f"    {url}")
        print(f"  And place it at: {dest}")

def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb  = downloaded / (1024 * 1024)
        tot = total_size / (1024 * 1024)
        print(f"\r  [{pct:3d}%] {mb:.1f} / {tot:.1f} MB", end="", flush=True)

# ── steps ────────────────────────────────────────────────────────────────

def step_venv():
    heading("1/5  Creating virtual environment")
    if VENV.exists():
        print("  .venv/ already exists — skipping")
        return
    run(f'"{sys.executable}" -m venv "{VENV}"')
    print("  Created .venv/")

def step_deps():
    heading("2/5  Installing Python dependencies")
    pip = get_pip()
    run(f'"{pip}" install --upgrade pip')
    run(f'"{pip}" install -r "{REQ}"')

def step_yolo():
    heading("3/5  Downloading YOLOv4-tiny weights")
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    if YOLO_WEIGHTS.exists() and YOLO_WEIGHTS.stat().st_size > 1_000_000:
        print(f"  Weights already present ({YOLO_WEIGHTS.stat().st_size / 1e6:.1f} MB) — skipping")
        return
    download(YOLO_URL, YOLO_WEIGHTS)

def step_dirs():
    heading("4/5  Creating required directories")
    dirs = [
        ROOT / "webapp" / "uploads",
        ROOT / "webapp" / "static",
        ROOT / "trained_models" / "image_model",
        ROOT / "trained_models" / "text_model",
        ROOT / "trained_models" / "voice_model",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  {d.relative_to(ROOT)}/")

def step_verify():
    heading("5/5  Verifying installation")
    py = get_python()
    checks = [
        ("Flask",        "import flask; print(f'  Flask {flask.__version__}')"),
        ("OpenCV",       "import cv2; print(f'  OpenCV {cv2.__version__}')"),
        ("NumPy",        "import numpy; print(f'  NumPy {numpy.__version__}')"),
        ("PyTorch",      "import torch; print(f'  PyTorch {torch.__version__}')"),
        ("torchvision",  "import torchvision; print(f'  torchvision {torchvision.__version__}')"),
        ("scikit-learn", "import sklearn; print(f'  scikit-learn {sklearn.__version__}')"),
    ]
    all_ok = True
    for name, code in checks:
        try:
            subprocess.run(f'"{py}" -c "{code}"', shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"  [WARN] {name} not available — some features may be limited")
            all_ok = False

    # Check YOLO weights
    if YOLO_WEIGHTS.exists() and YOLO_WEIGHTS.stat().st_size > 1_000_000:
        print(f"  YOLOv4-tiny weights OK ({YOLO_WEIGHTS.stat().st_size / 1e6:.1f} MB)")
    else:
        print("  [WARN] YOLOv4-tiny weights missing — will fall back to MobileNet-SSD")
        all_ok = False

    return all_ok


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print(r"""
   ____                       _    ____ _                    _    ___
  / ___| _ __ ___   __ _ _ __| |_ / ___| | __ _ ___ ___    / \  |_ _|
  \___ \| '_ ` _ \ / _` | '__| __| |  _| |/ _` / __/ __|  / _ \  | |
   ___) | | | | | | (_| | |  | |_| |_| | | (_| \__ \__ \ / ___ \ | |
  |____/|_| |_| |_|\__,_|_|   \__|\____|_|\__,_|___/___//_/   \_\___|

  Automated Setup — Object Detection with OpenCV
  by Akshay Raj
""")

    step_venv()
    step_deps()
    step_yolo()
    step_dirs()
    ok = step_verify()

    heading("Setup Complete!")
    py = get_python()
    if sys.platform == "win32":
        activate = str(VENV / "Scripts" / "Activate.ps1")
        print(f"""
  To start the app:

    1. Activate the venv:
       & "{activate}"

    2. Run the server:
       cd webapp
       python app.py

    3. Open in browser:
       http://localhost:5000
""")
    else:
        print(f"""
  To start the app:

    1. Activate the venv:
       source .venv/bin/activate

    2. Run the server:
       cd webapp
       python app.py

    3. Open in browser:
       http://localhost:5000
""")

    if not ok:
        print("  [!] Some optional packages are missing — see warnings above.\n")


if __name__ == "__main__":
    main()
