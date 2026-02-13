"""
Auto-Training Adapter
Detects dataset type (image / voice / text), trains the appropriate model,
and hot-reloads it so the detection pipeline uses the updated weights
without restarting.

Usage (standalone):
    python akshay_raj_adapter.py --dataset path/to/dataset_folder

Usage (via web):
    Start the web server (akshay_raj_web_server.py) and upload from browser.
"""

import os
import sys
import json
import time
import shutil
import threading
import logging
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Optional heavy imports – gracefully degrade when libs are missing
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pickle
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("adapter.log"), logging.StreamHandler()],
)
logger = logging.getLogger("adapter")

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # project root
MODELS_DIR = BASE_DIR / "trained_models"
UPLOAD_DIR = BASE_DIR / "adapter" / "uploads"
STATUS_FILE = BASE_DIR / "adapter" / "status.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TEXT_EXTS  = {".txt", ".csv", ".json", ".tsv"}

for d in [MODELS_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ===================================================================
#  STATUS TRACKER  – shared across adapter + web server
# ===================================================================
class StatusTracker:
    """Thread-safe status tracker persisted to a JSON file."""

    _lock = threading.Lock()

    def __init__(self):
        self.reset()

    # ---- state helpers ----
    def reset(self):
        self._state = {
            "status": "idle",           # idle | detecting | training | done | error
            "dataset_type": None,       # image | voice | text
            "progress": 0,              # 0-100
            "epoch": 0,
            "total_epochs": 0,
            "loss": None,
            "accuracy": None,
            "message": "",
            "model_path": None,
            "started_at": None,
            "finished_at": None,
            "history": [],              # list of {epoch, loss, acc}
        }
        self._save()

    def update(self, **kwargs):
        with self._lock:
            self._state.update(kwargs)
            self._save()

    def get(self):
        with self._lock:
            return dict(self._state)

    def _save(self):
        try:
            STATUS_FILE.write_text(json.dumps(self._state, default=str))
        except Exception:
            pass


status_tracker = StatusTracker()


# ===================================================================
#  DATASET TYPE DETECTOR
# ===================================================================
def detect_dataset_type(dataset_path: str) -> str:
    """Walk the folder and vote on the dominant file type."""
    counts = {"image": 0, "audio": 0, "text": 0}
    dataset_path = Path(dataset_path)

    for f in dataset_path.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:
                counts["image"] += 1
            elif ext in AUDIO_EXTS:
                counts["audio"] += 1
            elif ext in TEXT_EXTS:
                counts["text"] += 1

    if sum(counts.values()) == 0:
        raise ValueError("No recognised data files found in the dataset folder.")

    dominant = max(counts, key=counts.get)
    logger.info(f"Dataset type detected: {dominant}  (files: {counts})")
    return dominant


# ===================================================================
#  IMAGE TRAINER  (fine-tunes a simple CNN classifier)
# ===================================================================
class ImageTrainer:
    """
    Expects folder structure:
        dataset/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                ...
    """

    def __init__(self, dataset_path, epochs=10, batch_size=16, lr=0.001, img_size=224):
        self.dataset_path = Path(dataset_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dataset ----
    class _ImgDataset(Dataset):
        def __init__(self, samples, img_size):
            self.samples = samples  # list of (path, label_idx)
            self.img_size = img_size

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = cv2.imread(str(path))
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # CHW
            return torch.tensor(img), torch.tensor(label, dtype=torch.long)

    # ---- model ----
    @staticmethod
    def _build_model(num_classes):
        import torchvision.models as models
        model = models.mobilenet_v2(pretrained=True)
        # freeze early layers
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model

    # ---- train ----
    def train(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for image training.  pip install torch torchvision")

        status_tracker.update(status="training", dataset_type="image",
                              total_epochs=self.epochs, message="Preparing image dataset…")

        # collect samples
        classes = sorted([d.name for d in self.dataset_path.iterdir() if d.is_dir()])
        if len(classes) < 2:
            raise ValueError("Need at least 2 sub-folders (one per class) inside the dataset folder.")

        class_to_idx = {c: i for i, c in enumerate(classes)}
        samples = []
        for cls in classes:
            cls_dir = self.dataset_path / cls
            for f in cls_dir.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    samples.append((f, class_to_idx[cls]))

        logger.info(f"Image dataset: {len(samples)} samples, {len(classes)} classes: {classes}")

        # split
        np.random.shuffle(samples)
        split = int(0.8 * len(samples))
        train_ds = self._ImgDataset(samples[:split], self.img_size)
        val_ds   = self._ImgDataset(samples[split:], self.img_size)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        # model
        model = self._build_model(len(classes)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)

        history = []
        best_acc = 0.0

        for epoch in range(1, self.epochs + 1):
            # --- train ---
            model.train()
            running_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)

            train_loss = running_loss / len(train_ds)

            # --- val ---
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    preds = model(imgs).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / max(total, 1)

            progress = int(epoch / self.epochs * 100)
            history.append({"epoch": epoch, "loss": round(train_loss, 4), "accuracy": round(val_acc, 4)})
            status_tracker.update(
                epoch=epoch, loss=round(train_loss, 4), accuracy=round(val_acc, 4),
                progress=progress, history=history,
                message=f"Epoch {epoch}/{self.epochs}  loss={train_loss:.4f}  acc={val_acc:.4f}",
            )
            logger.info(f"[IMAGE] Epoch {epoch}/{self.epochs}  loss={train_loss:.4f}  acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc

        # save
        save_dir = MODELS_DIR / "image_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "model.pth"
        meta_path  = save_dir / "meta.json"

        torch.save(model.state_dict(), str(model_path))
        meta = {"classes": classes, "img_size": self.img_size,
                "accuracy": best_acc, "trained_at": datetime.now().isoformat()}
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(f"[IMAGE] Model saved → {model_path}   best_acc={best_acc:.4f}")
        return str(model_path), meta


# ===================================================================
#  TEXT TRAINER  (TF-IDF + NaiveBayes classifier)
# ===================================================================
class TextTrainer:
    """
    Expects folder structure:
        dataset/
            class_a/
                doc1.txt
            class_b/
                doc2.txt
    OR a single CSV with columns: text, label
    """

    def __init__(self, dataset_path, **kwargs):
        self.dataset_path = Path(dataset_path)

    def train(self):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for text training.  pip install scikit-learn")

        status_tracker.update(status="training", dataset_type="text",
                              total_epochs=1, message="Preparing text dataset…")

        texts, labels = self._load_data()
        logger.info(f"[TEXT] {len(texts)} samples, {len(set(labels))} classes")

        X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer(max_features=10000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec   = vectorizer.transform(X_val)

        clf = MultinomialNB()
        clf.fit(X_train_vec, y_train)

        preds = clf.predict(X_val_vec)
        acc = accuracy_score(y_val, preds)

        status_tracker.update(epoch=1, progress=100, accuracy=round(acc, 4),
                              history=[{"epoch": 1, "loss": None, "accuracy": round(acc, 4)}],
                              message=f"Text model trained – accuracy {acc:.4f}")

        # save
        save_dir = MODELS_DIR / "text_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "model.pkl"
        meta_path  = save_dir / "meta.json"

        with open(model_path, "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "classifier": clf}, f)

        classes = sorted(set(labels))
        meta = {"classes": classes, "accuracy": acc, "trained_at": datetime.now().isoformat()}
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(f"[TEXT] Model saved → {model_path}   acc={acc:.4f}")
        return str(model_path), meta

    # ---- data loading helpers ----
    def _load_data(self):
        texts, labels = [], []

        # Try CSV first
        csv_files = list(self.dataset_path.glob("*.csv"))
        if csv_files:
            import csv
            for cf in csv_files:
                with open(cf, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        t = row.get("text", row.get("sentence", ""))
                        l = row.get("label", row.get("class", ""))
                        if t and l:
                            texts.append(t)
                            labels.append(l)
            if texts:
                return texts, labels

        # Fall back to sub-folder structure
        for cls_dir in sorted(self.dataset_path.iterdir()):
            if cls_dir.is_dir():
                for f in cls_dir.iterdir():
                    if f.suffix.lower() in TEXT_EXTS:
                        texts.append(f.read_text(encoding="utf-8", errors="ignore"))
                        labels.append(cls_dir.name)

        if not texts:
            raise ValueError("Could not find text data. Use sub-folders or a CSV with text,label columns.")
        return texts, labels


# ===================================================================
#  VOICE / AUDIO TRAINER  (MFCC features → simple NN classifier)
# ===================================================================
class VoiceTrainer:
    """
    Expects folder structure:
        dataset/
            class_a/
                audio1.wav
            class_b/
                audio2.wav
    """

    def __init__(self, dataset_path, epochs=20, batch_size=32, lr=0.001):
        self.dataset_path = Path(dataset_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _extract_features(file_path, max_len=100):
        """Extract MFCC features from an audio file."""
        y, sr = librosa.load(str(file_path), sr=22050, duration=5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc

    def train(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required.  pip install torch")
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa is required for voice training.  pip install librosa")

        status_tracker.update(status="training", dataset_type="voice",
                              total_epochs=self.epochs, message="Extracting audio features…")

        # collect
        classes = sorted([d.name for d in self.dataset_path.iterdir() if d.is_dir()])
        if len(classes) < 2:
            raise ValueError("Need at least 2 sub-folders (one per class).")

        class_to_idx = {c: i for i, c in enumerate(classes)}
        features, labels = [], []
        for cls in classes:
            for f in (self.dataset_path / cls).iterdir():
                if f.suffix.lower() in AUDIO_EXTS:
                    try:
                        feat = self._extract_features(f)
                        features.append(feat)
                        labels.append(class_to_idx[cls])
                    except Exception as e:
                        logger.warning(f"Skipping {f}: {e}")

        logger.info(f"[VOICE] {len(features)} samples, {len(classes)} classes: {classes}")

        X = np.array(features, dtype=np.float32)   # (N, 40, 100)
        y = np.array(labels, dtype=np.int64)

        # split
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        split = int(0.8 * len(X))
        X_train, X_val = X[idx[:split]], X[idx[split:]]
        y_train, y_val = y[idx[:split]], y[idx[split:]]

        # simple 1D-CNN
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
                x = self.conv(x)
                x = x.squeeze(-1)
                return self.fc(x)

        model = AudioCNN(len(classes)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        history = []
        best_acc = 0.0

        def _to_tensor(arr):
            return torch.tensor(arr)

        for epoch in range(1, self.epochs + 1):
            model.train()
            perm = np.random.permutation(len(X_train))
            running_loss = 0.0
            for i in range(0, len(X_train), self.batch_size):
                batch_idx = perm[i:i + self.batch_size]
                xb = _to_tensor(X_train[batch_idx]).to(self.device)
                yb = _to_tensor(y_train[batch_idx]).to(self.device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_idx)

            train_loss = running_loss / len(X_train)

            model.eval()
            with torch.no_grad():
                xv = _to_tensor(X_val).to(self.device)
                yv = _to_tensor(y_val).to(self.device)
                preds = model(xv).argmax(dim=1)
                val_acc = (preds == yv).float().mean().item()

            progress = int(epoch / self.epochs * 100)
            history.append({"epoch": epoch, "loss": round(train_loss, 4), "accuracy": round(val_acc, 4)})
            status_tracker.update(
                epoch=epoch, loss=round(train_loss, 4), accuracy=round(val_acc, 4),
                progress=progress, history=history,
                message=f"Epoch {epoch}/{self.epochs}  loss={train_loss:.4f}  acc={val_acc:.4f}",
            )
            logger.info(f"[VOICE] Epoch {epoch}/{self.epochs}  loss={train_loss:.4f}  acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc

        # save
        save_dir = MODELS_DIR / "voice_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "model.pth"
        meta_path  = save_dir / "meta.json"

        torch.save(model.state_dict(), str(model_path))
        meta = {"classes": classes, "accuracy": best_acc, "trained_at": datetime.now().isoformat()}
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(f"[VOICE] Model saved → {model_path}   best_acc={best_acc:.4f}")
        return str(model_path), meta


# ===================================================================
#  MODEL HOT-RELOADER
# ===================================================================
class ModelHotReloader:
    """
    Watches trained_models/ and re-loads the latest checkpoint into the
    running detection pipeline so there is no need to restart the app.
    """

    _lock = threading.Lock()

    def __init__(self):
        self._models = {}     # type → loaded model
        self._meta   = {}     # type → meta dict
        self._hashes = {}     # type → file hash (to detect changes)

    def reload_if_changed(self, model_type: str):
        """Check if model file changed on disk; if so, reload."""
        model_dir = MODELS_DIR / f"{model_type}_model"
        model_file = model_dir / ("model.pth" if model_type != "text" else "model.pkl")
        meta_file  = model_dir / "meta.json"

        if not model_file.exists():
            return None, None

        file_hash = hashlib.md5(model_file.read_bytes()).hexdigest()
        if file_hash == self._hashes.get(model_type):
            return self._models.get(model_type), self._meta.get(model_type)

        with self._lock:
            logger.info(f"[HOT-RELOAD] Loading new {model_type} model from {model_file}")
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

            if model_type == "text":
                import pickle
                with open(model_file, "rb") as f:
                    obj = pickle.load(f)
                self._models[model_type] = obj
            else:
                # For torch models we just store the path; consumer loads as needed
                self._models[model_type] = str(model_file)

            self._meta[model_type] = meta
            self._hashes[model_type] = file_hash
            logger.info(f"[HOT-RELOAD] {model_type} model reloaded successfully.")

        return self._models[model_type], meta

    def get_all_status(self):
        """Return overview of all available trained models."""
        out = {}
        for mt in ("image", "voice", "text"):
            model_dir = MODELS_DIR / f"{mt}_model"
            meta_file = model_dir / "meta.json"
            if meta_file.exists():
                out[mt] = json.loads(meta_file.read_text())
            else:
                out[mt] = None
        return out


hot_reloader = ModelHotReloader()


# ===================================================================
#  ADAPTER  – the main entry-point called by web server or CLI
# ===================================================================
def run_training(dataset_path: str, force_type: str = None,
                 epochs: int = None, batch_size: int = None, lr: float = None):
    """
    Detect dataset type, train the right model, save it, and
    notify the hot-reloader.  Runs **synchronously** – call from a
    background thread when invoked from the web server.
    """
    try:
        status_tracker.reset()
        status_tracker.update(status="detecting", message="Detecting dataset type…",
                              started_at=datetime.now().isoformat())

        ds_type = force_type or detect_dataset_type(dataset_path)
        status_tracker.update(dataset_type=ds_type,
                              message=f"Dataset type: {ds_type}. Starting training…")

        kwargs = {}
        if epochs:      kwargs["epochs"] = epochs
        if batch_size:  kwargs["batch_size"] = batch_size
        if lr:          kwargs["lr"] = lr

        if ds_type == "image":
            trainer = ImageTrainer(dataset_path, **kwargs)
        elif ds_type in ("audio", "voice"):
            trainer = VoiceTrainer(dataset_path, **kwargs)
        elif ds_type == "text":
            trainer = TextTrainer(dataset_path, **kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {ds_type}")

        model_path, meta = trainer.train()

        status_tracker.update(status="done", progress=100,
                              model_path=model_path,
                              finished_at=datetime.now().isoformat(),
                              message=f"Training complete! Model saved at {model_path}")

        # trigger hot-reload
        hot_reloader.reload_if_changed(ds_type if ds_type != "audio" else "voice")

        return {"ok": True, "model_path": model_path, "meta": meta}

    except Exception as exc:
        logger.exception("Training failed")
        status_tracker.update(status="error", message=str(exc))
        return {"ok": False, "error": str(exc)}


# ===================================================================
#  CLI
# ===================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Auto-Training Adapter")
    ap.add_argument("--dataset", required=True, help="Path to dataset folder")
    ap.add_argument("--type", choices=["image", "voice", "text"], default=None,
                    help="Force dataset type (auto-detected if omitted)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    result = run_training(args.dataset, force_type=args.type,
                          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    print(json.dumps(result, indent=2, default=str))
