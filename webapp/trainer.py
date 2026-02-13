"""
==========================================================================
  trainer.py  –  Auto-Training Adapter (Image / Voice / Text)
==========================================================================
Detects the type of dataset that was uploaded, trains the correct model,
saves weights + metadata to  trained_models/<type>_model/,  and the
ModelManager's background watcher hot-reloads it automatically.

Author : Akshay Raj
Project: Smart Glasses – Object Detection
==========================================================================
"""

import os
import json
import time
import logging
import threading
import hashlib
from pathlib import Path
from datetime import datetime

import xml.etree.ElementTree as XMLTree

import numpy as np
import cv2

# ── optional heavy imports ───────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pickle
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

logger = logging.getLogger("trainer")

# ── paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "trained_models"
UPLOAD_DIR   = PROJECT_ROOT / "webapp" / "uploads"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TEXT_EXTS  = {".txt", ".csv", ".json", ".tsv"}


def _is_number(s):
    """Check if a string looks like a number (int or float)."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ======================================================================
#  STATUS TRACKER  (thread-safe, polled by the web UI)
# ======================================================================
class StatusTracker:
    _lock = threading.Lock()

    def __init__(self):
        self.reset()

    def reset(self):
        self._s = dict(
            status="idle", dataset_type=None, progress=0,
            epoch=0, total_epochs=0, loss=None, accuracy=None,
            message="", model_path=None, started_at=None,
            finished_at=None, history=[],
        )

    def update(self, **kw):
        with self._lock:
            self._s.update(kw)

    def get(self):
        with self._lock:
            return dict(self._s)


status = StatusTracker()


# ======================================================================
#  DATASET TYPE DETECTOR
# ======================================================================
def detect_dataset_type(path: str) -> str:
    counts = {"image": 0, "audio": 0, "text": 0}
    for f in Path(path).rglob("*"):
        if f.is_file():
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:   counts["image"] += 1
            elif ext in AUDIO_EXTS: counts["audio"] += 1
            elif ext in TEXT_EXTS:  counts["text"]  += 1
    if sum(counts.values()) == 0:
        raise ValueError("No recognised data files found.")
    return max(counts, key=counts.get)


# ======================================================================
#  IMAGE TRAINER  (MobileNetV2 fine-tune)
# ======================================================================
class ImageTrainer:
    def __init__(self, path, epochs=10, batch_size=16, lr=0.001, img_size=224):
        self.path = Path(path); self.epochs = epochs
        self.bs = batch_size; self.lr = lr; self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _DS(Dataset):
        def __init__(self, samples, sz, augment=False):
            self.samples = samples; self.sz = sz; self.augment = augment
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            p, lbl = self.samples[i]
            img = cv2.imread(str(p))
            if img is None:
                img = np.zeros((self.sz, self.sz, 3), dtype=np.uint8)
            img = cv2.resize(img, (self.sz, self.sz))
            if self.augment:
                img = self._augment(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.tensor(np.transpose(img, (2,0,1))), torch.tensor(lbl, dtype=torch.long)

        @staticmethod
        def _augment(img):
            """Apply random augmentations to prevent overfitting."""
            rng = np.random.RandomState()
            # Random horizontal flip
            if rng.rand() > 0.5:
                img = cv2.flip(img, 1)
            # Random rotation (-15 to 15 degrees)
            if rng.rand() > 0.5:
                angle = rng.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            # Color jitter: brightness/contrast
            if rng.rand() > 0.4:
                alpha = rng.uniform(0.7, 1.3)  # contrast
                beta  = rng.randint(-30, 31)    # brightness
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            # Random Gaussian blur
            if rng.rand() > 0.7:
                ksize = rng.choice([3, 5])
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            # Random grayscale
            if rng.rand() > 0.85:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Random crop & resize back
            if rng.rand() > 0.5:
                h, w = img.shape[:2]
                crop_ratio = rng.uniform(0.7, 0.95)
                new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
                y = rng.randint(0, h - new_h + 1)
                x = rng.randint(0, w - new_w + 1)
                img = cv2.resize(img[y:y+new_h, x:x+new_w], (w, h))
            return img

    def train(self):
        if not TORCH_OK:
            raise RuntimeError("PyTorch required -- pip install torch torchvision")
        import torchvision.models as models

        status.update(status="training", dataset_type="image",
                      total_epochs=self.epochs, message="Analyzing dataset structure...")

        # ── Step 1: detect what kind of image dataset we have ──
        data_root, mode = self._prepare_dataset()

        if mode == "embedding":
            return self._train_embedding(data_root)

        # ── CLASSIFIER MODE (multi-class, >= 2 real classes) ──
        # ── Step 2: build class list & samples from folders ──
        classes = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
        if len(classes) < 2:
            raise ValueError(
                f"Need >= 2 class folders after preparation. Found {len(classes)}. "
                f"Check your dataset structure.")
        c2i = {c: i for i, c in enumerate(classes)}
        samples = [(f, c2i[c]) for c in classes
                   for f in (data_root / c).rglob("*") if f.suffix.lower() in IMAGE_EXTS]

        if len(samples) == 0:
            raise ValueError(
                f"No image files found in class folders under {data_root}.\n"
                f"Supported formats: {', '.join(IMAGE_EXTS)}")

        if len(samples) < 4:
            raise ValueError(
                f"Found only {len(samples)} images. Need at least 4 for train/val split.")

        logger.info(f"Found {len(samples)} images across {len(classes)} classes: {classes}")
        status.update(message=f"Found {len(samples)} images, {len(classes)} classes. Training...")
        np.random.shuffle(samples)
        sp = max(1, int(0.8 * len(samples)))
        tds = self._DS(samples[:sp], self.img_size, augment=True)
        vds = self._DS(samples[sp:] if sp < len(samples) else samples[:1], self.img_size, augment=False)
        tl = DataLoader(tds, batch_size=self.bs, shuffle=True)
        vl = DataLoader(vds, batch_size=self.bs)

        model = models.mobilenet_v2(pretrained=True)
        # Freeze early feature layers, unfreeze last 4 blocks + classifier
        # MobileNetV2 has features[0..18]; unfreeze blocks 15-18 for fine-tuning
        for p in model.features.parameters():
            p.requires_grad = False
        for block in model.features[15:]:
            for p in block.parameters():
                p.requires_grad = True
        # Add dropout before classifier for regularization
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.last_channel, len(classes)),
        )
        model = model.to(self.device)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        opt  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=0.0003, weight_decay=1e-4)

        hist, best = [], 0.0
        for ep in range(1, self.epochs + 1):
            model.train(); rl = 0
            for imgs, lbls in tl:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                opt.zero_grad(); loss = crit(model(imgs), lbls); loss.backward(); opt.step()
                rl += loss.item() * imgs.size(0)
            tl_loss = rl / len(tds)
            model.eval(); cor = tot = 0
            with torch.no_grad():
                for imgs, lbls in vl:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    cor += (model(imgs).argmax(1) == lbls).sum().item(); tot += lbls.size(0)
            acc = cor / max(tot, 1); best = max(best, acc)
            hist.append({"epoch": ep, "loss": round(tl_loss,4), "accuracy": round(acc,4)})
            status.update(epoch=ep, loss=round(tl_loss,4), accuracy=round(acc,4),
                          progress=int(ep/self.epochs*100), history=hist,
                          message=f"Epoch {ep}/{self.epochs}  loss={tl_loss:.4f}  acc={acc:.4f}")
            logger.info(f"[IMG] E{ep}/{self.epochs} loss={tl_loss:.4f} acc={acc:.4f}")

        sd = MODELS_DIR / "image_model"; sd.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(sd / "model.pth"))
        meta = {"classes": classes, "img_size": self.img_size,
                "accuracy": best, "trained_at": datetime.now().isoformat(),
                "model_type": "classifier"}
        (sd / "meta.json").write_text(json.dumps(meta, indent=2))
        return str(sd / "model.pth"), meta

    # ------------------------------------------------------------------
    #  _train_embedding  –  single-class embedding-distance approach
    # ------------------------------------------------------------------
    def _train_embedding(self, data_root: Path):
        """
        For single-class datasets, compute the centroid of pretrained feature
        embeddings.  At inference, the model compares new images to this
        centroid via cosine similarity — no synthetic negatives needed.
        """
        import torchvision.models as models

        # Find the single class
        class_dirs = [d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith("_")]
        cls_name = class_dirs[0].name
        img_files = [f for f in class_dirs[0].rglob("*")
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

        if not img_files:
            raise ValueError(f"No images found in {class_dirs[0]}")

        logger.info(f"Embedding training: {len(img_files)} images of '{cls_name}'")
        status.update(message=f"Extracting features from {len(img_files)} '{cls_name}' images...")

        # Load pretrained feature extractor (no training needed)
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Identity()  # remove classifier, output raw features
        model = model.to(self.device)
        model.eval()

        # Extract features for all images
        embeddings = []
        for idx, img_path in enumerate(img_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = model(tensor).squeeze().cpu().numpy()  # (1280,)
            # L2 normalize
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            embeddings.append(feat)

            if (idx + 1) % 50 == 0 or idx == len(img_files) - 1:
                progress = int((idx + 1) / len(img_files) * 80)
                status.update(progress=progress,
                              message=f"Extracting features: {idx+1}/{len(img_files)}")

        if len(embeddings) < 2:
            raise ValueError("Need at least 2 valid images for embedding training")

        embeddings = np.array(embeddings)  # (N, 1280)
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # normalize

        # Compute similarity of each image to the centroid
        sims = embeddings @ centroid  # cosine similarities
        mean_sim = float(sims.mean())
        std_sim  = float(sims.std())
        # Threshold = mean - 2*std (captures 95% of the distribution)
        threshold = max(0.3, mean_sim - 2.0 * std_sim)

        logger.info(f"Embedding stats: mean_sim={mean_sim:.4f} std={std_sim:.4f} threshold={threshold:.4f}")
        status.update(progress=90, message=f"Similarity: mean={mean_sim:.3f}, threshold={threshold:.3f}")

        # Save
        sd = MODELS_DIR / "image_model"; sd.mkdir(parents=True, exist_ok=True)
        np.savez(str(sd / "embedding.npz"), centroid=centroid, threshold=threshold)
        meta = {
            "classes": ["background", cls_name],
            "img_size": self.img_size,
            "model_type": "embedding",
            "target_class": cls_name,
            "mean_similarity": round(mean_sim, 4),
            "threshold": round(threshold, 4),
            "num_samples": len(embeddings),
            "accuracy": round(mean_sim, 4),  # use mean similarity as "accuracy"
            "trained_at": datetime.now().isoformat(),
        }
        (sd / "meta.json").write_text(json.dumps(meta, indent=2))

        # Also save a dummy model.pth to trigger hot-reload via MD5 change
        dummy_state = {"centroid": centroid.tolist(), "threshold": threshold}
        torch.save(dummy_state, str(sd / "model.pth"))

        status.update(progress=100, status="done",
                      accuracy=round(mean_sim, 4),
                      message=f"Embedding model ready! {cls_name} detection with "
                              f"cosine threshold={threshold:.3f}")
        logger.info(f"[IMG-EMB] Saved embedding model: {cls_name}, "
                     f"threshold={threshold:.4f}, samples={len(embeddings)}")

        return str(sd / "model.pth"), meta

    # ------------------------------------------------------------------
    #  _prepare_dataset  –  auto-detect & restructure various formats
    # ------------------------------------------------------------------
    def _prepare_dataset(self):
        """
        Detects the dataset layout and returns a Path to a folder with
        sub-directories named after classes, each containing images.

        Supported layouts (auto-detected):
          1) Classification   –  root/class1/*.jpg  root/class2/*.jpg
          2) Train/val/test   –  split dirs with class sub-folders, CSV,
                                 YOLO txt, or COCO JSON inside each split
          3) YOLO project     –  data.yaml  +  images/ + labels/ *.txt
          4) YOLO txt labels  –  images/ + labels/*.txt  (or flat side-by-side)
          5) COCO JSON        –  images/ + annotations*.json
          6) Pascal-VOC XML   –  Images/ + Annotations/*.xml  (or flat)
          7) Flat CSV-labeled –  *.jpg + labels.csv  (filename, label cols)
        """
        p = self.path

        # ── Unwrap single wrapper directories ──
        _data_exts = IMAGE_EXTS | AUDIO_EXTS | {".xml", ".csv", ".tsv", ".json",
                                                  ".yaml", ".yml"}
        for _ in range(5):
            items = list(p.iterdir())
            dirs  = [d for d in items if d.is_dir()
                     and not d.name.startswith(("_", "."))]
            files = [f for f in items if f.is_file()
                     and f.suffix.lower() in _data_exts]
            if len(dirs) == 1 and len(files) == 0:
                logger.info(f"Unwrapping single wrapper dir: {dirs[0].name}/")
                p = dirs[0]
            else:
                break

        all_items = list(p.iterdir())
        _internal = {'_merged', '_prepared_crops'}
        top_dirs  = [d for d in all_items
                     if d.is_dir() and d.name not in _internal]
        top_files = [f for f in all_items if f.is_file()]
        dir_names = {d.name.lower() for d in top_dirs}

        ann_keywords   = {'annotations', 'labels', 'labelimg', 'pascal',
                          'yolo', 'coco'}
        split_keywords = {'train', 'test', 'val', 'valid', 'validation',
                          'training', 'testing'}

        # ── (1) YOLO project file  (data.yaml / data.yml) ──
        yolo_cfg = self._read_yolo_yaml(p)
        if yolo_cfg:
            logger.info("Dataset layout: YOLO project (data.yaml)")
            return self._yolo_project_to_classification(p, yolo_cfg)

        # ── (2) Train/test/val splits ──
        if dir_names and dir_names <= (split_keywords | _internal):
            actual_splits = [d for d in top_dirs
                             if d.name.lower() in split_keywords]
            if actual_splits:
                logger.info(f"Dataset layout: train/test/val split "
                            f"dirs={[d.name for d in actual_splits]}")
                return self._merge_splits(p, actual_splits)

        # ── (3) Classification folders  (>= 2 dirs with images) ──
        if len(top_dirs) >= 2 and not (dir_names & ann_keywords):
            has_imgs = sum(
                1 for d in top_dirs
                if any(f.suffix.lower() in IMAGE_EXTS
                       for f in d.rglob("*") if f.is_file()))
            if has_imgs >= 2:
                logger.info("Dataset layout: classification (class-folder)")
                return self._ensure_multiclass(p)

        # ── (4) COCO JSON  (images/ + *.json with COCO structure) ──
        coco_json = self._find_coco_json(p)
        if coco_json:
            img_dir = (self._find_dir(p, [
                'images', 'image', 'img', 'train', 'train_images',
                'train2017', 'val2017', 'photos']) or p)
            logger.info(f"Dataset layout: COCO JSON ({coco_json.name})")
            crop_root = self._coco_to_classification(img_dir, coco_json)
            return self._ensure_multiclass(crop_root)

        # ── (5) YOLO txt labels  (images/ + labels/ with *.txt) ──
        label_dir = self._find_dir(p, ['labels', 'label', 'yolo_labels'])
        if label_dir and any(f.suffix == '.txt' for f in label_dir.iterdir()
                             if f.is_file()):
            img_dir = (self._find_dir(p, ['images', 'image', 'img']) or p)
            class_names = self._read_class_names(p)
            logger.info(f"Dataset layout: YOLO txt labels "
                        f"({len(class_names)} classes)")
            crop_root = self._yolo_txt_to_classification(
                img_dir, label_dir, class_names)
            return self._ensure_multiclass(crop_root)

        # ── (6a) Pascal VOC: annotation dir + image dir ──
        ann_dir = self._find_dir(p, ['annotations', 'annotation',
                                      'xmls', 'xml'])
        img_dir = self._find_dir(p, ['images', 'image', 'img', 'jpegimages',
                                      'mobile_image', 'photos', 'pics'])
        # unwrap nested same-name dirs
        if ann_dir and not list(ann_dir.glob("*.xml")):
            nested = [d for d in ann_dir.iterdir() if d.is_dir()]
            if len(nested) == 1:
                ann_dir = nested[0]
        if img_dir:
            has_imgs_direct = any(f.suffix.lower() in IMAGE_EXTS
                                  for f in img_dir.iterdir() if f.is_file())
            if not has_imgs_direct:
                nested = [d for d in img_dir.iterdir() if d.is_dir()]
                if len(nested) == 1:
                    img_dir = nested[0]
        if ann_dir and img_dir and list(ann_dir.glob("*.xml")):
            logger.info(f"Dataset layout: Pascal-VOC  "
                        f"ann={ann_dir.name}  img={img_dir.name}")
            crop_root = self._voc_to_classification(img_dir, ann_dir)
            return self._ensure_multiclass(crop_root)

        # ── (6b) Flat: images + xml in same folder ──
        xml_files = list(p.glob("*.xml"))
        img_files = [f for f in top_files
                     if f.suffix.lower() in IMAGE_EXTS]
        if xml_files and img_files:
            logger.info("Dataset layout: flat images + XML annotations")
            crop_root = self._voc_to_classification(p, p)
            return self._ensure_multiclass(crop_root)

        # ── (7a) Flat images + CSV labels at root ──
        csv_files = [f for f in top_files if f.suffix.lower() == '.csv']
        if csv_files and img_files:
            csv_map = self._read_csv_labels(csv_files)
            if csv_map:
                logger.info(f"Dataset layout: flat images + CSV "
                            f"({len(csv_map)} labels)")
                out = self._csv_flat_to_classification(p, csv_map)
                return self._ensure_multiclass(out)

        # ── (7b) Flat YOLO txt alongside images ──
        txt_files = [f for f in top_files
                     if f.suffix.lower() == '.txt'
                     and f.stem.lower() not in ('classes', 'labels')]
        if txt_files and img_files:
            # Heuristic: check first txt – YOLO lines have 5 numbers
            try:
                sample = (txt_files[0].read_text(errors='ignore')
                          .strip().split('\n')[0].split())
                if len(sample) == 5 and all(
                        _is_number(x) for x in sample):
                    class_names = self._read_class_names(p)
                    logger.info(f"Dataset layout: flat YOLO txt "
                                f"({len(class_names)} classes)")
                    crop_root = self._yolo_txt_to_classification(
                        p, p, class_names)
                    return self._ensure_multiclass(crop_root)
            except Exception:
                pass

        # ── Nothing matched ──
        found = ", ".join(d.name for d in top_dirs)
        raise ValueError(
            f"Could not determine dataset layout.  "
            f"Top-level dirs: [{found}]\n\n"
            f"Supported layouts:\n"
            f"  1) Classification:  dataset/class1/*.jpg  class2/*.jpg\n"
            f"  2) Splits:  train/class1/*.jpg  test/class1/*.jpg\n"
            f"  3) CSV-labeled:  train/*.jpg + labels.csv (filename,label)\n"
            f"  4) YOLO:  images/*.jpg + labels/*.txt  (or data.yaml)\n"
            f"  5) COCO JSON:  images/*.jpg + annotations.json\n"
            f"  6) Pascal VOC:  images/*.jpg + annotations/*.xml\n"
            f"  7) Flat:  *.jpg + *.xml  or  *.jpg + *.csv\n")

    # ------------------------------------------------------------------
    #  Helpers – find dirs, read configs, parse labels
    # ------------------------------------------------------------------
    @staticmethod
    def _find_dir(root: Path, candidates: list):
        """Return the first sub-dir whose lowercased name matches a candidate."""
        for d in root.iterdir():
            if d.is_dir() and d.name.lower() in candidates:
                return d
        return None

    @staticmethod
    def _read_yolo_yaml(p: Path):
        """Read a YOLO data.yaml and return parsed config or None."""
        for name in ('data.yaml', 'data.yml', 'dataset.yaml', 'dataset.yml'):
            yp = p / name
            if not yp.exists():
                continue
            try:
                text = yp.read_text(encoding='utf-8', errors='ignore')
                cfg = {}
                names_list, collecting = [], False
                for line in text.split('\n'):
                    line = line.strip()
                    if collecting:
                        if line.startswith('-'):
                            val = line.lstrip('- ').strip().strip("'\"")
                            if val:
                                names_list.append(val)
                            continue
                        else:
                            collecting = False
                            cfg['names'] = names_list
                    if ':' not in line or line.startswith('#'):
                        continue
                    key, val = line.split(':', 1)
                    key, val = key.strip(), val.strip()
                    if key == 'names':
                        if val.startswith('['):
                            cfg['names'] = [
                                v.strip().strip("'\"")
                                for v in val.strip('[]').split(',')
                                if v.strip()]
                        else:
                            collecting = True
                            names_list = []
                    elif key in ('train', 'val', 'test', 'nc', 'path'):
                        cfg[key] = val
                if collecting:
                    cfg['names'] = names_list
                if 'names' in cfg and ('train' in cfg or 'val' in cfg):
                    logger.info(f"Parsed {name}: {len(cfg.get('names',[]))} "
                                f"classes, keys={list(cfg.keys())}")
                    return cfg
            except Exception as e:
                logger.warning(f"Could not parse {name}: {e}")
        return None

    @staticmethod
    def _read_class_names(p: Path):
        """Read class-id→name map from classes.txt / obj.names / data.yaml."""
        names = {}
        for fname in ('classes.txt', 'obj.names', 'labels.txt',
                       'names.txt', 'obj.data'):
            fp = p / fname
            if fp.exists():
                for i, line in enumerate(
                        fp.read_text(errors='ignore').strip().split('\n')):
                    line = line.strip()
                    if line:
                        names[i] = line
                if names:
                    return names
        # Fallback: try data.yaml
        cfg = ImageTrainer._read_yolo_yaml(p)
        if cfg and 'names' in cfg:
            return {i: n for i, n in enumerate(cfg['names'])}
        return names  # may be empty → class_0, class_1, …

    @staticmethod
    def _find_coco_json(p: Path):
        """Find a COCO-format JSON file in the directory."""
        # Prioritized name patterns
        for pattern in ('_annotations.coco.json', 'annotations.json',
                        'instances_*.json', '*_annotations.json',
                        'result.json', 'labels.json'):
            hits = list(p.glob(pattern))
            if hits:
                # Quick validation: must have "images" key
                try:
                    raw = hits[0].read_text(encoding='utf-8', errors='ignore')
                    if '"images"' in raw[:2000] and '"annotations"' in raw[:5000]:
                        return hits[0]
                except Exception:
                    pass
        # Brute-force: check any .json at root
        for jf in p.glob("*.json"):
            try:
                raw = jf.read_text(encoding='utf-8', errors='ignore')
                if '"images"' in raw[:2000] and '"annotations"' in raw[:5000]:
                    return jf
            except Exception:
                pass
        return None

    @staticmethod
    def _read_csv_labels(csv_files):
        """Read {filename: label} from CSV files. Handles BOM, many col names."""
        import csv as csv_mod
        csv_map = {}
        _fname_aliases = {
            'filename', 'file', 'image', 'image_name', 'img', 'fname',
            'filepath', 'file_name', 'image_id', 'id', 'image_path',
            'path', 'file_path', 'img_name', 'photo',
        }
        _label_aliases = {
            'label', 'class', 'category', 'species', 'classname',
            'class_name', 'target', 'breed', 'type', 'name',
            'classification', 'diagnosis', 'tag', 'group', 'dx',
        }
        for csv_file in csv_files:
            try:
                with open(csv_file, newline='', encoding='utf-8-sig') as fh:
                    reader = csv_mod.DictReader(fh)
                    cols = reader.fieldnames or []
                    fname_col = label_col = None
                    for c in cols:
                        cl = c.strip().lower()
                        if cl in _fname_aliases:
                            fname_col = c
                        elif cl in _label_aliases:
                            label_col = c
                    if fname_col and label_col:
                        for row in reader:
                            fn = row[fname_col].strip()
                            lb = row[label_col].strip()
                            if fn and lb:
                                csv_map[fn] = lb
                        logger.info(f"CSV labels: {csv_file.name} -> "
                                    f"{len(csv_map)} entries  "
                                    f"(cols: {fname_col}, {label_col})")
            except Exception as e:
                logger.warning(f"Could not parse CSV {csv_file.name}: {e}")
        return csv_map

    # ------------------------------------------------------------------
    #  COCO JSON → classification crops
    # ------------------------------------------------------------------
    def _coco_to_classification(self, img_dir: Path, json_path: Path):
        """Crop bounding boxes from COCO JSON into class sub-folders."""
        import shutil
        from collections import defaultdict

        out_root = self.path / "_prepared_crops"
        if out_root.exists():
            shutil.rmtree(out_root)

        with open(json_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        cat_map = {c['id']: c['name'] for c in coco.get('categories', [])}
        img_map = {}
        for im in coco.get('images', []):
            img_map[im['id']] = im.get('file_name', '')

        anns_by_img = defaultdict(list)
        for ann in coco.get('annotations', []):
            anns_by_img[ann['image_id']].append(ann)

        class_counts = {}
        idx = 0
        total = len(anns_by_img)

        for img_i, (img_id, anns) in enumerate(anns_by_img.items()):
            fname = img_map.get(img_id)
            if not fname:
                continue
            img_path = img_dir / fname
            if not img_path.exists():
                cands = list(img_dir.rglob(fname))
                img_path = cands[0] if cands else None
            if img_path is None or not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            for ann in anns:
                cat_id = ann.get('category_id')
                bbox = ann.get('bbox')            # [x, y, w, h]
                if cat_id is None or not bbox or len(bbox) < 4:
                    continue
                cls_name = cat_map.get(cat_id, f"class_{cat_id}")
                cls_name = cls_name.strip().lower().replace(" ", "_")
                x, y, bw, bh = (int(v) for v in bbox)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + bw), min(h, y + bh)
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                crop = img[y1:y2, x1:x2]
                cls_dir = out_root / cls_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cls_dir / f"{idx:06d}.jpg"), crop)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                idx += 1

            if (img_i + 1) % 200 == 0:
                status.update(
                    message=f"COCO annotations: {img_i+1}/{total} images")

        if idx == 0:
            raise ValueError("No valid crops from COCO JSON annotations.")
        summary = ", ".join(f"{k}={v}"
                            for k, v in sorted(class_counts.items()))
        logger.info(f"COCO crops: {idx} total ({summary})")
        status.update(message=f"Prepared {idx} COCO crops. Training...")
        return out_root

    # ------------------------------------------------------------------
    #  YOLO txt → classification crops
    # ------------------------------------------------------------------
    def _yolo_txt_to_classification(self, img_dir: Path,
                                     label_dir: Path,
                                     class_names: dict):
        """Crop bounding boxes from YOLO txt labels into class folders."""
        import shutil
        out_root = self.path / "_prepared_crops"
        if out_root.exists():
            shutil.rmtree(out_root)

        skip_names = {'classes.txt', 'labels.txt', 'obj.data'}
        txt_files  = sorted(f for f in label_dir.glob("*.txt")
                            if f.name.lower() not in skip_names)
        class_counts = {}
        idx = 0

        for ti, txt_file in enumerate(txt_files):
            stem = txt_file.stem
            img_path = None
            for ext in IMAGE_EXTS:
                cand = img_dir / (stem + ext)
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            for line in txt_file.read_text(errors='ignore').strip().split('\n'):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                    xc  = float(parts[1])
                    yc  = float(parts[2])
                    bw  = float(parts[3])
                    bh  = float(parts[4])
                except (ValueError, IndexError):
                    continue
                x1 = max(0, int((xc - bw / 2) * w))
                y1 = max(0, int((yc - bh / 2) * h))
                x2 = min(w, int((xc + bw / 2) * w))
                y2 = min(h, int((yc + bh / 2) * h))
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                cls_name = class_names.get(cid, f"class_{cid}")
                cls_name = cls_name.strip().lower().replace(" ", "_")
                crop = img[y1:y2, x1:x2]
                cls_dir = out_root / cls_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cls_dir / f"{idx:06d}.jpg"), crop)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                idx += 1

            if (ti + 1) % 200 == 0:
                status.update(
                    message=f"YOLO labels: {ti+1}/{len(txt_files)} files")

        if idx == 0:
            raise ValueError("No valid crops from YOLO txt annotations.")
        summary = ", ".join(f"{k}={v}"
                            for k, v in sorted(class_counts.items()))
        logger.info(f"YOLO crops: {idx} total ({summary})")
        status.update(message=f"Prepared {idx} YOLO crops. Training...")
        return out_root

    # ------------------------------------------------------------------
    #  YOLO project  (data.yaml → find splits → crop)
    # ------------------------------------------------------------------
    def _yolo_project_to_classification(self, p: Path, cfg: dict):
        """Handle a YOLO project with data.yaml pointing to splits."""
        import shutil
        names = cfg.get('names', [])
        class_names = {i: n for i, n in enumerate(names)}
        base_path = p / cfg['path'] if cfg.get('path') else p

        out_root = self.path / "_prepared_crops"
        if out_root.exists():
            shutil.rmtree(out_root)

        class_counts = {}
        idx = 0

        for key in ('train', 'val', 'test'):
            if key not in cfg:
                continue
            raw = cfg[key].strip().strip("'\"").replace('\\', '/')
            img_dir = base_path / raw
            if not img_dir.exists():
                img_dir = p / Path(raw).name
            if not img_dir.exists():
                img_dir = p / raw
            if not img_dir.exists():
                continue

            # YOLO convention: labels/ mirrors images/
            label_dir = Path(
                str(img_dir).replace('/images', '/labels')
                            .replace('\\images', '\\labels'))
            if not label_dir.exists():
                label_dir = img_dir.parent / 'labels'
            if not label_dir.exists():
                continue

            skip = {'classes.txt', 'labels.txt', 'obj.data'}
            for txt in sorted(label_dir.glob("*.txt")):
                if txt.name.lower() in skip:
                    continue
                stem = txt.stem
                img_path = None
                for ext in IMAGE_EXTS:
                    cand = img_dir / (stem + ext)
                    if cand.exists():
                        img_path = cand
                        break
                if not img_path:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                for line in txt.read_text(errors='ignore').strip().split('\n'):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = int(parts[0])
                        xc, yc = float(parts[1]), float(parts[2])
                        bw, bh = float(parts[3]), float(parts[4])
                    except (ValueError, IndexError):
                        continue
                    x1 = max(0, int((xc - bw / 2) * w))
                    y1 = max(0, int((yc - bh / 2) * h))
                    x2 = min(w, int((xc + bw / 2) * w))
                    y2 = min(h, int((yc + bh / 2) * h))
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue
                    cn = class_names.get(cid, f"class_{cid}")
                    cn = cn.strip().lower().replace(" ", "_")
                    d = out_root / cn
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{idx:06d}.jpg"),
                                img[y1:y2, x1:x2])
                    class_counts[cn] = class_counts.get(cn, 0) + 1
                    idx += 1

            status.update(message=f"YOLO split '{key}': {idx} crops so far")

        if idx == 0:
            raise ValueError(
                "No valid crops from YOLO project (data.yaml). "
                "Check that images/ and labels/ dirs exist next to each other.")
        summary = ", ".join(f"{k}={v}"
                            for k, v in sorted(class_counts.items()))
        logger.info(f"YOLO project: {idx} crops ({summary})")
        status.update(message=f"Prepared {idx} YOLO crops. Training...")
        return self._ensure_multiclass(out_root)

    # ------------------------------------------------------------------
    #  CSV flat  →  class sub-folders
    # ------------------------------------------------------------------
    def _csv_flat_to_classification(self, img_root: Path, csv_map: dict):
        """Organize flat images into class-named sub-folders via CSV map."""
        import shutil
        out_root = self.path / "_merged"
        if out_root.exists():
            shutil.rmtree(out_root)
        count = 0
        total = len(csv_map)
        for f in img_root.iterdir():
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            label = csv_map.get(f.name)
            if label is None:
                continue
            cls = label.strip().lower().replace(" ", "_")
            dest = out_root / cls
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(f), str(dest / f.name))
            count += 1
            if count % 500 == 0:
                status.update(
                    message=f"Organizing images: {count}/{total}")
        if count == 0:
            raise ValueError(
                "No images matched CSV labels. "
                "Make sure CSV 'filename' column matches actual file names.")
        classes = [d.name for d in out_root.iterdir() if d.is_dir()]
        logger.info(f"CSV flat: {count} images -> {len(classes)} classes")
        status.update(message=f"Organized {count} images -> {len(classes)} classes")
        return out_root

    # ------------------------------------------------------------------
    #  _merge_splits  –  unify train/val/test into one folder
    # ------------------------------------------------------------------
    def _merge_splits(self, base: Path, split_dirs: list):
        """
        Merge train/test/val splits into a unified class-folder layout.

        Auto-detects the format **inside** the splits:
          A) Class sub-folders   :  train/cat/*.jpg            → _merged/cat/
          B) CSV-labeled flat    :  train/*.jpg + labels.csv   → _merged/class/
          C) YOLO  (images/+labels/) inside splits             → crop to _prepared_crops/
          D) COCO JSON inside splits                           → crop to _prepared_crops/
          E) Flat images  (no labels)                          → _merged/data/
        """
        import shutil
        import csv as csv_mod
        out_root = base / "_merged"
        if out_root.exists():
            shutil.rmtree(out_root)

        merged_count = 0

        # ── Detect what's inside the splits ──
        first_sd = split_dirs[0]
        sub_items  = list(first_sd.iterdir())
        sub_dirs   = {d.name.lower(): d for d in sub_items if d.is_dir()}
        sub_files  = [f for f in sub_items if f.is_file()]

        # ── Format C: YOLO inside splits (images/ + labels/) ─────────
        if 'images' in sub_dirs and 'labels' in sub_dirs:
            logger.info("Split format: YOLO txt (images/ + labels/)")
            class_names = self._read_class_names(base)
            # also check inside each split dir
            if not class_names:
                class_names = self._read_class_names(first_sd)
            crop_root = self.path / "_prepared_crops"
            if crop_root.exists():
                shutil.rmtree(crop_root)

            class_counts = {}
            idx = 0
            skip = {'classes.txt', 'labels.txt', 'obj.data'}
            for sd in split_dirs:
                sd_img  = sd / 'images'
                sd_lbl  = sd / 'labels'
                if not sd_img.exists() or not sd_lbl.exists():
                    continue
                for txt in sorted(sd_lbl.glob("*.txt")):
                    if txt.name.lower() in skip:
                        continue
                    stem = txt.stem
                    img_path = None
                    for ext in IMAGE_EXTS:
                        cand = sd_img / (stem + ext)
                        if cand.exists():
                            img_path = cand
                            break
                    if not img_path:
                        continue
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    for line in (txt.read_text(errors='ignore')
                                 .strip().split('\n')):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            cid = int(parts[0])
                            xc, yc = float(parts[1]), float(parts[2])
                            bw, bh = float(parts[3]), float(parts[4])
                        except (ValueError, IndexError):
                            continue
                        x1 = max(0, int((xc - bw / 2) * w))
                        y1 = max(0, int((yc - bh / 2) * h))
                        x2 = min(w, int((xc + bw / 2) * w))
                        y2 = min(h, int((yc + bh / 2) * h))
                        if x2 - x1 < 10 or y2 - y1 < 10:
                            continue
                        cn = class_names.get(cid, f"class_{cid}")
                        cn = cn.strip().lower().replace(" ", "_")
                        d = crop_root / cn
                        d.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(d / f"{idx:06d}.jpg"),
                                    img[y1:y2, x1:x2])
                        class_counts[cn] = class_counts.get(cn, 0) + 1
                        idx += 1
                status.update(
                    message=f"YOLO split '{sd.name}': {idx} crops so far")

            if idx == 0:
                raise ValueError(
                    "YOLO splits detected but no crops could be extracted.")
            summary = ", ".join(f"{k}={v}"
                                for k, v in sorted(class_counts.items()))
            logger.info(f"YOLO splits: {idx} crops ({summary})")
            status.update(
                message=f"Prepared {idx} YOLO crops. Training...")
            return self._ensure_multiclass(crop_root)

        # ── Format D: COCO JSON inside a split dir ───────────────────
        for sd in split_dirs:
            coco_json = self._find_coco_json(sd)
            if coco_json:
                logger.info(f"Split format: COCO JSON in {sd.name}/")
                img_dir = (self._find_dir(sd, ['images', 'image', 'img'])
                           or sd)
                crop_root = self._coco_to_classification(img_dir, coco_json)
                return self._ensure_multiclass(crop_root)

        # ── Format A: class sub-folder splits ────────────────────────
        has_class_subdirs = any(
            sub.is_dir() and sub.name.lower() not in
            ('images', 'labels', 'image', 'label')
            for sd in split_dirs
            for sub in sd.iterdir()
            if sub.is_dir()
        )
        if has_class_subdirs:
            for split_d in split_dirs:
                for class_d in split_d.iterdir():
                    if not class_d.is_dir():
                        continue
                    dest = out_root / class_d.name.lower().replace(" ", "_")
                    dest.mkdir(parents=True, exist_ok=True)
                    for f in class_d.rglob("*"):
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                            target = dest / f"{split_d.name}_{f.name}"
                            shutil.copy2(str(f), str(target))
                            merged_count += 1
                            if merged_count % 500 == 0:
                                status.update(
                                    message=f"Merging class folders: "
                                            f"{merged_count} images...")

        # ── Format B: CSV-labeled flat images ────────────────────────
        if merged_count == 0:
            csv_map = self._read_csv_labels(sorted(base.glob("*.csv")))
            if csv_map:
                status.update(
                    message=f"Found {len(csv_map)} CSV labels. "
                            f"Reorganizing by class...")
                for split_d in split_dirs:
                    for f in split_d.iterdir():
                        if (not f.is_file()
                                or f.suffix.lower() not in IMAGE_EXTS):
                            continue
                        label = csv_map.get(f.name)
                        if label is None:
                            continue
                        cls = label.strip().lower().replace(" ", "_")
                        dest = out_root / cls
                        dest.mkdir(parents=True, exist_ok=True)
                        target = dest / f"{split_d.name}_{f.name}"
                        shutil.copy2(str(f), str(target))
                        merged_count += 1
                        if merged_count % 500 == 0:
                            status.update(
                                message=f"CSV reorganizing: "
                                        f"{merged_count} images...")

        # ── Format E: flat images, no labels → single class ─────────
        if merged_count == 0:
            for split_d in split_dirs:
                for f in split_d.rglob("*"):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                        dest = out_root / "data"
                        dest.mkdir(parents=True, exist_ok=True)
                        target = dest / f"{split_d.name}_{f.name}"
                        shutil.copy2(str(f), str(target))
                        merged_count += 1

        if merged_count == 0:
            raise ValueError(
                "No images found in train/test/val splits.\n"
                "Supported split formats:\n"
                "  1) Class sub-folders: train/cat/*.jpg, train/dog/*.jpg\n"
                "  2) CSV labels: train/*.jpg + labels.csv (filename,label)\n"
                "  3) YOLO: train/images/*.jpg + train/labels/*.txt\n"
                "  4) COCO: train/images/*.jpg + train/_annotations.coco.json\n"
                "  5) Flat images: train/*.jpg (all treated as one class)")

        classes = [d.name for d in out_root.iterdir() if d.is_dir()]
        logger.info(f"Merged {merged_count} images from splits into "
                    f"{len(classes)} classes: {classes[:20]}")
        status.update(
            message=f"Merged {merged_count} images -> {len(classes)} classes")
        return self._ensure_multiclass(out_root)

    def _ensure_multiclass(self, root: Path):
        """
        If the dataset has only one class folder, use an EMBEDDING-BASED
        approach instead of generating synthetic negatives.

        For single-class datasets, we compute the centroid of feature
        embeddings from the pretrained MobileNetV2. At inference, images
        are compared to this centroid via cosine similarity.

        Returns (root, 'embedding') if single-class, (root, 'classifier')
        if multi-class.
        """
        class_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")]
        if len(class_dirs) >= 2:
            return root, "classifier"

        cls_name = class_dirs[0].name
        logger.info(f"Single-class dataset ({cls_name}) – using embedding-distance approach")
        status.update(message=f"Single class '{cls_name}'. Computing feature embeddings...")
        return root, "embedding"

    def _voc_to_classification(self, img_dir: Path, ann_dir: Path):
        """
        Parse Pascal-VOC XMLs, crop bounding boxes, and save crops into
        class-named sub-directories.  Returns the path to the prepared
        classification root.
        """
        status.update(message="Parsing annotations & cropping objects...")
        out_root = self.path / "_prepared_crops"
        if out_root.exists():
            import shutil; shutil.rmtree(out_root)

        class_counts = {}
        idx = 0

        for xml_file in sorted(ann_dir.glob("*.xml")):
            try:
                tree = XMLTree.parse(str(xml_file))
            except XMLTree.ParseError:
                logger.warning(f"Bad XML, skipping: {xml_file.name}")
                continue
            root = tree.getroot()

            # find matching image
            fname_el = root.find("filename")
            if fname_el is not None and fname_el.text:
                img_path = img_dir / fname_el.text
            else:
                stem = xml_file.stem
                img_path = None
                for ext in IMAGE_EXTS:
                    candidate = img_dir / (stem + ext)
                    if candidate.exists():
                        img_path = candidate; break
            if img_path is None or not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            for obj in root.iter("object"):
                name_el = obj.find("name")
                if name_el is None or not name_el.text:
                    continue
                cls_name = name_el.text.strip().lower().replace(" ", "_")

                bb = obj.find("bndbox")
                if bb is None:
                    continue
                try:
                    xmin = max(0, int(float(bb.findtext("xmin", "0"))))
                    ymin = max(0, int(float(bb.findtext("ymin", "0"))))
                    xmax = min(w, int(float(bb.findtext("xmax", str(w)))))
                    ymax = min(h, int(float(bb.findtext("ymax", str(h)))))
                except (ValueError, TypeError):
                    continue
                if xmax - xmin < 10 or ymax - ymin < 10:
                    continue  # too small

                crop = img[ymin:ymax, xmin:xmax]
                cls_dir = out_root / cls_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cls_dir / f"{idx:05d}.jpg"), crop)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                idx += 1

        if idx == 0:
            raise ValueError("No valid bounding-box crops could be extracted from annotations.")

        total = sum(class_counts.values())
        summary = ", ".join(f"{k}={v}" for k, v in sorted(class_counts.items()))
        logger.info(f"Prepared {total} crops: {summary}")
        status.update(message=f"Prepared {total} crops ({summary}). Starting training...")
        return out_root


# ======================================================================
#  TEXT TRAINER  (TF-IDF + Naive Bayes)
# ======================================================================
class TextTrainer:
    def __init__(self, path, **kw):
        self.path = Path(path)

    def train(self):
        if not SKLEARN_OK:
            raise RuntimeError("scikit-learn required – pip install scikit-learn")
        status.update(status="training", dataset_type="text",
                      total_epochs=1, message="Preparing text dataset…")
        texts, labels = self._load()
        Xt, Xv, yt, yv = train_test_split(texts, labels, test_size=0.2, random_state=42)
        vec = TfidfVectorizer(max_features=10000)
        Xt_v = vec.fit_transform(Xt); Xv_v = vec.transform(Xv)
        clf = MultinomialNB(); clf.fit(Xt_v, yt)
        acc = accuracy_score(yv, clf.predict(Xv_v))
        status.update(epoch=1, progress=100, accuracy=round(acc,4),
                      history=[{"epoch":1,"loss":None,"accuracy":round(acc,4)}],
                      message=f"Text model trained – accuracy {acc:.4f}")
        sd = MODELS_DIR / "text_model"; sd.mkdir(parents=True, exist_ok=True)
        with open(sd / "model.pkl", "wb") as f:
            pickle.dump({"vectorizer": vec, "classifier": clf}, f)
        meta = {"classes": sorted(set(labels)), "accuracy": acc,
                "trained_at": datetime.now().isoformat()}
        (sd / "meta.json").write_text(json.dumps(meta, indent=2))
        return str(sd / "model.pkl"), meta

    def _load(self):
        texts, labels = [], []
        for cf in self.path.glob("*.csv"):
            import csv
            with open(cf, "r", encoding="utf-8", errors="ignore") as f:
                for row in csv.DictReader(f):
                    t = row.get("text", row.get("sentence", ""))
                    l = row.get("label", row.get("class", ""))
                    if t and l: texts.append(t); labels.append(l)
            if texts: 
                if len(texts) < 4:
                    raise ValueError(f"Found only {len(texts)} text samples in CSV. Need at least 4 for train/val split.")
                return texts, labels
        for d in sorted(self.path.iterdir()):
            if d.is_dir():
                for f in d.iterdir():
                    if f.suffix.lower() in TEXT_EXTS:
                        texts.append(f.read_text(encoding="utf-8", errors="ignore"))
                        labels.append(d.name)
        if not texts: 
            raise ValueError(f"No text data found. Expected either:\n" +
                           f"  1) CSV file with 'text' and 'label' columns: {self.path}/data.csv\n" +
                           f"  2) Class folders with text files: {self.path}/class1/text1.txt, {self.path}/class2/text2.txt\n" +
                           f"  Supported formats: {', '.join(TEXT_EXTS)}")
        if len(texts) < 4:
            raise ValueError(f"Found only {len(texts)} text samples. Need at least 4 for train/val split.")
        return texts, labels


# ======================================================================
#  VOICE TRAINER  (MFCC → 1D-CNN)
# ======================================================================
class VoiceTrainer:
    def __init__(self, path, epochs=20, batch_size=32, lr=0.001):
        self.path = Path(path); self.epochs = epochs
        self.bs = batch_size; self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _mfcc(fp, mx=100):
        y, sr = librosa.load(str(fp), sr=22050, duration=5)
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.pad(m, ((0,0),(0,max(0,mx-m.shape[1]))))[:, :mx] if m.shape[1] < mx else m[:, :mx]

    def train(self):
        if not TORCH_OK: raise RuntimeError("PyTorch required")
        if not LIBROSA_OK: raise RuntimeError("librosa required – pip install librosa")
        status.update(status="training", dataset_type="voice",
                      total_epochs=self.epochs, message="Extracting audio features…")
        classes = sorted([d.name for d in self.path.iterdir() if d.is_dir()])
        if len(classes) < 2: 
            raise ValueError(f"Expected folder structure: dataset/class1/, dataset/class2/, etc. Found {len(classes)} class folders. Need ≥ 2 class folders.")
        c2i = {c: i for i, c in enumerate(classes)}
        feats, lbls = [], []
        for c in classes:
            for f in (self.path / c).iterdir():
                if f.suffix.lower() in AUDIO_EXTS:
                    try: feats.append(self._mfcc(f)); lbls.append(c2i[c])
                    except Exception as e: logger.warning(f"Skip {f}: {e}")
        
        if len(feats) == 0:
            raise ValueError(f"No audio files found in class folders. Expected structure:\n" +
                           f"  {self.path}/class1/audio1.wav\n" +
                           f"  {self.path}/class1/audio2.mp3\n" +
                           f"  {self.path}/class2/audio3.wav\n" +
                           f"Supported formats: {', '.join(AUDIO_EXTS)}")
        
        if len(feats) < 4:
            raise ValueError(f"Found only {len(feats)} audio files. Need at least 4 for train/val split.")
        
        logger.info(f"Found {len(feats)} audio files across {len(classes)} classes: {classes}")
        X = np.array(feats, dtype=np.float32); Y = np.array(lbls, dtype=np.int64)
        idx = np.random.permutation(len(X)); sp = int(0.8*len(X))
        if sp == 0: sp = 1  # Ensure at least 1 training sample
        Xt, Xv, Yt, Yv = X[idx[:sp]], X[idx[sp:]], Y[idx[:sp]], Y[idx[sp:]]

        class CNN(nn.Module):
            def __init__(self, nc):
                super().__init__()
                self.c = nn.Sequential(
                    nn.Conv1d(40,64,3,padding=1),nn.ReLU(),nn.MaxPool1d(2),
                    nn.Conv1d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool1d(2),
                    nn.Conv1d(128,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
                self.fc = nn.Linear(128, nc)
            def forward(self, x): return self.fc(self.c(x).squeeze(-1))

        model = CNN(len(classes)).to(self.device)
        crit = nn.CrossEntropyLoss(); opt = optim.Adam(model.parameters(), lr=self.lr)
        hist, best = [], 0.0
        tt = lambda a: torch.tensor(a)
        for ep in range(1, self.epochs+1):
            model.train(); perm = np.random.permutation(len(Xt)); rl = 0
            for i in range(0, len(Xt), self.bs):
                bi = perm[i:i+self.bs]
                xb, yb = tt(Xt[bi]).to(self.device), tt(Yt[bi]).to(self.device)
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
                rl += loss.item()*len(bi)
            tl = rl/len(Xt); model.eval()
            with torch.no_grad():
                acc = (model(tt(Xv).to(self.device)).argmax(1)==tt(Yv).to(self.device)).float().mean().item()
            best = max(best, acc)
            hist.append({"epoch":ep,"loss":round(tl,4),"accuracy":round(acc,4)})
            status.update(epoch=ep,loss=round(tl,4),accuracy=round(acc,4),
                          progress=int(ep/self.epochs*100),history=hist,
                          message=f"Epoch {ep}/{self.epochs} loss={tl:.4f} acc={acc:.4f}")
        sd = MODELS_DIR/"voice_model"; sd.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(sd/"model.pth"))
        meta = {"classes":classes,"accuracy":best,"trained_at":datetime.now().isoformat()}
        (sd/"meta.json").write_text(json.dumps(meta,indent=2))
        return str(sd/"model.pth"), meta


# ======================================================================
#  MAIN ENTRY POINT
# ======================================================================
def run_training(dataset_path, force_type=None, epochs=None, batch_size=None, lr=None):
    """Synchronous – call from a background thread."""
    try:
        status.reset()
        status.update(status="detecting", message="Detecting dataset type…",
                      started_at=datetime.now().isoformat())
        ds = force_type or detect_dataset_type(dataset_path)
        status.update(dataset_type=ds, message=f"Type: {ds}. Training…")
        kw = {}
        if epochs:      kw["epochs"] = epochs
        if batch_size:  kw["batch_size"] = batch_size
        if lr:          kw["lr"] = lr

        T = {"image": ImageTrainer, "audio": VoiceTrainer,
             "voice": VoiceTrainer, "text": TextTrainer}
        trainer = T[ds](dataset_path, **kw)
        mp, meta = trainer.train()
        status.update(status="done", progress=100, model_path=mp,
                      finished_at=datetime.now().isoformat(),
                      message=f"Done! Model → {mp}")
        return {"ok": True, "model_path": mp, "meta": meta}
    except Exception as e:
        logger.exception("Training failed")
        status.update(status="error", message=str(e))
        return {"ok": False, "error": str(e)}
