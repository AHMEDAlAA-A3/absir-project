"""
utils/storage.py
Handles saving uploaded files into organised subfolders.
Structure:
  uploads/
  ├── images/        raw uploaded images (before processing)
  ├── annotated/     processed frames with bounding boxes
  ├── text/          OCR result text files
  └── currency/      currency detection snapshots
"""
import cv2
import uuid
import json
from datetime import datetime
from pathlib import Path
from config.settings import UPLOAD_DIR
def _make_dir(sub: str) -> Path:
    d = UPLOAD_DIR / sub
    d.mkdir(parents=True, exist_ok=True)
    return d
def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
def save_upload(file_bytes: bytes, original_filename: str = "") -> Path:
    """Save raw uploaded image. Returns saved path."""
    folder = _make_dir("images")
    ext    = Path(original_filename).suffix or ".jpg"
    name   = f"{_timestamp()}_{uuid.uuid4().hex[:8]}{ext}"
    path   = folder / name
    path.write_bytes(file_bytes)
    return path
def save_annotated(frame, mode: str = "auto") -> Path:
    """Save annotated OpenCV frame. Returns saved path."""
    folder = _make_dir("annotated")
    name   = f"{_timestamp()}_{mode}_{uuid.uuid4().hex[:8]}.jpg"
    path   = folder / name
    cv2.imwrite(str(path), frame)
    return path
def save_text_result(result: dict) -> Path:
    """Save OCR result as a .json file. Returns saved path."""
    folder = _make_dir("text")
    name   = f"{_timestamp()}_{uuid.uuid4().hex[:8]}.json"
    path   = folder / name
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
def save_currency_result(result: dict, frame=None) -> Path:
    """Save currency detection result + optional frame snapshot."""
    folder = _make_dir("currency")
    uid    = uuid.uuid4().hex[:8]
    ts     = _timestamp()
    json_path = folder / f"{ts}_{uid}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if frame is not None:
        img_path = folder / f"{ts}_{uid}.jpg"
        cv2.imwrite(str(img_path), frame)
    return json_path