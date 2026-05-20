import os
import time
import cv2
import base64
import numpy as np
from fastapi import HTTPException, UploadFile
def load_frame_from_upload(file: UploadFile) -> np.ndarray:
    raw   = file.file.read()
    try:
        os.makedirs("uploads/images", exist_ok=True)
        filename = f"uploads/images/upload_{int(time.time()*1000)}.jpg"
        with open(filename, "wb") as f:
            f.write(raw)
    except Exception:
        pass
    arr   = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Cannot decode image")
    return frame
def load_frame_from_b64(b64: str) -> np.ndarray:
    try:
        data  = base64.b64decode(b64)
        arr   = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError
        return frame
    except Exception:
        raise HTTPException(400, "Cannot decode base64 frame")
def frame_to_b64(frame: np.ndarray, quality: int = 80) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()
def clean_det(d: dict) -> dict:
    out = {
        "name_en":    str(d.get("name_en", "")),
        "name_ar":    str(d.get("name_ar", "")),
        "confidence": round(float(d.get("confidence", 0)), 2),
    }
    if "bbox" in d:
        out["bbox"] = {k: int(v) for k, v in d["bbox"].items()}
    return out
def clean_danger(danger) -> tuple[str | None, dict | None]:
    """Returns (level_str, cleaned_dict)."""
    if not danger:
        return None, None
    level = str(danger.get("level", "low"))
    return level, {
        "name_en":    str(danger.get("name_en", "")),
        "name_ar":    str(danger.get("name_ar", "")),
        "level":      level,
        "confidence": round(float(danger.get("confidence", 0)), 2),
        "size_ratio": round(float(danger.get("size_ratio", 0)), 3),
        "bbox":       {k: int(v) for k, v in danger["bbox"].items()},
    }
def save_annotated_frame(annotated_frame: np.ndarray, mode: str) -> str | None:
    """Save the annotated frame (with BBox) to uploads/{mode}/ and return the path."""
    try:
        folder = f"uploads/{mode}"
        os.makedirs(folder, exist_ok=True)
        ts = int(time.time() * 1000)
        path = os.path.join(folder, f"{mode}_{ts}_annotated.jpg")
        cv2.imwrite(path, annotated_frame)
        return path
    except Exception as e:
        print(f"[{mode}] Could not save annotated image: {e}")
        return None