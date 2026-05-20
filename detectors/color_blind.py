
import cv2
import numpy as np
from collections import Counter
_SIM = {
    "protanopia": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281,  0.099216],
        [-0.003882, -0.048116, 1.051998],
    ], dtype=np.float32),
    "deuteranopia": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501,  0.047413],
        [-0.011820, 0.042940,  0.968881],
    ], dtype=np.float32),
    "tritanopia": np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411,  0.930809,  0.147602],
        [0.004733,  0.691367,  0.303900],
    ], dtype=np.float32),
}
_CORRECT = {
    "protanopia": np.array([
        [0.0,  2.0, -1.0],
        [0.0,  1.0,  0.0],
        [0.0,  0.0,  1.0],
    ], dtype=np.float32),
    "deuteranopia": np.array([
        [1.0,  0.0,  0.0],
        [0.5,  0.0,  0.5],
        [0.0,  0.0,  1.0],
    ], dtype=np.float32),
    "tritanopia": np.array([
        [1.0,  0.0,  0.0],
        [0.0,  1.0,  0.0],
        [-0.5, 0.5,  0.0],
    ], dtype=np.float32),
}
_COLOR_NAMES = {
    "أحمر":      ([0, 120, 70],  [10, 255, 255]),
    "برتقالي":   ([11, 100, 100], [25, 255, 255]),
    "أصفر":      ([26, 100, 100], [34, 255, 255]),
    "أخضر":      ([35, 50, 50],  [85, 255, 255]),
    "أزرق فاتح": ([86, 50, 50],  [100, 255, 255]),
    "أزرق":      ([101, 50, 50], [130, 255, 255]),
    "بنفسجي":    ([131, 50, 50], [160, 255, 255]),
    "وردي":      ([161, 50, 50], [179, 255, 255]),
}
_STRUGGLE = {
    "protanopia":   "لا يرى الأحمر جيداً",
    "deuteranopia": "لا يرى الأخضر جيداً",
    "tritanopia":   "لا يرى الأزرق جيداً",
}
TYPES = list(_SIM.keys())
def _apply_matrix(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3×3 color transform matrix to a BGR frame."""
    img = frame.astype(np.float32) / 255.0
    img = img[:, :, ::-1]
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3) @ matrix.T
    flat = np.clip(flat, 0, 1)
    result = flat.reshape(h, w, 3)
    return (result[:, :, ::-1] * 255).astype(np.uint8)
def _dominant_colors(frame: np.ndarray, top: int = 5) -> list[str]:
    """Return list of dominant color names (Arabic) in a frame."""
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    found  = Counter()
    for name, (lo, hi) in _COLOR_NAMES.items():
        mask  = cv2.inRange(hsv, np.array(lo), np.array(hi))
        count = int(np.count_nonzero(mask))
        if count > 500:
            found[name] = count
    total = sum(found.values()) or 1
    return [
        {"name": n, "percent": round(c / total * 100, 1)}
        for n, c in found.most_common(top)
    ]
def process(frame: np.ndarray, cb_type: str) -> dict:
    if cb_type not in _SIM:
        cb_type = "protanopia"
    sim_matrix  = _SIM[cb_type]
    cor_matrix  = _CORRECT[cb_type]
    simulated  = _apply_matrix(frame, sim_matrix)
    corrected  = _apply_matrix(frame, cor_matrix)
    orig_colors = _dominant_colors(frame)
    corr_colors = _dominant_colors(corrected)
    def to_b64(img):
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        import base64
        return base64.b64encode(buf).decode()
    return {
        "type":             cb_type,
        "struggle":         _STRUGGLE[cb_type],
        "original_colors":  orig_colors,
        "corrected_colors": corr_colors,
        "simulated_b64":    to_b64(simulated),
        "corrected_b64":    to_b64(corrected),
    }