import cv2
import base64
import numpy as np
def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")
def base64_to_frame(b64: str) -> np.ndarray:
    data  = base64.b64decode(b64)
    arr   = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
def frame_to_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()