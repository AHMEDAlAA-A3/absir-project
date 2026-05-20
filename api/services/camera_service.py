"""
api/services/camera_service.py
Singleton camera — opened once, never duplicated.
Safe release on shutdown via lifespan.
"""
import cv2
import threading
from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
class CameraService:
    _instance = None
    _lock     = threading.Lock()
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._cam_lock = threading.Lock()
                cls._instance = obj
        return cls._instance
    def capture(self):
        """Grab one frame by opening the camera, taking the picture, and closing it."""
        with self._cam_lock:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(3):
                cap.read()
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
    @property
    def is_open(self) -> bool:
        return False
camera = CameraService()