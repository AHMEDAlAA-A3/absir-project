from ultralytics import YOLO
from utils.drawing import (
    draw_corner_box,
    draw_label_box,
)
import cv2
import torch
import traceback
import threading
import numpy as np
class ObjectDetector:
    AR_OBJECTS = {
        "person": "شخص",
        "car": "سياره",
        "bicycle": "عجله",
        "motorcycle": "موتوسيكل",
        "bus": "اتوبيس",
        "truck": "شاحنه",
        "chair": "كرسي",
        "couch": "كنبه",
        "bed": "سرير",
        "dining table": "طاوله",
        "laptop": "لابتوب",
        "mouse": "ماوس",
        "keyboard": "كيبورد",
        "cell phone": "موبايل",
        "bottle": "زجاجه",
        "cup": "كوبايه",
        "fork": "شوكه",
        "knife": "سكينه",
        "spoon": "معلقه",
        "bowl": "طبق",
        "banana": "موزه",
        "apple": "تفاحه",
        "orange": "برتقاله",
        "book": "كتاب",
        "scissors": "مقص",
        "backpack": "شنطه",
        "handbag": "حقيبه",
        "tie": "كرافته",
        "umbrella": "شمسيه",
        "remote": "ريموت",
        "tv": "تلفزيون",
        "monitor": "شاشه",
        "cat": "قطه",
        "dog": "كلب",
    }
    def __init__(
        self,
        model_path: str,
        conf: float = 0.45,
        iou: float = 0.5,
        max_det: int = 15,
    ):
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self._lock = threading.Lock()
        self.model = YOLO(model_path)
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        self._last_results = []
        print(
            f"[ObjectDetector] Loaded on {self.device}"
        )
    def detect_frame(
        self,
        frame: np.ndarray,
    ):
        if frame is None:
            return None, []
        try:
            frame = np.ascontiguousarray(frame)
            with self._lock:
                results = self.model.predict(
                    source=frame,
                    conf=self.conf,
                    iou=self.iou,
                    max_det=self.max_det,
                    verbose=False,
                    device=self.device,
                )
            self._last_results = results
            annotated = frame.copy()
            detections = []
            seen = set()
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    try:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )
                        name_en = (
                            self.model.names[cls]
                        )
                        name_ar = (
                            self.AR_OBJECTS.get(
                                name_en,
                                name_en
                            )
                        )
                        unique_key = (
                            name_en,
                            x1 // 20,
                            y1 // 20,
                        )
                        if unique_key in seen:
                            continue
                        seen.add(unique_key)
                        detection = {
                            "name_en": name_en,
                            "name_ar": name_ar,
                            "confidence": round(
                                conf,
                                2
                            ),
                            "bbox": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                            },
                        }
                        detections.append(
                            detection
                        )
                        color = (
                            255,
                            200,
                            0,
                        )
                        annotated = draw_corner_box(
                            annotated,
                            x1,
                            y1,
                            x2,
                            y2,
                            color=color,
                            thickness=2,
                        )
                        draw_label_box(
                            annotated,
                            name_en,
                            name_ar,
                            x1,
                            y1,
                            x2,
                            box_color=color,
                        )
                    except Exception:
                        traceback.print_exc()
            detections.sort(
                key=lambda d: d["confidence"],
                reverse=True,
            )
            return annotated, detections
        except Exception:
            traceback.print_exc()
            return None, []
    def get_last_results(self):
        return (
            self._last_results,
            self.model.names,
        )