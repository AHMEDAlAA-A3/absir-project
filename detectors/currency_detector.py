import threading
from ultralytics import YOLO
from utils.drawing import draw_corner_box, draw_label_box
import cv2
class CurrencyDetector:
    AR_CURRENCY = {
        "100Egp": "مية جنيه",
        "200Egp": "مائتين جنيه",
        "50Egp":  "خمسون جنيه",
        "20Egp":  "عشرون جنيه",
        "10Egp":  "عشرة جنيهات",
        "5Egp":   "خمسة جنيهات",
    }
    VALUES = {
        "100Egp": 100, "200Egp": 200, "50Egp": 50,
        "20Egp":  20,  "10Egp":  10,  "5Egp":  5,
    }
    def __init__(self, model_path, conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf
        self._lock = threading.Lock()
    def detect_currency(self, image_path_or_frame):
        frame = (
            cv2.imread(image_path_or_frame)
            if isinstance(image_path_or_frame, str)
            else image_path_or_frame
        )
        with self._lock:
            results = self.model(frame, conf=self.conf, verbose=False)
        detected, total, raw = [], 0, []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name_en = self.model.names[cls]
                name_ar = self.AR_CURRENCY.get(name_en, name_en)
                val = self.VALUES.get(name_en, 0)
                detected.append(name_ar)
                total += val
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw.append({
                    "name_en": name_en,
                    "name_ar": name_ar,
                    "value": val,
                    "confidence": round(float(box.conf[0]), 2),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })
        if detected:
            unique = list(set(detected))
            if len(unique) == 1:
                msg = f"دي ورقة {unique[0]}"
            else:
                msg = "في " + " و ".join(unique)
                if total:
                    msg += f". المجموع {total} جنيه"
            return {"detected": unique, "total": total, "message": msg, "detections": raw}
        return None
    def detect_frame(self, frame):
        with self._lock:
            results = self.model(frame, conf=self.conf, verbose=False)
        detected_set = set()
        total = 0
        annotated = frame.copy()
        raw = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name_en = self.model.names[cls]
                name_ar = self.AR_CURRENCY.get(name_en, name_en)
                val = self.VALUES.get(name_en, 0)
                detected_set.add(name_ar)
                total += val
                raw.append({
                    "name_en": name_en,
                    "name_ar": name_ar,
                    "value": val,
                    "confidence": round(conf, 2),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })
                box_color = (130, 255, 100)
                annotated = draw_corner_box(annotated, x1, y1, x2, y2, color=box_color, thickness=2)
                draw_label_box(annotated, name_en, name_ar, x1, y1, x2, box_color=box_color)
        if detected_set:
            return annotated, raw
        return None, []