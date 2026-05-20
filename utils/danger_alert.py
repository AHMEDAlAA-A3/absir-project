import time
from utils.voice import VoiceEngine
DANGEROUS_OBJECTS = {
    "car":        {"level": "high",   "ar": "سيارة",     "threshold": 0.15},
    "truck":      {"level": "high",   "ar": "شاحنة",     "threshold": 0.15},
    "bus":        {"level": "high",   "ar": "اوتوبيس",   "threshold": 0.15},
    "motorcycle": {"level": "medium", "ar": "موتوسيكل",  "threshold": 0.12},
    "bicycle":    {"level": "medium", "ar": "عجلة",      "threshold": 0.10},
    "knife":      {"level": "high",   "ar": "سكينة",     "threshold": 0.05},
    "scissors":   {"level": "medium", "ar": "مقص",       "threshold": 0.05},
    "dog":        {"level": "medium", "ar": "كلب",       "threshold": 0.10},
}
LEVEL_PRIORITY = {"high": 3, "medium": 2, "low": 1}
class DangerAlert:
    def __init__(self, cooldown: float = 4.0):
        self.cooldown    = cooldown
        self.voice       = VoiceEngine(lang="ar")
        self._last_alert: dict[str, float] = {}
    def process(self, yolo_results, model_names, frame_shape) -> dict | None:

        fh, fw = frame_shape[:2]
        frame_area = fh * fw
        now        = time.time()
        best       = None
        best_score = 0.0
        for r in yolo_results:
            for box in r.boxes:
                cls     = int(box.cls[0])
                name_en = model_names[cls].lower()
                info    = DANGEROUS_OBJECTS.get(name_en)
                if not info:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area  = (x2 - x1) * (y2 - y1)
                size_ratio = box_area / frame_area if frame_area else 0
                if size_ratio < info["threshold"]:
                    continue
                score = LEVEL_PRIORITY[info["level"]] * size_ratio
                if score > best_score:
                    best_score = score
                    best = {
                        "name_en":    name_en,
                        "name_ar":    info["ar"],
                        "level":      info["level"],
                        "confidence": float(box.conf[0]),
                        "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "size_ratio": round(size_ratio, 3),
                    }
        if best:
            key  = best["name_en"]
            last = self._last_alert.get(key, 0)
            if now - last >= self.cooldown:
                self.voice.speak(f"خلي بالك، أمامك {best['name_ar']}")
                self._last_alert[key] = now
        return best
    def add_object(self, name_en: str, name_ar: str, level: str = "medium", threshold: float = 0.10):
        DANGEROUS_OBJECTS[name_en.lower()] = {"level": level, "ar": name_ar, "threshold": threshold}
    def remove_object(self, name_en: str):
        DANGEROUS_OBJECTS.pop(name_en.lower(), None)