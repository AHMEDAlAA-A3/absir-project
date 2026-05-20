import cv2
import numpy as np
class ColorRecognizer:
    def __init__(self):
        pass
    def detect_dominant_color(self, frame_or_path) -> dict:
        if isinstance(frame_or_path, str):
            img = cv2.imread(frame_or_path)
        else:
            img = frame_or_path
        if img is None:
            return {"color_en": "unknown", "color_ar": "غير معروف"}
        img = cv2.resize(img, (100, 100))
        avg = np.mean(img.reshape(-1, 3), axis=0)
        b, g, r = avg
        if r > g and r > b and r > 120:
            color_en, color_ar = "red",   "أحمر"
        elif g > r and g > b and g > 120:
            color_en, color_ar = "green", "أخضر"
        elif b > r and b > g and b > 120:
            color_en, color_ar = "blue",  "أزرق"
        elif r > 180 and g > 180 and b < 100:
            color_en, color_ar = "yellow", "أصفر"
        elif r > 200 and g > 200 and b > 200:
            color_en, color_ar = "white", "أبيض"
        elif r < 50 and g < 50 and b < 50:
            color_en, color_ar = "black", "أسود"
        else:
            color_en, color_ar = "unknown", "غير معروف"
        return {"color_en": color_en, "color_ar": color_ar}