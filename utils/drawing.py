import cv2
import numpy as np
from utils.arabic_utils import put_arabic_text, measure_arabic_text
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False
def draw_corner_box(img, x1, y1, x2, y2, color=(0, 255, 0), thickness=2, corner_length_ratio=0.2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, thickness - 1))
    w = x2 - x1
    h = y2 - y1
    l = int(min(w, h) * corner_length_ratio)
    t = thickness + 2
    cv2.line(img, (x1, y1), (x1 + l, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, t)
    cv2.line(img, (x2, y1), (x2 - l, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, t)
    cv2.line(img, (x1, y2), (x1 + l, y2), color, t)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, t)
    cv2.line(img, (x2, y2), (x2 - l, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, t)
    return img
def draw_label_box(img, label_en, label_ar, x1, y1, x2, box_color, font_size=20):
    """
    Draws a clean label centered above a bounding box.
    """
    font      = cv2.FONT_HERSHEY_SIMPLEX
    padding   = 8
    (en_w, en_h), _ = cv2.getTextSize(label_en, font, 0.6, 2)
    ar_w, ar_h = measure_arabic_text(label_ar, font_size)
    total_h = ar_h + en_h + padding * 3
    box_w   = max(en_w, ar_w) + padding * 4
    cx = (x1 + x2) // 2
    bg_x1 = cx - box_w // 2
    bg_x2 = cx + box_w // 2
    h_img, w_img = img.shape[:2]
    if bg_x1 < 0:
        bg_x1 = 0
        bg_x2 = box_w
    if bg_x2 > w_img:
        bg_x2 = w_img
        bg_x1 = w_img - box_w
    bg_y2 = y1
    bg_y1 = y1 - total_h
    if bg_y1 < 0:
        bg_y1 = y1
        bg_y2 = y1 + total_h
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (25, 25, 25), -1)
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, 2)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    en_x = bg_x1 + (box_w - en_w) // 2
    en_y = bg_y2 - padding
    cv2.putText(img, label_en, (en_x, en_y),
                font, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    ar_right_x = bg_x1 + (box_w + ar_w) // 2
    ar_y = bg_y1 + padding
    put_arabic_text(img, label_ar,
                    position=(ar_right_x, ar_y),
                    font_size=font_size,
                    color=(255, 255, 255),
                    align="right")
    return img