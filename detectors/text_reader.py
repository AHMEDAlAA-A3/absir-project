import cv2
import pytesseract
import numpy as np
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except Exception:
    ARABIC_SUPPORT = False
class TextReader:
    TESSERACT_PATH = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    def __init__(self):
        import os
        import platform
        if (
            platform.system() == "Windows"
            and os.path.exists(self.TESSERACT_PATH)
        ):
            pytesseract.pytesseract.tesseract_cmd = (
                self.TESSERACT_PATH
            )
    def preprocess(self, frame):
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )
        gray = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        return gray
    def annotate_image(self, frame):
        annotated = frame.copy()
        processed = self.preprocess(frame)
        try:
            data = pytesseract.image_to_data(
                processed,
                lang="ara+eng",
                output_type=pytesseract.Output.DICT
            )
        except Exception:
            return annotated
        total = len(data["text"])
        for i in range(total):
            try:
                text = (
                    data["text"][i]
                    .strip()
                )
                conf = int(
                    data["conf"][i]
                )
                if conf < 40 or not text:
                    continue
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 180),
                    2
                )
                cv2.putText(
                    annotated,
                    f"{conf}%",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
            except Exception:
                continue
        return annotated
    def read_image(self, image_path_or_frame):
        if isinstance(image_path_or_frame, str):
            frame = cv2.imread(
                image_path_or_frame
            )
        else:
            frame = image_path_or_frame
        if frame is None:
            return None
        processed = self.preprocess(frame)
        try:
            text = pytesseract.image_to_string(
                processed,
                lang="ara+eng"
            ).strip()
        except Exception:
            return None
        if len(text) < 2:
            return None
        display_text = text
        if ARABIC_SUPPORT:
            try:
                reshaped = arabic_reshaper.reshape(
                    text
                )
                display_text = get_display(
                    reshaped
                )
            except Exception:
                pass
        annotated = self.annotate_image(
            frame
        )
        return {
            "text": display_text,
            "raw_text": text,
            "message": (
                f"النص المكتوب: {text}"
            ),
            "annotated_frame": annotated,
        }