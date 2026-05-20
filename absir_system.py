import traceback
import threading
from detectors.object_detector import ObjectDetector
from detectors.currency_detector import CurrencyDetector
from detectors.color_recognizer import ColorRecognizer
from detectors.text_reader import TextReader
from utils.danger_alert import DangerAlert
from config.settings import OBJECTS_MODEL_PATH, CURRENCY_MODEL_PATH, DANGER_COOLDOWN
class ABSIRSystem:
    _instance = None
    _lock = threading.Lock()
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._ocr_lock = threading.Lock()
        self._obj_lock = threading.Lock()
        self._curr_lock = threading.Lock()
        
        self._object_detector = None
        self._currency_detector = None
        self._text_reader = None
        
        print("\n[ABSIR] System initialized (Models lazy-loaded)...\n")
        try:
            self.color_recognizer = ColorRecognizer()
            self.danger_alert = DangerAlert(cooldown=DANGER_COOLDOWN)
        except Exception:
            traceback.print_exc()
            raise RuntimeError("ABSIR initialization failed")

    @property
    def object_detector(self):
        if self._object_detector is None:
            with self._obj_lock:
                if self._object_detector is None:
                    print("[ABSIR] Loading Object Detector (YOLO)...")
                    self._object_detector = ObjectDetector(OBJECTS_MODEL_PATH)
        return self._object_detector

    @property
    def currency_detector(self):
        if self._currency_detector is None:
            with self._curr_lock:
                if self._currency_detector is None:
                    print("[ABSIR] Loading Currency Detector (YOLO)...")
                    self._currency_detector = CurrencyDetector(CURRENCY_MODEL_PATH)
        return self._currency_detector

    @property
    def text_reader(self):
        if self._text_reader is None:
            with self._ocr_lock:
                if self._text_reader is None:
                    print("[ABSIR] Loading OCR...")
                    self._text_reader = TextReader()
        return self._text_reader
    def analyze_danger(self, frame_shape):
        try:
            last_results, names = self.object_detector.get_last_results()
            return self.danger_alert.process(last_results, names, frame_shape)
        except Exception:
            traceback.print_exc()
            return None
    def process_frame(self, frame, mode="auto"):
        try:
            if frame is None:
                return None, [], None
            if mode == "object":
                ann, raw = self.object_detector.detect_frame(frame)
                danger = self.analyze_danger(frame.shape)
                return ann if ann is not None else frame, raw, danger
            elif mode == "currency":
                ann, raw = self.currency_detector.detect_frame(frame)
                return ann if ann is not None else frame, raw, None
            elif mode == "text":
                res = self.text_reader.read_image(frame)
                if res:
                    return res.get("annotated_frame", frame), [res], None
                return frame, [], None
            elif mode == "color":
                result = self.color_recognizer.detect_dominant_color(frame)
                return frame, result or [], None
            ann, raw = self.object_detector.detect_frame(frame)
            if raw:
                danger = self.analyze_danger(frame.shape)
                return ann, raw, danger
            ann, raw = self.currency_detector.detect_frame(frame)
            if raw:
                return ann, raw, None
            return frame, [], None
        except Exception:
            traceback.print_exc()
            return frame, [], None