import cv2
import time
import numpy as np
from absir_system import ABSIRSystem
from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
from utils.voice import VoiceEngine
_RGB_TO_LMS = np.array([
    [0.31399022, 0.63951294, 0.04649755],
    [0.15537241, 0.75789446, 0.08670142],
    [0.01775239, 0.10944209, 0.87256922],
], dtype=np.float32)
_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)
_SIM_MATS = {
    "protanopia":   np.array([[0.0, 2.02344, -2.52581], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    "deuteranopia": np.array([[1.0, 0.0, 0.0], [0.49421, 0.0, 1.24827], [0.0, 0.0, 1.0]], dtype=np.float32),
    "tritanopia":   np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.86744, 1.86744, 0.0]], dtype=np.float32),
}
_SHIFT_MATS = {
    "protanopia":   np.array([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]], dtype=np.float32),
    "deuteranopia": np.array([[1, 0.7, 0], [0, 0, 0], [0, 0.7, 1]], dtype=np.float32),
    "tritanopia":   np.array([[1, 0, 0.7], [0, 1, 0.7], [0, 0, 0]], dtype=np.float32),
}
CB_LABELS = {
    "protanopia":   "Protanopia (Red-blind)",
    "deuteranopia": "Deuteranopia (Green-blind)",
    "tritanopia":   "Tritanopia (Blue-blind)",
}
CB_MODES = ["protanopia", "deuteranopia", "tritanopia"]
_CB_COLORS = {
    "protanopia":   (50, 50, 232),
    "deuteranopia": (50, 196, 50),
    "tritanopia":   (232, 130, 50),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
SPEAK_INTERVAL = 5.0
def _srgb_to_linear(img):
    img = img / 255.0
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
def _linear_to_srgb(img):
    img = np.clip(img, 0, 1)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * img ** (1 / 2.4) - 0.055)
def daltonize_frame(bgr, cb_mode):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    linear = _srgb_to_linear(rgb).astype(np.float32)
    h, w, _ = linear.shape
    pixels = linear.reshape(-1, 3)
    lms = pixels @ _RGB_TO_LMS.T
    lms_sim = lms @ _SIM_MATS[cb_mode].T
    rgb_sim = np.clip(lms_sim @ _LMS_TO_RGB.T, 0, 1).reshape(h, w, 3)
    simulated = _linear_to_srgb(rgb_sim).astype(np.float32)
    error = linear - simulated
    correction = np.einsum("hwc,dc->hwd", error, _SHIFT_MATS[cb_mode])
    enhanced = np.clip(linear + correction, 0, 1)
    result_rgb = (_linear_to_srgb(enhanced) * 255).astype(np.uint8)
    gray = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY)
    edges = np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3)) * 0.15, 0, 50).astype(np.uint8)
    edge3 = np.stack([edges] * 3, axis=-1)
    result_rgb = np.clip(result_rgb.astype(np.int16) + edge3, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
class FPSCounter:
    def __init__(self, window=30):
        self._t = []
        self._w = window
    def tick(self):
        self._t.append(time.time())
        if len(self._t) > self._w:
            self._t.pop(0)
    @property
    def fps(self):
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0] + 1e-6)
def _draw_hud(frame, mode, cb_mode, voice_on, fps_val):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (440, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
    cv2.putText(frame, "ABSIR Vision", (10, 26), _FONT, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Mode: {mode.upper()}", (10, 52), _FONT, 0.6, (255, 255, 255), 2)
    vc = (0, 255, 0) if voice_on else (0, 0, 255)
    cv2.putText(frame, f"Voice: {'ON' if voice_on else 'OFF'}  |  FPS: {fps_val:.1f}",
                (10, 76), _FONT, 0.5, vc, 1)
def _draw_colorblind_hud(frame, cb_mode):
    h, w = frame.shape[:2]
    label = CB_LABELS[cb_mode]
    accent = _CB_COLORS[cb_mode]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 34), (380, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"ColorBlind Filter: {label}", (8, h - 10), _FONT, 0.50, accent, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), accent, 2)
def _draw_danger_banner(frame, danger):
    from utils.arabic_utils import put_arabic_text, measure_arabic_text
    h, w = frame.shape[:2]
    text = f"احترس، {danger['name_ar']} أمامك"
    font_size = 28
    tw, th = measure_arabic_text(text, font_size=font_size)
    x, y = (w - tw) // 2, h - 50
    ov2 = frame.copy()
    cv2.rectangle(ov2, (x - 12, y - 10), (x + tw + 12, y + th + 10), (0, 0, 160), -1)
    cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)
    put_arabic_text(frame, text, position=(x + tw, y), font_size=font_size, color=(255, 255, 255), align="right")
def main():
    print("=" * 60)
    print("  ABSIR Vision System")
    print("=" * 60)
    print("  ESC / Q   Exit")
    print("  C         Currency mode")
    print("  O         Object mode")
    print("  A         Auto mode")
    print("  B         Colorblind mode")
    print("  SPACE     Toggle Voice")
    print("=" * 60)
    system = ABSIRSystem()
    voice = VoiceEngine()
    fps = FPSCounter()
    _last_spoken = {"objects": set(), "time": 0.0}
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Cannot open camera!")
        return
    mode = "auto"
    cb_mode = "deuteranopia"
    cb_idx = 1
    voice_on = True
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fps.tick()
        if mode == "colorblind":
            display = daltonize_frame(frame, cb_mode)
            _draw_hud(display, mode, cb_mode, voice_on, fps.fps)
            _draw_colorblind_hud(display, cb_mode)
        else:
            ann_frame, detections, danger = system.process_frame(frame, mode=mode)
            display = ann_frame if ann_frame is not None else frame
            _draw_hud(display, mode, cb_mode, voice_on, fps.fps)
            if danger:
                _draw_danger_banner(display, danger)
            if voice_on and detections:
                now = time.time()
                names = {d["name_ar"] for d in detections[:3]}
                if (now - _last_spoken["time"]) >= SPEAK_INTERVAL or names != _last_spoken["objects"]:
                    if danger:
                        msg = f"خلي بالك، أمامك {danger['name_ar']}"
                    elif len(names) == 1:
                        msg = f"شايف {list(names)[0]}"
                    else:
                        msg = "شايف " + " و ".join(names)
                    voice.speak(msg)
                    _last_spoken["objects"] = names
                    _last_spoken["time"] = now
        cv2.imshow("ABSIR Vision", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('c'):
            mode = "currency"
        elif key == ord('o'):
            mode = "object"
        elif key == ord('a'):
            mode = "auto"
        elif key == ord('b'):
            if mode != "colorblind":
                mode = "colorblind"
            else:
                cb_idx = (cb_idx + 1) % len(CB_MODES)
                cb_mode = CB_MODES[cb_idx]
            print(f"[ColorBlind] mode → {cb_mode}")
        elif key == ord(' '):
            voice_on = not voice_on
            VoiceEngine.set_mute(not voice_on)
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")
if __name__ == "__main__":
    main()