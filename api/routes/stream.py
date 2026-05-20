import cv2
import asyncio
import numpy as np
from fastapi import APIRouter, Query, File, UploadFile
from fastapi.responses import StreamingResponse
from utils.voice import VoiceEngine
from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
router = APIRouter(prefix="/api", tags=["Live Stream"])
_voice = VoiceEngine()
_sys = None
def _get_sys():
    global _sys
    if _sys is None:
        from absir_system import ABSIRSystem
        _sys = ABSIRSystem()
    return _sys
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
_VALID_CB = ("protanopia", "deuteranopia", "tritanopia")
_FONT = cv2.FONT_HERSHEY_SIMPLEX
def _srgb_to_linear(img):
    img = img / 255.0
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
def _linear_to_srgb(img):
    img = np.clip(img, 0, 1)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * img ** (1 / 2.4) - 0.055)
def _daltonize(bgr, cb_mode):
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
def _draw_hud(frame, mode, cb_label=""):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (460, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
    cv2.putText(frame, "ABSIR Vision  [LIVE]", (10, 26), _FONT, 0.7, (0, 255, 128), 2, cv2.LINE_AA)
    label = cb_label if cb_label else mode.upper()
    cv2.putText(frame, f"Mode: {label}", (10, 55), _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "ABSIR v3.0", (10, 80), _FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
def _draw_cb_border(frame, cb_mode):
    accent_map = {"protanopia": (50, 50, 232), "deuteranopia": (50, 196, 50), "tritanopia": (232, 130, 50)}
    h, w = frame.shape[:2]
    accent = accent_map.get(cb_mode, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), accent, 3)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 34), (420, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"ColorBlind Filter: {CB_LABELS.get(cb_mode, cb_mode)}", (8, h - 10),
                _FONT, 0.50, accent, 1, cv2.LINE_AA)
def _frame_to_b64(frame, quality=85):
    import base64
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()
@router.post("/stream/frame")
async def process_stream_frame(
    file:    UploadFile = File(...),
    mode:    str        = Query("object"),
    cb:      str        = Query("deuteranopia"),
    quality: int        = Query(70, ge=1, le=95),
):
    if cb not in _VALID_CB:
        cb = "deuteranopia"
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "error", "message": "Invalid frame"}
    loop = asyncio.get_event_loop()
    if mode == "colorblind":
        display = await loop.run_in_executor(None, _daltonize, frame, cb)
        _draw_hud(display, mode, CB_LABELS.get(cb, cb))
        _draw_cb_border(display, cb)
        message = CB_LABELS.get(cb, cb)
        detections = []
        danger = None
    else:
        ann, detections, danger = await loop.run_in_executor(
            None, _get_sys().process_frame, frame, mode
        )
        display = ann if ann is not None else frame
        _draw_hud(display, mode)
        names = [d["name_ar"] for d in (detections or [])[:3]]
        if danger:
            message = f"خلي بالك، أمامك {danger['name_ar']}"
        elif names:
            message = "شايف " + " و ".join(names)
        else:
            message = None
    audio_b64 = await _voice.to_audio_b64(message) if message else None
    return {
        "status":     "success",
        "mode":       mode,
        "message":    message,
        "audio_b64":  audio_b64,
        "detections": detections or [],
        "danger":     danger,
        "image_b64":  _frame_to_b64(display, quality),
    }
def _generate_frames(mode, cb, quality):
    if cb not in _VALID_CB:
        cb = "deuteranopia"
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        err = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(err, "Camera not available", (30, 130), _FONT, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        _, buf = cv2.imencode(".jpg", err)
        yield buf.tobytes()
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if mode == "colorblind":
                display = _daltonize(frame, cb)
                _draw_hud(display, mode, CB_LABELS.get(cb, cb))
                _draw_cb_border(display, cb)
            else:
                ann, _, _ = _get_sys().process_frame(frame, mode=mode)
                display = ann if ann is not None else frame
                _draw_hud(display, mode)
            ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                yield buf.tobytes()
    finally:
        cap.release()
async def _async_mjpeg(mode, cb, quality):
    boundary = b"--frame\r\n"
    loop = asyncio.get_event_loop()
    gen = _generate_frames(mode, cb, quality)
    while True:
        try:
            jpeg = await loop.run_in_executor(None, next, gen)
        except StopIteration:
            break
        header = (
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
        )
        yield boundary + header + jpeg + b"\r\n"
@router.get("/stream")
async def live_stream(
    mode:    str = Query("object"),
    cb:      str = Query("deuteranopia"),
    quality: int = Query(70, ge=1, le=95),
):
    return StreamingResponse(
        _async_mjpeg(mode, cb, quality),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma":        "no-cache",
            "Expires":       "0",
        },
    )