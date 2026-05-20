from fastapi import APIRouter, File, UploadFile, Query
from api.services.camera_service import camera
from api.services.processing_service import load_frame_from_upload
from detectors.color_blind import process as cb_process, TYPES
from utils.voice import VoiceEngine
router = APIRouter(prefix="/api/color", tags=["Color Recognition"])
_sys = None
_voice = VoiceEngine()
def sys():
    global _sys
    if _sys is None:
        from absir_system import ABSIRSystem
        _sys = ABSIRSystem()
    return _sys
async def _build(frame, input_type: str, cb_type: str) -> dict:
    color_info = sys().color_recognizer.detect_dominant_color(frame)
    result = cb_process(frame, cb_type)
    message = f"اللون الغالب هو {color_info['color_ar']}."
    if cb_type != "none":
        message += f" وتم تصحيح الصورة لنوع: {result['struggle']}."
    return {
        "status": "success",
        "mode": "color",
        "input_type": input_type,
        "message": message,
        "audio_b64": await _voice.to_audio_b64(message),
        "danger": None,
        "detections": [],
        "extra": {
            "type": result["type"],
            "struggle": result["struggle"],
            "detected_color": color_info,
            "original_colors": result["original_colors"],
            "corrected_colors": result["corrected_colors"],
        },
        "image_b64": result["corrected_b64"],
    }
@router.post("/upload")
async def color_upload(
    file: UploadFile = File(...),
    type: str = Query("none", enum=TYPES + ["none"]),
):
    frame = load_frame_from_upload(file)
    return await _build(frame, "upload", type)
@router.post("/capture")
async def color_capture(
    type: str = Query("none", enum=TYPES + ["none"])
):
    frame = camera.capture()
    if frame is None:
        return {
            "status": "error",
            "message": "Camera not available"
        }
    return await _build(frame, "capture", type)