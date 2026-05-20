from fastapi import APIRouter, File, UploadFile, Form
from api.services.camera_service import camera
from api.services.processing_service import (
    load_frame_from_upload,
    frame_to_b64,
    clean_det,
    clean_danger,
    save_annotated_frame,
)
from utils.voice import VoiceEngine
router = APIRouter(prefix="/api/object", tags=["Object Detection"])
_voice = VoiceEngine()
_system = None
def _get_system():
    global _system
    if _system is None:
        from absir_system import ABSIRSystem
        _system = ABSIRSystem()
    return _system
def _build_message(detections, danger):
    if danger:
        return f"خلي بالك، أمامك {danger['name_ar']}"
    names = [d["name_ar"] for d in (detections or [])[:3]]
    if names:
        return "شايف " + " و ".join(names)
    return "مش شايف حاجة واضحة."
async def _process_and_respond(frame, input_type, return_image):
    ann, detections, danger = _get_system().process_frame(frame, mode="object")
    _, danger_dict = clean_danger(danger)
    message = _build_message(detections, danger)
    if detections:
        save_annotated_frame(ann, "object")
    audio_b64 = await _voice.to_audio_b64(message)
    return {
        "status":     "success",
        "mode":       "object",
        "input_type": input_type,
        "message":    message,
        "audio_b64":  audio_b64,
        "danger":     danger_dict,
        "detections": [clean_det(d) for d in (detections or [])],
        "extra":      None,
        "image_b64":  frame_to_b64(ann) if return_image else None,
    }
@router.post("/upload")
async def object_upload(
    file: UploadFile = File(...),
    return_image: bool = Form(True),
):
    frame = load_frame_from_upload(file)
    return await _process_and_respond(frame, "upload", return_image)
@router.post("/capture")
async def object_capture(return_image: bool = Form(True)):
    frame = camera.capture()
    if frame is None:
        return {"status": "error", "message": "الكاميرا مش متاحة."}
    return await _process_and_respond(frame, "capture", return_image)