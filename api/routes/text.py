from fastapi import APIRouter, File, UploadFile, Form
from api.services.camera_service import camera
from api.services.processing_service import (
    load_frame_from_upload,
    frame_to_b64,
    save_annotated_frame
)
from utils.voice import VoiceEngine
router = APIRouter(
    prefix="/api/text",
    tags=["Text Recognition"]
)
_sys = None
_voice = VoiceEngine()
def sys():
    global _sys
    if _sys is None:
        from absir_system import ABSIRSystem
        _sys = ABSIRSystem()
    return _sys
async def _build(
    frame,
    input_type: str,
    return_image: bool = True
) -> dict:
    result = sys().text_reader.read_image(frame)
    if not result:
        audio = await _voice.to_audio_b64(
            "لم يتم اكتشاف نص"
        )
        return {
            "status": "success",
            "mode": "text",
            "input_type": input_type,
            "message": "لم يتم اكتشاف نص",
            "audio_b64": audio,
            "danger": None,
            "detections": [],
            "extra": {
                "text": None
            },
            "image_b64": None,
        }
    ann_frame = result.get("annotated_frame")
    image_b64 = None
    if return_image and ann_frame is not None:
        image_b64 = frame_to_b64(ann_frame)
    if ann_frame is not None:
        save_annotated_frame(ann_frame, "text")
    message = result.get("message")
    audio = None
    if message:
        audio = await _voice.to_audio_b64(message)
    return {
        "status": "success",
        "mode": "text",
        "input_type": input_type,
        "message": message,
        "audio_b64": audio,
        "danger": None,
        "detections": [],
        "extra": {
            "text": result.get("text"),
            "raw_text": result.get("raw_text")
        },
        "image_b64": image_b64,
    }
@router.post("/upload")
async def text_upload(
    file: UploadFile = File(...),
    return_image: bool = Form(True),
):
    frame = load_frame_from_upload(file)
    return await _build(
        frame,
        "upload",
        return_image
    )
@router.post("/capture")
async def text_capture(
    return_image: bool = Form(True)
):
    frame = camera.capture()
    if frame is None:
        return {
            "status": "error",
            "message": "Camera not available"
        }
    return await _build(
        frame,
        "capture",
        return_image
    )