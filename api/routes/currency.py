import cv2
from fastapi import APIRouter, File, UploadFile, Form
from api.services.camera_service import camera
from api.services.processing_service import (
    load_frame_from_upload,
    frame_to_b64,
    save_annotated_frame
)
from utils.voice import VoiceEngine
router = APIRouter(
    prefix="/api/currency",
    tags=["Currency Detection"]
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
    return_image: bool
) -> dict:
    result = sys().currency_detector.detect_currency(frame)
    if not result:
        return {
            "status": "success",
            "mode": "currency",
            "input_type": input_type,
            "message": "لم يتم اكتشاف عملة",
            "audio_b64":
                await _voice.to_audio_b64(
                    "لم يتم اكتشاف عملة"
                ),
            "danger": None,
            "detections": [],
            "extra": {"total": 0},
            "image_b64": None,
        }
    ann, raw = sys().currency_detector.detect_frame(frame)
    if ann is not None:
        save_annotated_frame(ann, "currency")
    message = result.get("message")
    return {
        "status": "success",
        "mode": "currency",
        "input_type": input_type,
        "message": message,
        "audio_b64":
            await _voice.to_audio_b64(message)
            if message else None,
        "danger": None,
        "detections": [
            {
                "name_en": d.get("name_en", ""),
                "name_ar": d.get("name_ar", ""),
                "confidence": round(
                    float(d.get("confidence", 0)),
                    2
                ),
                "bbox":
                    {
                        k: int(v)
                        for k, v in d["bbox"].items()
                    }
                    if "bbox" in d else None,
            }
            for d in (result.get("detections") or [])
        ],
        "extra": {
            "total": result.get("total", 0),
            "detected": result.get("detected", [])
        },
        "image_b64":
            frame_to_b64(ann)
            if (return_image and ann is not None)
            else None,
    }
@router.post("/upload")
async def currency_upload(
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
async def currency_capture(
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