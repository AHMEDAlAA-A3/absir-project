
from fastapi import (
    APIRouter,
    File,
    UploadFile,
    Form
)
from ai.assistant import ask as assistant_ask
from ai.memory import memory
from api.services.processing_service import (
    load_frame_from_upload
)
from utils.voice import VoiceEngine
router = APIRouter(
    prefix="/api/assistant",
    tags=["Smart Assistant"]
)
_voice = VoiceEngine()
@router.post("/ask")
async def assistant_ask_text(
    query: str = Form(...),
    session_id: str = Form("default"),
):
    result = await assistant_ask(
        query=query,
        frame=None,
        session_id=session_id
    )
    if result.get("message"):
        result["audio_b64"] = (
            await _voice.to_audio_b64(
                result["message"]
            )
        )
    return result
@router.post("/ask-image")
async def assistant_ask_with_image(
    query: str = Form(...),
    file: UploadFile = File(...),
    session_id: str = Form("default"),
):
    try:
        frame = load_frame_from_upload(file)
    except Exception:
        err = {
            "status": "error",
            "intent": "unknown",
            "message": "مش قادر أقرأ الصورة.",
            "detections": [],
            "extra": {},
            "session_id": session_id,
        }
        err["audio_b64"] = (
            await _voice.to_audio_b64(
                err["message"]
            )
        )
        return err
    result = await assistant_ask(
        query=query,
        frame=frame,
        session_id=session_id
    )
    if result.get("message"):
        result["audio_b64"] = (
            await _voice.to_audio_b64(
                result["message"]
            )
        )
    return result
@router.post("/clear")
async def assistant_clear(
    session_id: str = Form("default")
):
    memory.clear(session_id)
    return {
        "status": "success",
        "message": "تم مسح الجلسة.",
        "session_id": session_id,
    }
@router.get("/history")
async def assistant_history(
    session_id: str = "default"
):
    history = memory.get_history(
        session_id
    )
    return {
        "status": "success",
        "session_id": session_id,
        "count": len(history),
        "history": history,
    }