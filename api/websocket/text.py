import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from api.services.processing_service import load_frame_from_b64, frame_to_b64
from utils.voice import VoiceEngine
_sys = None
_voice = VoiceEngine()
def sys():
    global _sys
    if _sys is None:
        from absir_system import ABSIRSystem
        _sys = ABSIRSystem()
    return _sys
async def text_stream(websocket: WebSocket):
    """
    WS /ws/text/stream
    Continuous OCR on incoming frames.
    Useful for live reading (signs, labels, documents).
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"status": "ping"}))
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps({"status": "error", "message": "Invalid JSON"}))
                continue
            b64 = payload.get("image_b64", "")
            if not b64:
                await websocket.send_text(json.dumps({"status": "error", "message": "Missing image_b64"}))
                continue
            try:
                frame = load_frame_from_b64(b64)
            except Exception:
                await websocket.send_text(json.dumps({"status": "error", "message": "Bad frame"}))
                continue
            result = await loop.run_in_executor(
                None, sys().text_reader.read_image, frame
            )
            return_image = payload.get("return_image", True)
            if result:
                ann_frame = result.get("annotated_frame")
                image_b64 = frame_to_b64(ann_frame, 65) if (return_image and ann_frame is not None) else None
                message = result.get("message")
                resp = {
                    "status":     "success",
                    "mode":       "text",
                    "input_type": "stream",
                    "message":    message,
                    "audio_b64":  _voice.to_audio_b64(message) if message else None,
                    "danger":     None,
                    "detections": [],
                    "extra":      {"text": result.get("text"), "raw_text": result.get("raw_text")},
                    "image_b64":  image_b64,
                }
            else:
                message = "لم يتم اكتشاف نص"
                resp = {
                    "status": "success", "mode": "text", "input_type": "stream",
                    "message": message, "danger": None, "detections": [],
                    "audio_b64": _voice.to_audio_b64(message),
                    "extra": {"text": None, "raw_text": None}, "image_b64": None,
                }
            await websocket.send_text(json.dumps(resp, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))
        except Exception:
            pass