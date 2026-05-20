import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from api.services.processing_service import load_frame_from_b64
from detectors.color_blind import process as cb_process, TYPES
from utils.voice import VoiceEngine
_voice = VoiceEngine()
async def color_stream(websocket: WebSocket, cb_type: str = "protanopia"):
    """
    WS /ws/color/stream?type=protanopia
    Live color blindness correction for continuous camera feed.
    """
    if cb_type not in TYPES:
        cb_type = "protanopia"
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
            frame_type = payload.get("type", cb_type)
            if frame_type not in TYPES:
                frame_type = cb_type
            try:
                frame = load_frame_from_b64(b64)
            except Exception:
                await websocket.send_text(json.dumps({"status": "error", "message": "Bad frame"}))
                continue
            result = await loop.run_in_executor(None, cb_process, frame, frame_type)
            message = f"تم تصحيح الصورة — {result['struggle']}"
            resp = {
                "status":     "success",
                "mode":       "color",
                "input_type": "stream",
                "message":    message,
                "audio_b64":  _voice.to_audio_b64(message),
                "danger":     None,
                "detections": [],
                "extra": {
                    "type":             result["type"],
                    "struggle":         result["struggle"],
                    "original_colors":  result["original_colors"],
                    "corrected_colors": result["corrected_colors"],
                    "simulated_b64":    result["simulated_b64"],
                },
                "image_b64": result["corrected_b64"],
            }
            await websocket.send_text(json.dumps(resp, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))
        except Exception:
            pass