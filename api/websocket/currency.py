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
async def currency_stream(websocket: WebSocket):
    """WS /ws/currency/stream"""
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
                continue
            b64 = payload.get("image_b64", "")
            if not b64:
                continue
            try:
                frame = load_frame_from_b64(b64)
            except Exception:
                continue
            return_image = payload.get("return_image", True)
            ann, raw_dets = await loop.run_in_executor(
                None, sys().currency_detector.detect_frame, frame
            )
            result = sys().currency_detector.detect_currency(frame)
            message = result.get("message") if result else None
            total   = result.get("total", 0) if result else 0
            audio_b64 = _voice.to_audio_b64(message) if message else None
            resp = {
                "status":     "success",
                "mode":       "currency",
                "input_type": "stream",
                "message":    message,
                "audio_b64":  audio_b64,
                "danger":     None,
                "detections": [
                    {
                        "name_en":    d.get("name_en", ""),
                        "name_ar":    d.get("name_ar", ""),
                        "confidence": round(float(d.get("confidence", 0)), 2),
                        "bbox":       {k: int(v) for k, v in d["bbox"].items()}
                                      if "bbox" in d else None,
                    }
                    for d in (result.get("detections") or [])
                ],
                "extra":     {"total": total, "detected": result.get("detected", [])},
                "image_b64": frame_to_b64(ann, 65) if (return_image and ann is not None) else None,
            }
            await websocket.send_text(json.dumps(resp, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))
        except Exception:
            pass