import asyncio
import traceback
import cv2
import numpy as np
from groq import Groq
from ai.intent import classify, needs_context
from ai.memory import get as mem_get, add as mem_add
from absir_system import ABSIRSystem
from config.settings import GROQ_API_KEY
from ai.assistant_logic import (
    SYSTEM_PROMPT,
    build_empty_image_response,
    build_low_quality_response,
    build_response,
)
if not GROQ_API_KEY or not GROQ_API_KEY.strip():
    raise ValueError("GROQ_API_KEY not found in .env")
client = Groq(api_key=GROQ_API_KEY.strip())
_sys = ABSIRSystem()
_object_det = _sys.object_detector
_text_reader = _sys.text_reader
_color_rec = _sys.color_recognizer
_currency_det = _sys.currency_detector
def _is_dark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < 25
def _is_blurry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 40
def _describe_positions(detections, width):
    results = []
    for d in detections[:5]:
        conf = round(d.get("confidence", 0) * 100)
        if conf < 40:
            continue
        bbox = d.get("bbox", {})
        x1 = bbox.get("x1", 0)
        x2 = bbox.get("x2", 0)
        center = (x1 + x2) / 2
        if center < width * 0.33:
            pos = "على الشمال"
        elif center > width * 0.66:
            pos = "على اليمين"
        else:
            pos = "قدامك"
        results.append(f"{d['name_ar']} {pos}")
    return results
async def ask(query: str, session_id: str, frame: np.ndarray | None = None):
    loop = asyncio.get_event_loop()
    if frame is None:
        if needs_context(query):
            last = mem_get(session_id)
            if last:
                return {
                    "status": "error",
                    "message": "محتاج صورة جديدة علشان أقدر أساعدك.",
                    "suggestions": [
                        "ارفع صورة",
                        "وصف الصورة",
                        "اقرالي المكتوب",
                        "ايه اللي قدامي",
                    ],
                }
        return build_empty_image_response()
    if _is_dark(frame) or _is_blurry(frame):
        return build_low_quality_response()
    classified = classify(query)
    intent = classified["intent"]
    mem_add(session_id=session_id, role="user", content=query, intent=intent)
    detections = []
    extra = {}
    vision_context = ""
    h, w = frame.shape[:2]
    if intent == "object":
        result = await loop.run_in_executor(None, _object_det.detect_frame, frame)
        detections = result[1] if result else []
        vision_context = (
            "الأشياء الظاهرة: "
            + ", ".join(_describe_positions(detections, w))
            if detections
            else "لم يتم اكتشاف عناصر واضحة."
        )
    elif intent == "text":
        result = await loop.run_in_executor(None, _text_reader.read_image, frame)
        text = (result or {}).get("text", "")
        extra["text"] = text
        vision_context = f"النص الموجود: {text[:300]}" if text.strip() else "لا يوجد نص واضح."
    elif intent == "color":
        result = await loop.run_in_executor(None, _color_rec.detect_dominant_color, frame)
        color = (result or {}).get("color_ar", "غير معروف")
        extra["color"] = color
        vision_context = f"اللون الغالب: {color}"
    elif intent == "currency":
        result = await loop.run_in_executor(None, _currency_det.detect_currency, frame)
        detections = (result or {}).get("detections", [])
        vision_context = (result or {}).get("message", "")
    elif intent in ("scene", "unknown"):
        obj_task = loop.run_in_executor(None, _object_det.detect_frame, frame)
        txt_task = loop.run_in_executor(None, _text_reader.read_image, frame)
        col_task = loop.run_in_executor(None, _color_rec.detect_dominant_color, frame)
        obj_r, txt_r, col_r = await asyncio.gather(obj_task, txt_task, col_task)
        parts = []
        if obj_r and obj_r[1]:
            detections = obj_r[1]
            pos = _describe_positions(detections, w)
            if pos:
                parts.append("العناصر: " + ", ".join(pos))
        if txt_r and txt_r.get("text"):
            extra["text"] = txt_r["text"]
            parts.append(f"النص: {txt_r['text'][:120]}")
        if col_r:
            extra["color"] = col_r.get("color_ar", "غير معروف")
            parts.append(f"اللون الغالب: {extra['color']}")
        vision_context = " | ".join(parts) if parts else "الصورة غير واضحة."
    history = mem_get(session_id)
    history_text = "\n".join(
        f"{'المستخدم' if m['role']=='user' else 'ABSIR'}: {m['content']}"
        for m in history[-6:]
    )
    full_prompt = "\n".join(
        [
            SYSTEM_PROMPT,
            f"[معلومات الصورة]\n{vision_context}" if vision_context else "",
            f"[المحادثة السابقة]\n{history_text}" if history_text else "",
            f"[سؤال المستخدم]\n{query}",
            "[ردك]",
        ]
    )
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.4,
                max_tokens=300,
            ),
        )
        message = response.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        print(e)
        message = "حصلت مشكلة أثناء تحليل الصورة."
    mem_add(session_id=session_id, role="assistant", content=message)
    return build_response(
        ai_reply=message,
        detections=detections,
        mode=intent,
        extracted_text=extra.get("text"),
        image_b64=None,
        audio_b64=None,
    )