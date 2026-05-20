import asyncio
import traceback
import warnings
import cv2
import numpy as np
from ai.intent import classify, needs_context, VISION_INTENTS
from ai.memory import get as mem_get, add as mem_add
from ai.assistant_logic import (
    SYSTEM_PROMPT,
    build_empty_image_response,
    build_low_quality_response,
    build_response,
)
from config.settings import GROQ_API_KEY
from absir_system import ABSIRSystem
_client = None
if not GROQ_API_KEY:
    warnings.warn(
        "GROQ_API_KEY missing — AI assistant disabled",
        RuntimeWarning,
        stacklevel=1,
    )
else:
    try:
        from groq import Groq
        _client = Groq(api_key=GROQ_API_KEY)
        print("[ABSIR] Groq client ready.")
    except Exception as e:
        warnings.warn(f"Groq init failed: {e}", RuntimeWarning)
_system = None
def _get_system():
    global _system
    if _system is None:
        from absir_system import ABSIRSystem
        _system = ABSIRSystem()
    return _system
def _is_dark(frame):
    if frame is None or frame.size == 0:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < 25
def _is_blurry(frame):
    if frame is None or frame.size == 0:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 40
def _describe_positions(detections, width):
    results = []
    for d in detections[:5]:
        bbox = d.get("bbox", {})
        center = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2
        if center < width * 0.33:
            pos = "على الشمال"
        elif center > width * 0.66:
            pos = "على اليمين"
        else:
            pos = "قدامك"
        results.append(f"{d.get('name_ar', 'عنصر')} {pos}")
    return results
async def _call_groq(messages):
    if _client is None:
        return None
    last_err = None
    for attempt in range(3):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: _client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        temperature=0.6,
                        max_tokens=250,
                    )
                ),
                timeout=15.0,
            )
            return response.choices[0].message.content.strip()
        except asyncio.TimeoutError:
            last_err = "timeout"
            if attempt < 2:
                await asyncio.sleep(1)
        except Exception as e:
            last_err = str(e)
            err_lower = last_err.lower()
            if "rate" in err_lower and attempt < 2:
                await asyncio.sleep(2)
            elif "invalid" in err_lower or "auth" in err_lower:
                break
            elif attempt < 2:
                await asyncio.sleep(1)
            else:
                break
    return None
def _groq_fallback(err_type="generic", intent="general_chat"):
    fallbacks = {
        "timeout": "معلش، الاتصال اتأخر شوية. جرب تاني.",
        "rate": "كتير الأسئلة شوية، استنى ثانية وجرب.",
        "auth": "مفيش صلاحية للـ AI دلوقتي.",
        "disabled": "المساعد الذكي مش شغال دلوقتي.",
        "generic": "معلش، فيه مشكلة في الاتصال دلوقتي. جرب تاني.",
    }
    if intent in ["greeting", "capabilities", "general_chat"] and err_type == "disabled":
        return "أنا ABSIR المساعد الذكي بتاعك. للأسف خدمات المحادثة معطلة دلوقتي، بس لسه أقدر أساعدك باستخدام الكاميرا."
    return fallbacks.get(err_type, fallbacks["generic"])
async def ask(query: str, session_id: str, frame: np.ndarray | None = None):
    try:
        classified = classify(query)
        intent = classified.get("intent", "scene")
        mem_add(session_id=session_id, role="user", content=query, intent=intent)
        is_vision_intent = intent in VISION_INTENTS
        detections = []
        extra = {}
        vision_context = ""
        if is_vision_intent:
            if frame is None:
                if needs_context(query) or intent in ["object", "color", "currency", "text", "scene"]:
                    return build_empty_image_response()
            if frame is not None:
                if _is_dark(frame) or _is_blurry(frame):
                    return build_low_quality_response()
                _, frame_w = frame.shape[:2]
                if intent == "object":
                    result = await asyncio.to_thread(_get_system().object_detector.detect_frame, frame)
                    detections = result[1] if result and len(result) > 1 else []
                    if detections:
                        vision_context = "الأشياء الظاهرة: " + ", ".join(
                            _describe_positions(detections, frame_w)
                        )
                    else:
                        vision_context = "لم يتم اكتشاف عناصر واضحة."
                elif intent == "text":
                    result = await asyncio.to_thread(_get_system().text_reader.read_image, frame)
                    if result and isinstance(result, dict):
                        text = result.get("text") or ""
                        extra["text"] = text
                        vision_context = f"النص الموجود: {text[:300]}" if text.strip() else "لا يوجد نص واضح."
                elif intent == "color":
                    result = await asyncio.to_thread(_get_system().color_recognizer.detect_dominant_color, frame)
                    if result and isinstance(result, dict):
                        color_name = result.get("color_ar", "غير معروف")
                        extra["color"] = color_name
                        vision_context = f"اللون الغالب: {color_name}"
                elif intent == "currency":
                    result = await asyncio.to_thread(_get_system().currency_detector.detect_currency, frame)
                    if result and isinstance(result, dict):
                        detections = result.get("detections", [])
                        vision_context = f"نتيجة كشف العملة: {result.get('message', '')}"
                else:
                    obj_r, txt_r, col_r = await asyncio.gather(
                        asyncio.to_thread(_get_system().object_detector.detect_frame, frame),
                        asyncio.to_thread(_get_system().text_reader.read_image, frame),
                        asyncio.to_thread(_get_system().color_recognizer.detect_dominant_color, frame),
                    )
                    parts = []
                    if obj_r and len(obj_r) > 1 and obj_r[1]:
                        detections = obj_r[1]
                        parts.append("العناصر: " + ", ".join(_describe_positions(detections, frame_w)))
                    if txt_r and isinstance(txt_r, dict) and txt_r.get("text"):
                        extra["text"] = txt_r["text"]
                        parts.append(f"النص: {txt_r['text'][:120]}")
                    if col_r and isinstance(col_r, dict):
                        color_name = col_r.get("color_ar", "غير معروف")
                        extra["color"] = color_name
                        parts.append(f"اللون الغالب: {color_name}")
                    vision_context = " | ".join(parts) if parts else "الصورة غير واضحة."
        history = mem_get(session_id)
        history_text = ""
        for msg in history[-6:]:
            role = "المستخدم" if msg["role"] == "user" else "ABSIR"
            history_text += f"{role}: {msg['content']}\n"
        prompt_parts = []
        if vision_context:
            prompt_parts.append(f"[معلومات الصورة]\n{vision_context}")
        if history_text:
            prompt_parts.append(f"[المحادثة السابقة]\n{history_text}")
        prompt_parts.append(f"[سؤال المستخدم]\n{query}")
        prompt_parts.append("[ردك]")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(prompt_parts)},
        ]
        message = await _call_groq(messages)
        if message is None:
            if _client is None:
                message = _groq_fallback("disabled", intent)
            else:
                message = _groq_fallback("generic", intent)
        mem_add(session_id=session_id, role="assistant", content=message)
        return build_response(
            ai_reply=message,
            detections=detections,
            mode=intent,
            extracted_text=extra.get("text"),
            image_b64=None,
            audio_b64=None,
        )
    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "message": "حصل خطأ غير متوقع. جرب تاني.",
            "error": str(e),
        }