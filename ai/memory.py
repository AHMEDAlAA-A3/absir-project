import time
import threading
from collections import defaultdict
MAX_HISTORY = 10
SESSION_TTL = 3600
class ConversationMemory:
    def __init__(self):
        self._store = defaultdict(list)
        self._last_access = {}
        self._lock = threading.Lock()
    def add(self, session_id, *, role, content, intent=None, vision_data=None):
        entry = {
            "role":        role,
            "content":     content,
            "intent":      intent,
            "vision_data": vision_data,
            "timestamp":   time.time(),
        }
        with self._lock:
            history = self._store[session_id]
            history.append(entry)
            if len(history) > MAX_HISTORY:
                self._store[session_id] = history[-MAX_HISTORY:]
            self._last_access[session_id] = time.time()
    def get_history(self, session_id):
        with self._lock:
            self._cleanup_expired()
            self._last_access[session_id] = time.time()
            return list(self._store.get(session_id, []))
    def get_last(self, session_id):
        with self._lock:
            self._cleanup_expired()
            self._last_access[session_id] = time.time()
            history = self._store.get(session_id, [])
            return history[-1] if history else None
    def get_last_vision_data(self, session_id):
        for msg in reversed(self.get_history(session_id)):
            if msg.get("vision_data"):
                return msg["vision_data"]
        return None
    def get_last_intent(self, session_id):
        for msg in reversed(self.get_history(session_id)):
            if msg.get("intent"):
                return msg["intent"]
        return None
    def clear(self, session_id):
        with self._lock:
            self._store.pop(session_id, None)
            self._last_access.pop(session_id, None)
    def cleanup_expired(self):
        with self._lock:
            self._cleanup_expired()
    def _cleanup_expired(self):
        now = time.time()
        expired = [sid for sid, ts in self._last_access.items() if now - ts > SESSION_TTL]
        for sid in expired:
            self._store.pop(sid, None)
            self._last_access.pop(sid, None)
memory = ConversationMemory()
def add(session_id, role, content, intent=None, vision_data=None):
    memory.add(
        session_id=session_id,
        role=role,
        content=content,
        intent=intent,
        vision_data=vision_data,
    )
def get(session_id):
    return [
        {"role": h["role"], "content": h["content"]}
        for h in memory.get_history(session_id)
        if h.get("content")
    ]