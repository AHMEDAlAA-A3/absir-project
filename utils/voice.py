import os
import time
import queue
import base64
import asyncio
import tempfile
import threading
import traceback
import edge_tts
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    _PYGAME_OK = True
except Exception:
    _PYGAME_OK = False
class VoiceEngine:
    _shared_queue = queue.Queue(maxsize=5)
    _shared_lock = threading.Lock()
    _last_said = {}
    _worker_active = False
    _mute = False
    def __init__(self, lang="ar", repeat_gap=3.0):
        self.lang = lang
        self.repeat_gap = repeat_gap
        with VoiceEngine._shared_lock:
            if not VoiceEngine._worker_active:
                VoiceEngine._worker_active = True
                t = threading.Thread(target=self._worker, daemon=True)
                t.start()
    def _get_voice(self):
        return "ar-EG-ShakirNeural" if self.lang == "ar" else "en-US-AriaNeural"
    async def to_audio_b64(self, text):
        if not text or not text.strip():
            return None
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            communicate = edge_tts.Communicate(text=text, voice=self._get_voice())
            await communicate.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            traceback.print_exc()
            return None
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    def speak(self, text):
        if VoiceEngine._mute or not text:
            return
        text = text.strip()
        if not text:
            return
        now = time.time()
        with VoiceEngine._shared_lock:
            last = VoiceEngine._last_said.get(text, 0)
            if now - last < self.repeat_gap:
                return
            VoiceEngine._last_said[text] = now
        try:
            VoiceEngine._shared_queue.put_nowait(text)
        except queue.Full:
            try:
                VoiceEngine._shared_queue.get_nowait()
                VoiceEngine._shared_queue.put_nowait(text)
            except Exception:
                pass
    @classmethod
    def set_mute(cls, mute):
        with cls._shared_lock:
            cls._mute = mute
    def _worker(self):
        while True:
            try:
                text = VoiceEngine._shared_queue.get(timeout=0.5)
                self._play_tts(text)
            except queue.Empty:
                continue
            except Exception:
                traceback.print_exc()
    def _play_tts(self, text):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            loop = asyncio.new_event_loop()
            try:
                async def _gen():
                    comm = edge_tts.Communicate(text=text, voice=self._get_voice())
                    await comm.save(tmp_path)
                loop.run_until_complete(_gen())
            finally:
                loop.close()
            self._play_file(tmp_path)
        except Exception:
            traceback.print_exc()
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    def _play_file(self, path):
        if _PYGAME_OK:
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                pygame.mixer.music.unload()
                return
            except Exception:
                pass
        try:
            from playsound import playsound
            playsound(path, block=True)
        except Exception:
            pass