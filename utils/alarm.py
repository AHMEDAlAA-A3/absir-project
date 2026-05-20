import time
import threading
import os
import numpy as np
class AlarmSystem:
    ALARM_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "alarm.wav")
    def __init__(self, cooldown=3.0):
        self.cooldown = cooldown
        self._last_alarm = 0.0
        self._lock = threading.Lock()
        self._pygame_ok = False
        self._init_audio()
    def _init_audio(self):
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._pygame_ok = True
            self._beep_sound = self._make_beep()
            alarm_path = os.path.abspath(self.ALARM_PATH)
            self._custom_sound = pygame.mixer.Sound(alarm_path) if os.path.exists(alarm_path) else None
        except Exception:
            self._pygame_ok = False
    def _make_beep(self, freq=880, duration=0.4, volume=0.8):
        import pygame
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave = (np.sin(2 * np.pi * freq * t) * 0.6 +
                np.sin(2 * np.pi * freq * 1.5 * t) * 0.3 +
                np.sin(2 * np.pi * freq * 2 * t) * 0.1)
        envelope = np.minimum(t / 0.01, 1.0) * np.minimum((duration - t) / 0.05, 1.0)
        wave = (wave * envelope * volume * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(wave)
    def trigger(self, severity="danger"):
        if severity == "safe":
            return
        now = time.time()
        with self._lock:
            if now - self._last_alarm < self.cooldown:
                return
            self._last_alarm = now
        threading.Thread(target=self._play, daemon=True).start()
    def _play(self):
        try:
            if self._pygame_ok:
                sound = self._custom_sound or self._beep_sound
                sound.play()
                import pygame
                pygame.time.wait(600)
            else:
                self._fallback_beep()
        except Exception as e:
            print(f"[Alarm] {e}")
    @staticmethod
    def _fallback_beep():
        try:
            import winsound
            winsound.Beep(880, 400)
        except Exception:
            print("\a", end="", flush=True)