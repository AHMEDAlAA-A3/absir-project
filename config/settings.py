import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
BASE_DIR = Path(__file__).parent.parent
CURRENCY_MODEL_PATH = os.getenv(
    "CURRENCY_MODEL_PATH",
    str(BASE_DIR / "models" / "best.pt")
)
OBJECTS_MODEL_PATH = os.getenv(
    "OBJECTS_MODEL_PATH",
    str(BASE_DIR / "models" / "yolov8l.pt")
)
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CURRENCY_CONFIDENCE = float(os.getenv("CURRENCY_CONFIDENCE", "0.5"))
OBJECT_CONFIDENCE = float(os.getenv("OBJECT_CONFIDENCE", "0.4"))
VOICE_LANGUAGE = os.getenv("VOICE_LANGUAGE", "ar")
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").lower() == "true"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DANGER_COOLDOWN = float(os.getenv("DANGER_COOLDOWN", "4.0"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    warnings.warn(
        "GROQ_API_KEY not set in .env — AI assistant will be disabled",
        RuntimeWarning,
        stacklevel=2,
    )