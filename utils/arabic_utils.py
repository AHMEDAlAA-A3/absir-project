"""
utils/arabic_utils.py
Fix summary:
  - arabic_reshaper ONLY  →  correct (connects letters, font handles RTL)
  - get_display           →  WRONG  (reverses the string, causes mirroring)
  - anchor="ra"           →  right-align so text doesn't spill left of box
"""
try:
    import arabic_reshaper
    ARABIC_OK = True
except Exception:
    ARABIC_OK = False
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False
import cv2
import numpy as np
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]
_font_cache: dict = {}
def _get_font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_CANDIDATES:
        try:
            f = ImageFont.truetype(path, size)
            _font_cache[size] = f
            return f
        except Exception:
            pass
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f
def prepare_arabic(text: str) -> str:
    """
    reshape ONLY — connects Arabic letters correctly.
    Do NOT use get_display: it reverses the string and causes mirroring.
    PIL + a proper font (DejaVu / Arial) handles RTL direction automatically.
    """
    if not ARABIC_OK:
        return text
    try:
        return arabic_reshaper.reshape(text)
    except Exception:
        return text
def put_arabic_text(
    img: np.ndarray,
    text: str,
    position: tuple,
    font_size: int = 22,
    color=(255, 255, 255),
    align: str = "right",
) -> np.ndarray:
    """
    Draw Arabic text on an OpenCV frame.
    position = (x, y)
      align="right"  →  x is the RIGHT edge  (recommended for Arabic)
      align="left"   →  x is the LEFT  edge
    """
    prepared = prepare_arabic(text)
    if not PIL_OK:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
        return img
    try:
        pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = _get_font(font_size)
        fill = (color[2], color[1], color[0])   
        x, y = position
        anchor = "ra" if align == "right" else "la"
        draw.text((x, y), prepared, font=font, fill=fill, anchor=anchor)
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    return img
def measure_arabic_text(text: str, font_size: int = 22) -> tuple[int, int]:
    """Returns (width, height) in pixels."""
    if not PIL_OK:
        return (len(text) * font_size // 2, font_size)
    try:
        prepared = prepare_arabic(text)
        font     = _get_font(font_size)
        dummy    = Image.new("RGB", (1, 1))
        draw     = ImageDraw.Draw(dummy)
        bbox     = draw.textbbox((0, 0), prepared, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        return (len(text) * font_size // 2, font_size)