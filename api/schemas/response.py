from pydantic import BaseModel
from typing  import Optional, List, Any
class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
class Detection(BaseModel):
    name_en:    str
    name_ar:    str
    confidence: float
    bbox:       Optional[BBox] = None
class DangerInfo(BaseModel):
    name_en:    str
    name_ar:    str
    level:      str
    confidence: float
    size_ratio: float
    bbox:       BBox
class ABSIRResponse(BaseModel):
    """Unified response for ALL endpoints — REST and WebSocket."""
    status:     str                        
    mode:       str                        
    input_type: str                        
    message:    Optional[str]   = None     
    audio_b64:  Optional[str]   = None     
    danger:     Optional[DangerInfo] = None
    detections: List[Detection] = []
    extra:      Optional[Any]   = None
    image_b64:  Optional[str]   = None