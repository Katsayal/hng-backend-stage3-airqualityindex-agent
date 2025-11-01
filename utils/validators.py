from typing import Optional
from pydantic import BaseModel
import re

class TelexEventData(BaseModel):
    location: Optional[str] = None
    aqi: Optional[int] = None
    summary: Optional[str] = None

class TelexEvent(BaseModel):
    type: str
    data: TelexEventData

def validate_location(location: Optional[str]) -> Optional[str]:
    if not location or not location.strip():
        return None
    location = location.strip()
    # Remove common trailing words (like "right now", "today")
    location = re.sub(r"\b(right now|today|currently|please|now)\b", "", location, flags=re.I)
    normalized = re.sub(r"[^a-zA-Z\s,]", "", location).strip()
    return normalized if normalized else None

def validate_aqi(aqi: Optional[int]) -> Optional[int]:
    if aqi is None:
        return None
    if not isinstance(aqi, int):
        try:
            aqi = int(aqi)
        except (ValueError, TypeError):
            return None
    return aqi if 0 <= aqi <= 500 else None

def validate_summary(summary: Optional[str]) -> str:
    if not summary or not summary.strip():
        return "Sorry, I couldn't generate a summary."
    return summary.strip()

def build_telex_response(location: Optional[str], aqi: Optional[int], summary: Optional[str]) -> dict:
    location = validate_location(location)
    aqi = validate_aqi(aqi)
    summary = validate_summary(summary)
    return {
        "type": "message",
        "data": {
            "location": location,
            "aqi": aqi,
            "summary": summary
        }
    }
