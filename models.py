from pydantic import BaseModel
from typing import Optional, Any

class TelexEventData(BaseModel):
    text: str

class TelexEvent(BaseModel):
    type: str
    data: TelexEventData

class TelexResponseData(BaseModel):
    location: Optional[str]
    aqi: Optional[int]
    summary: str

class TelexResponse(BaseModel):
    type: str
    data: TelexResponseData
