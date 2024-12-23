from pydantic import BaseModel
from typing import Optional

class MemeRequest(BaseModel):
    text: str  

class MemeResponse(BaseModel):
    image_url: str  
    mime_type: Optional[str] = "image/png"