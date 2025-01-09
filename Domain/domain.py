from pydantic import BaseModel
from typing import Optional
from typing import Optional
from fastapi import UploadFile
class MemeRequest(BaseModel):
    #text: str  
    text: Optional[str] = None
    image: Optional[UploadFile] = None
class MemeResponse(BaseModel):
    image_url: str  
    mime_type: Optional[str] = "image/png"

class TemplateMemeResponse(BaseModel):    
    image_url: str  
    mime_type: Optional[str] = "image/png"  
    similarity: float
    