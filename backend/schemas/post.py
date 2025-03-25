from pydantic import BaseModel
from datetime import datetime

class PostCreate(BaseModel):
    text: str

class PostOut(BaseModel):
    id: int
    text: str
    ban: bool
    created_at: datetime

    class Config:
        orm_mode = True
