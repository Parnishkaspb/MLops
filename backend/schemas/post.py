from pydantic import BaseModel
from datetime import datetime
from typing import List
from .comment import CommentOut


class PostCreate(BaseModel):
    text: str

class PostOut(BaseModel):
    id: int
    text: str
    ban: bool
    created_at: datetime

    class Config:
        orm_mode = True

class PostWithComments(BaseModel):
    id: int
    text: str
    created_at: datetime
    comments: List[CommentOut]

    class Config:
        orm_mode = True