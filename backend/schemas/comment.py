from datetime import datetime

from pydantic import BaseModel


class CommentCreate(BaseModel):
    post_id: int
    text: str


class CommentOut(BaseModel):
    id: int
    post_id: int
    text: str
    ban: bool
    created_at: datetime

    class Config:
        orm_mode = True


class CommentAdminOut(BaseModel):
    text: str
