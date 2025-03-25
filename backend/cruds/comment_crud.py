from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.models.models import Comment
from backend.deps import get_db
from backend.schemas.comment import CommentCreate


async def store_comment(db: AsyncSession, data: CommentCreate) -> Comment:
    comment = Comment(post_id=data.post_id, text=data.text)
    db.add(comment)
    await db.commit()
    await db.refresh(comment)
    return comment


def get_comments_by_post_id(post_id: int, db):
    result = db.execute(select(Comment).where(Comment.post_id == post_id).where(Comment.ban == False).order_by(Comment.created_at.desc()))
    return result.scalars().all()