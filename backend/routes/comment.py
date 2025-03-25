from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.models.models import Comment
from backend.schemas.comment import CommentCreate, CommentOut
from backend.deps import get_db

router = APIRouter(prefix="/comments", tags=["Comments"])

@router.post("/", response_model=CommentOut)
async def create_post(data: CommentCreate, db: AsyncSession = Depends(get_db)):
    post = Comment(post_id=data.post_id,text=data.text)
    db.add(post)
    await db.commit()
    await db.refresh(post)
    return post

@router.get("/", response_model=list[CommentOut])
async def get_posts(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Comment).where('post_id', post_id).order_by(Comment.created_at.desc()))
    return result.scalars().all()