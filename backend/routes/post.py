from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from backend.models.models import Post
from backend.schemas.post import PostCreate, PostOut
from backend.deps import get_db

router = APIRouter(prefix="/posts", tags=["Posts"])

@router.post("/", response_model=PostOut)
async def create_post(data: PostCreate, db: AsyncSession = Depends(get_db)):
    post = Post(text=data.text)
    db.add(post)
    await db.commit()
    await db.refresh(post)
    return post

@router.get("/", response_model=list[PostOut])
async def get_posts(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Post).order_by(Post.created_at.desc()))
    return result.scalars().all()

@router.get("/{post_id}", response_model=PostOut)
async def get_post(post_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()
    if post is None:
        raise HTTPException(status_code=404, detail="Пост не найден")
    return post