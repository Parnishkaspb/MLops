from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from backend.models.models import Post, Comment
from backend.schemas.post import PostCreate, PostOut, PostWithComments
from backend.deps import get_db

router = APIRouter(prefix="/posts", tags=["Posts"])

@router.post("/", response_model=PostOut)
async def create_post(data: PostCreate, db: AsyncSession = Depends(get_db)):
    post = Post(text=data.text)
    db.add(post)
    await db.commit()
    await db.refresh(post)
    return post

@router.get("/", response_model=list[PostWithComments])
async def get_posts(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Post).options(selectinload(Post.comments))
    )
    posts = result.scalars().all()

    for post in posts:
        post.comments = [c for c in post.comments if not c.ban]

    return posts


@router.get("/{post_id}", response_model=PostWithComments)
async def get_post_with_comments(post_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Post).where(Post.id == post_id).options(
            selectinload(Post.comments.and_(Comment.ban == False))
        )
    )
    post = result.scalars().first()
    if not post:
        raise HTTPException(status_code=404, detail="Пост не найден")
    return post