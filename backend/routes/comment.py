from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from backend.models.models import Comment
from backend.schemas.comment import CommentCreate, CommentOut
from backend.deps import get_db
from backend.services.clickhouse import ch_client

router = APIRouter(prefix="/comments", tags=["Comments"])

@router.post("/", response_model=CommentOut)
async def create_comment(data: CommentCreate, db: AsyncSession = Depends(get_db)):
    post = Comment(post_id=data.post_id,text=data.text)
    if len(data.text) >= 50:
        try:
            ch_client.insert("long_comments", [[data.text]], column_names=["text"])
            print("✅ Добавлено в ClickHouse")
        except Exception as e:
            print("Ошибка:", type(e), "-", e)

    db.add(post)
    await db.commit()
    await db.refresh(post)
    return post

@router.get("/{post_id}", response_model=list[CommentOut])
async def get_comments_by_post_id(post_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Comment).where(Comment.post_id == post_id).where(Comment.ban == False).order_by(Comment.created_at.desc()))
    return result.scalars().all()