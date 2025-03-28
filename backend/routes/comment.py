from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.schemas.comment import CommentCreate, CommentOut
from backend.services.postgresql import get_db
from backend.cruds import comment_crud, click_crud

router = APIRouter(prefix="/comments", tags=["Comments"])

@router.post("/", response_model=CommentOut)
async def create_comment(data: CommentCreate, db: AsyncSession = Depends(get_db)):
    if len(data.text) >= 50:
        click_crud.insert_click(data)

    return await comment_crud.store_comment(db, data)


@router.get("/{post_id}", response_model=list[CommentOut])
async def get_comments_by_post_id(post_id: int, db: AsyncSession = Depends(get_db)):
    return await comment_crud.get_comments_by_post_id(db, post_id)