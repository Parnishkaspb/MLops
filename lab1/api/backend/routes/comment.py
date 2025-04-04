import json
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.cruds import comment_crud
from backend.schemas.comment import CommentCreate, CommentOut
from backend.services.kafka import produce_message
from backend.services.logger_config import logger
from backend.services.postgresql import get_db

router = APIRouter(prefix="/comments", tags=["Comments"])


@router.post("/", response_model=dict)
async def create_comment(
    data: CommentCreate,
):
    try:
        # Сериализуем данные в JSON
        produce_message(data)
        
        logger.info(f'Message sent to Kafka: {data}')
        return {"status": "success", "message": "Message sent to Kafka"}
    except Exception as e:
        logger.error(f'Got error during sending message to Kafka: {e}')
        return {"status": "error", "message": str(e)}


@router.get("/{post_id}", response_model=list[CommentOut])
async def get_comments_by_post_id(
    post_id: int,
    db: AsyncSession = Depends(get_db),
):
    return await comment_crud.get_comments_by_post_id(db, post_id)
