from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.cruds import (
    # click_crud,
    comment_crud,
)
from backend.schemas.comment import CommentCreate, CommentOut
from backend.services.kafka import producer, KAFKA_TOPIC
from backend.services.logger_config import logger
from backend.services.postgresql import get_db

router = APIRouter(prefix="/comments", tags=["Comments"])


@router.post("/", response_model=CommentOut)
async def create_comment(
    data: CommentCreate,
    db: AsyncSession = Depends(get_db),
):
    try:
        producer.send(KAFKA_TOPIC, value=data)
        producer.flush()
        logger.info(f'Message sent to Kafka: {data}')
        return {"status": "success", "message": "Message sent to Kafka"}
    except Exception as e:
        logger.error(f'Got error during sending message to Kafka: {e}')
        return {"status": "error", "message": str(e)}
    # if len(data.text) >= 50:
    #     click_crud.insert_click(data)

    # return await comment_crud.store_comment(db, data)


@router.get("/{post_id}", response_model=list[CommentOut])
async def get_comments_by_post_id(
    post_id: int,
    db: AsyncSession = Depends(get_db),
):
    return await comment_crud.get_comments_by_post_id(db, post_id)
