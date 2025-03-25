from backend.models.models import Comment
from backend.services.logger_config import logger
from backend.services.clickhouse import ch_client
from backend.schemas.comment import CommentCreate, CommentOut
from sqlalchemy.future import select


def store_comment(db, data: CommentCreate):
    comment = Comment(post_id=data.post_id, text=data.text)
    db.add(comment)
    db.commit()
    db.refresh(comment)
    return comment


def get_comments_by_post_id(post_id: int, db):
    result = db.execute(select(Comment).where(Comment.post_id == post_id).where(Comment.ban == False).order_by(Comment.created_at.desc()))
    return result.scalars().all()