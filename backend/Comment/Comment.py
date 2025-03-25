from datetime import datetime
from sqlalchemy import Column, ForeignKey, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import relationship
from src.backend.database import Base


class Post(Base):
    __tablename__ = 'posts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    ban = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")


class Comment(Base):
    __tablename__ = 'comments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(Integer, ForeignKey('posts.id'), nullable=False)
    text = Column(Text, nullable=False)
    ban = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    post = relationship("Post", back_populates="comments")

#
# class CommentCreate(BaseModel):
#     post_id: int
#     text: str
#
# class CommentService:
#     @staticmethod
#     async def get_comments():
#         async with pg_pool.acquire() as conn:
#             comments = await conn.fetch("SELECT * FROM comments;")
#         return {"comments": [dict(row) for row in comments]}
#
#     @staticmethod
#     async def get_comments_by_post_ID(post_id: int):
#         async with pg_pool.acquire() as conn:
#             comments = await conn.fetchrow("SELECT * FROM comments WHERE post_id=$1 AND ban = FALSE", post_id)
#         if comments:
#             return {"comments": dict(comments)}
#         else:
#             return {"error": "Комментарии не найден"}
#
#     @staticmethod
#     async def create_comment(data: CommentCreate):
#         async with pg_pool.acquire() as conn:
#             comments = await conn.fetchrow(
#                 "INSERT INTO comments (post_id, text) VALUES ($1, $2) RETURNING *;",
#                 [data.post_id, data.text]
#             )
#         return {"comments": dict(comments)}