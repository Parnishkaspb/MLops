from db import pg_pool
from pydantic import BaseModel

class PostCreate(BaseModel):
    text: str

class PostService:
    @staticmethod
    async def get_posts():
        async with pg_pool.acquire() as conn:
            posts = await conn.fetch("SELECT * FROM posts;")
        return {"posts": [dict(row) for row in posts]}

    @staticmethod
    async def get_post(post_id: int):
        async with pg_pool.acquire() as conn:
            post = await conn.fetchrow("SELECT * FROM posts WHERE id = $1", post_id)
        if post:
            return {"post": dict(post)}
        else:
            return {"error": "Пост не найден"}

    @staticmethod
    async def create_post(data: PostCreate):
        async with pg_pool.acquire() as conn:
            post = await conn.fetchrow(
                "INSERT INTO posts (text) VALUES ($1) RETURNING *;",
                data.text
            )
        return {"post": dict(post)}

