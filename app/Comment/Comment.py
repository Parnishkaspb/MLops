from db import pg_pool

class CommentService:
    @staticmethod
    async def get_comments():
        async with pg_pool.acquire() as conn:
            comments = await conn.fetch("SELECT * FROM comments;")
        return {"comments": [dict(row) for row in comments]}

    @staticmethod
    async def get_comments_by_post_ID(post_id: int):
        async with pg_pool.acquire() as conn:
            comments = await conn.fetchrow("SELECT * FROM comments WHERE post_id=$1 AND ban = FALSE", post_id)
        if comments:
            return {"comments": dict(comments)}
        else:
            return {"error": "Комментарии не найден"}
