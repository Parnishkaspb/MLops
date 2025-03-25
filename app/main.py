from fastapi import FastAPI, APIRouter
import asyncpg
import clickhouse_connect
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from tenacity import retry, stop_after_attempt, wait_fixed
# from backend.Post import PostService, PostCreate
# from backend.Comment.Comment import CommentService
from backend.routes import post

app = FastAPI()

pg_pool = None
ch_client = None

app.include_router(post.router)


@retry(stop=stop_after_attempt(10), wait=wait_fixed(2))
async def connect_pg():
    global pg_pool
    pg_pool = await asyncpg.create_pool(
        user="user",
        password="password",
        database="mydb",
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
    )
    print("✅ Connected to PostgreSQL")

def connect_clickhouse():
    global ch_client
    ch_client = clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
        port=int(os.getenv("CLICKHOUSE_PORT", 8123)),
        username="default",
        password=os.getenv("CLICKHOUSE_PASSWORD", "")
    )
    print("✅ Connected to ClickHouse")



@app.on_event("startup")
async def startup():
    await connect_pg()
    connect_clickhouse()

    async with pg_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                ban BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id SERIAL PRIMARY KEY,
                post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                ban BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

@app.on_event("shutdown")
async def shutdown():
    await pg_pool.close()

@app.get("/")
async def root():
    return {"message": "FastAPI работает!"}

@app.get("/pg-version")
async def get_pg_version():
    async with pg_pool.acquire() as conn:
        version = await conn.fetchval("SELECT version();")
        return {"PostgreSQL version": version}

@app.get("/ch-version")
async def get_ch_version():
    version = ch_client.query('SELECT version()').result_rows[0][0]
    return {"ClickHouse version": version}

#
#
#
# posts = APIRouter(prefix="/post", tags=["posts"])
# @posts.get("/")
# async def get_posts():
#     return await PostService.get_posts()
#
# @posts.get("/{post_id}")
# async def get_post(post_id: int):
#     return await PostService.get_post(post_id)
#
# @posts.post("/")
# async def create_post(data: PostCreate):
#     return await PostService.create_post(data)
#
#
# comments = APIRouter(prefix="/comments", tags=["comments"])
# @app.get("/admin/comments")
# async def get_admin_comments():
#     return await CommentService.get_comments()
#
# @comments.get("/")
# async def get_comments_by_post_ID(post_id: int):
#     return await CommentService.get_comments_by_post_ID()



