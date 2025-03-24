from fastapi import FastAPI
import asyncpg
import clickhouse_connect
import os
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import BaseModel

app = FastAPI()

pg_pool = None
ch_client = None


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


@app.get("/comments")
async def get_comments():
    async with pg_pool.acquire() as conn:
        comments = await conn.fetch("SELECT * FROM comments;")
    return {"comments": [dict(row) for row in comments]}


@app.get("/posts")
async def get_posts():
    async with pg_pool.acquire() as conn:
        posts = await conn.fetch("SELECT * FROM posts;")
    return {"posts": [dict(row) for row in posts]}

@app.get("/posts/{post_id}")
async def get_posts():
    async with pg_pool.acquire() as conn:
        posts = await conn.fetch("SELECT * FROM posts WHERE id ;")
    return {"posts": [dict(row) for row in posts]}

class PostCreate(BaseModel):
    text: str

@app.post("/posts")
async def create_post(data: PostCreate):
    async with pg_pool.acquire() as conn:
        post = await conn.fetchrow(
            "INSERT INTO posts (text) VALUES ($1) RETURNING *;",
            data.text
        )
    return {"post": dict(post)}