from fastapi import FastAPI, APIRouter
import asyncpg
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from tenacity import retry, stop_after_attempt, wait_fixed
from backend.routes import post, comment
from backend.services.clickhouse import ch_client
app = FastAPI()

pg_pool = None
# ch_client = None

app.include_router(post.router)
app.include_router(comment.router)


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

@app.on_event("startup")
async def startup():
    await connect_pg()

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

    try:
        ch_client.command("""
                CREATE TABLE IF NOT EXISTS long_comments (
                    text String
                ) ENGINE = MergeTree() ORDER BY tuple()
            """)
        print("✅ ClickHouse table ready")
    except Exception as e:
        print("Ошибка при создании ClickHouse таблицы:", e)

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
