import os
import asyncpg
from tenacity import retry, stop_after_attempt, wait_fixed
from .logger_config import logger

pg_pool = None

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
    logger.info("Connected to Postgres DB")
    return pg_pool