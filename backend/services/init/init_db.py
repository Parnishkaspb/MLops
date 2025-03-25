from backend.services.logger_config import logger


async def init_db(pg_pool):
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
        logger.info("Таблицы созданы в PostgreSQL")
