from backend.services.clickhouse import ch_client
from backend.services.logger_config import logger


async def init_ch():
    try:
        ch_client.command("""
                CREATE TABLE IF NOT EXISTS long_comments (
                    text String
                ) ENGINE = MergeTree() ORDER BY tuple()
            """)
        logger.info("Таблица создана в ClickHouse")
    except Exception as e:
        logger.error("Ошибка при создании ClickHouse таблицы:", e)
