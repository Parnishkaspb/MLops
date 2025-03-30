from backend.services.clickhouse import ch_client
from backend.services.logger_config import logger
from backend.schemas.comment import CommentCreate


def insert_click(data: CommentCreate):
    try:
        ch_client.insert("long_comments", [[data.text]], column_names=["text"])
        logger.info("Добавлено в ClickHouse")
    except Exception as e:
        logger.error("Ошибка:", type(e), "-", e)
