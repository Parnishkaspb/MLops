from backend.services.clickhouse import ch_client

async def init_ch():
    try:
        ch_client.command("""
                CREATE TABLE IF NOT EXISTS long_comments (
                    text String
                ) ENGINE = MergeTree() ORDER BY tuple()
            """)
        print("✅ ClickHouse table ready")
    except Exception as e:
        print("Ошибка при создании ClickHouse таблицы:", e)