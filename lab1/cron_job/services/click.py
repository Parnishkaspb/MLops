import os
import clickhouse_connect

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", 8123))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "123456")

ch_client = clickhouse_connect.get_client(
    host=CLICKHOUSE_HOST,
    port=CLICKHOUSE_PORT,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASSWORD,
)


def insert_click_toxic(data):
    ch_client.insert("long_comments", [[data["text"]]], column_names=["text"])

def insert_handle_moderate(data):
    ch_client.insert("hand_moderate", [[data["text"]]], column_names=["text"])
