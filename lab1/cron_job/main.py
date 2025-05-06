import json
import time
import logging
import sys

import torch
from confluent_kafka import Consumer, KafkaException

from nn_model.model import MyModel, check_ur_comment, device
from services.click import insert_click_toxic, insert_handle_moderate
from services.postgre import store_comment

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
logger = logging.getLogger("cron_job")

logger.debug("import all completed")

model = MyModel(None, 768)
model.load_state_dict(torch.load('model_weights.pt',  map_location=device))

logger.debug("load model completed")

KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"
KAFKA_TOPIC = "fastapi_messages"

def create_kafka_consumer():
    conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS, 
        'group.id': 'my_consumer_group',
        'auto.offset.reset': 'earliest',
        'allow.auto.create.topics': True,
        'enable.auto.commit': False
    }
    return Consumer(conf)

def filter_messages(data):
    logger.info(f"process message. data: {data}")
    text = data.get('text')
    p = check_ur_comment(text, model)
    logger.info(f"got probability of toxic: {p}")
    if p > 0.75:
        insert_click_toxic(data=data)
        logger.info(f"insert to clickhouse msg")
    elif p<0.25:
        store_comment(data=data)
        logger.info("insert to postgre")
    else:
        insert_handle_moderate(data)
        logger.info("insert to another click")

def check_kafka_messages(consumer:Consumer, topic):
    # Подписываемся на топик
    consumer.subscribe([topic])

    # Опросить Kafka на наличие сообщений (без блокировки)
    messages = consumer.consume(timeout=1.0, num_messages=1)

    if not messages:
        return False
    else:
        # Если есть сообщения, обрабатываем их
        for message in messages:
            if message.error():
                raise KafkaException(message.error())
        
            try:
                msg_value = message.value().decode('utf-8')
                data = json.loads(msg_value)
                filter_messages(data)
                consumer.commit(message)
            except Exception as e:
                logger.error(f"Got error: {e}")
        return True

def main():
    consumer = create_kafka_consumer()
    logger.info("cron job started successfully")
    try:
        while True:
            has_messages = check_kafka_messages(consumer, KAFKA_TOPIC)
            
            if not has_messages:
                logger.warning("No messages, sleeping for 5 minutes")
            time.sleep(20)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        consumer.close()

if __name__ == '__main__':
    main()