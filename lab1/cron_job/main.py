import json
import time

import torch
from confluent_kafka import Consumer, KafkaException

from nn_model.model import MyModel, check_ur_comment, device

model = MyModel(None, 768)
model.load_state_dict(torch.load('model_weights.pt',  map_location=device))
print(check_ur_comment("ебланище", model))

KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"
KAFKA_TOPIC = "fastapi_messages"

def create_kafka_consumer():
    conf = {
        'bootstrap.servers': 'kafka:9092', 
        'group.id': 'my_consumer_group',
        'auto.offset.reset': 'earliest',
        'allow.auto.create.topics': True,
        'enable.auto.commit': False
    }
    return Consumer(conf)

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
                text = data.get('text')
                print(f"process message. Text: {text}")
                consumer.commit(message)
            except Exception as e:
                print(f"Got error: {e}")
        return True

def main():
    consumer = create_kafka_consumer()

    try:
        while True:
            has_messages = check_kafka_messages(consumer, KAFKA_TOPIC)
            
            if not has_messages:
                print("No messages, sleeping for 5 minutes")
            time.sleep(300)
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        consumer.close()

if __name__ == '__main__':
    main()