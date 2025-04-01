import json
from confluent_kafka import Producer, KafkaException

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "fastapi_messages"

# Конфигурация продюсера
conf = {
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'message.send.max.retries': 5,
    'retry.backoff.ms': 1000,  # 1 секунда между попытками
    'default.topic.config': {
        'acks': 'all'  # Ждем подтверждения от всех реплик
    }
}

producer = Producer(conf)

def delivery_report(err, msg):
    """Callback-функция для обработки результатов доставки"""
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def produce_message(value):
    """Отправка сообщения в Kafka"""
    try:
        # Сериализуем данные в JSON
        json_value = json.dumps(value).encode('utf-8')
        
        # Асинхронная отправка с callback
        producer.produce(
            topic=KAFKA_TOPIC,
            value=json_value,
            callback=delivery_report
        )
        
        # Ожидаем доставки сообщений (опционально)
        producer.flush()
        
    except KafkaException as e:
        print(f'Kafka producer error: {e}')
    except Exception as e:
        print(f'General error: {e}')