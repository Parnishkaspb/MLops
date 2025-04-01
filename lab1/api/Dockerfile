FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /.

COPY req.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r req.txt

RUN pip install --extra-index-url https://pypi.clickhouse.tech \
    clickhouse-sqlalchemy \
    clickhouse-driver

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]