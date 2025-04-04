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

COPY . .

CMD ["python", "main.py"]