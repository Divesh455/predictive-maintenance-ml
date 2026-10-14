
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-docker.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-docker.txt

COPY app/ ./app/
COPY models/ ./models/
COPY reports/ ./reports/

EXPOSE 8000 8501
