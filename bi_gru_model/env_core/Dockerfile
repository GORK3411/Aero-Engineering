FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    libgeos-dev \
    libproj-dev \
    proj-data \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY env_core/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .