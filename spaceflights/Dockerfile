FROM python:3.9-slim

WORKDIR /app

# Install build tools (gcc, make) and HTTPS certificates to compile packages like xgboost and allow secure downloads
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY pyproject.toml ./

# Upgrade pip and install Python packages without caching, makes sure latest pip is used and avoids saving extra temp files in image
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["kedro", "run"]

