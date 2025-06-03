# Use a slim Python 3.9 base image (same in pipeline registry logs)
FROM python:3.11

# Create/app directory
WORKDIR /app

# Install system deps for any python packages that need building
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first
COPY requirements.txt ./
COPY pyproject.toml ./

# Install pip and Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire Kedro project
COPY . .

# Tell Python/Kedro where to find your src directory
ENV PYTHONPATH=/app/src

# Default command: run the Kedro pipeline
CMD ["kedro", "run"]
