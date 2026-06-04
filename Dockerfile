# Use a lightweight Python image
FROM python:3.10-slim

# Task requirement: build argument with sensible default
ARG HF_MODEL_NAME=nagaananth/MLOPS_group-v2

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_MODEL=${HF_MODEL_NAME}

# Working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Default input (can be overridden at runtime)
ENV INPUT_TEXT="Default message test."

# Run inference
CMD ["python", "src/inference.py"]
