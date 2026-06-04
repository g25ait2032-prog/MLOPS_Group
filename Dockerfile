# Use an official lightweight Python runtime as a base image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies and clear cache to keep the image lightweight
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory containing inference.py into the container
COPY src/ ./src/

# Define default fallback environment variables (can be overridden at runtime)
ENV HF_MODEL="nagaananth/MLOPS_group-v2"
ENV INPUT_TEXT="Default message test."

# Run the inference script when the container launches
CMD ["python", "src/inference.py"]
