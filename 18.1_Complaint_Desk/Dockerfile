# Use lightweight official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Set environment variables to prevent bytecode and buffer output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and evaluation files
COPY app.py .
COPY evaluation/ ./evaluation/
COPY .streamlit/secrets.toml.example ./.streamlit/secrets.toml.example

# Expose standard Streamlit port
EXPOSE 8501

# Health check to ensure container is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit server bound to 0.0.0.0
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
