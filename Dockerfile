FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for model download
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

# Download FastText language detection model (125 MB)
# This model is not in git due to GitHub's 100 MB file limit
RUN mkdir -p models/language_detection && \
    echo "Downloading FastText model (125 MB)..." && \
    curl -L -o models/language_detection/lid.176.bin \
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin && \
    echo "FastText model downloaded successfully!"

# Copy application code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Set environment variables
ENV PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application using the startup script
CMD ["./start.sh"]
