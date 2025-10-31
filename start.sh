#!/bin/bash

# Get PORT from environment or default to 8000
PORT=${PORT:-8000}

echo "=========================================="
echo "Starting Code-mixed NLP API"
echo "Port: $PORT"
echo "Python: $(python --version)"
echo "=========================================="

# Start uvicorn with proper error handling
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info
