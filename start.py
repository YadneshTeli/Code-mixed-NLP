#!/usr/bin/env python3
"""
Startup script for Railway deployment
Properly handles PORT environment variable
"""
import os
import sys
import subprocess

# Get PORT from environment, default to 8000
port = os.environ.get('PORT', '8000')

print("=" * 50)
print("🚀 Starting Code-mixed NLP API")
print(f"📍 Port: {port}")
print(f"🐍 Python: {sys.version.split()[0]}")
print("=" * 50)
print()

# Start uvicorn
cmd = [
    'uvicorn',
    'app.main:app',
    '--host', '0.0.0.0',
    '--port', port,
    '--log-level', 'info'
]

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\n👋 Shutting down gracefully...")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error starting server: {e}")
    sys.exit(1)
