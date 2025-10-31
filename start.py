#!/usr/bin/env python3
"""
Startup script for Railway deployment
Properly handles PORT environment variable
"""
import os
import sys
import subprocess
import time

# Get PORT from environment, default to 8000
port_env = os.environ.get('PORT', '8000')

# Clean up port value - remove any shell variable syntax
if port_env.startswith('$'):
    print(f"⚠️  WARNING: PORT contains shell variable syntax: {port_env}")
    port = '8000'  # Fallback to default
else:
    port = port_env

# Validate port is numeric
try:
    port_num = int(port)
    if not (1 <= port_num <= 65535):
        raise ValueError(f"Port {port_num} out of valid range")
except ValueError as e:
    print(f"⚠️  Invalid PORT value '{port}': {e}")
    print("   Using default port 8000")
    port = '8000'

print("=" * 50)
print("🚀 Starting Code-mixed NLP API")
print(f"📍 Port: {port}")
print(f"🐍 Python: {sys.version.split()[0]}")
print(f"🔍 PORT env: {repr(os.environ.get('PORT', 'NOT_SET'))}")
print("=" * 50)
print()

# Small delay to ensure container is ready
print("⏳ Waiting 2 seconds for container to be ready...")
time.sleep(2)

# Start uvicorn
cmd = [
    'uvicorn',
    'app.main:app',
    '--host', '0.0.0.0',
    '--port', port,
    '--log-level', 'info'
]

print(f"🎯 Running command: {' '.join(cmd)}")
print()

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\n👋 Shutting down gracefully...")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error starting server: {e}")
    sys.exit(1)
