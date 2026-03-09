"""Gunicorn configuration for Render deployment.

Uses threaded workers for Flask-Sock WebSocket support.
Avoids eventlet/gevent monkey-patching issues with --preload.
"""

import os

# Use threaded worker for websocket connections
worker_class = "gthread"

# Threads per worker
threads = int(os.environ.get("WEB_THREADS", 8))

# Single worker to stay within Render free-tier memory limits
workers = int(os.environ.get("WEB_CONCURRENCY", 1))

# Disable worker timeout so WebSocket connections aren't killed
timeout = 0

# Graceful shutdown window 
graceful_timeout = 30

# Bind to Render's PORT
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
