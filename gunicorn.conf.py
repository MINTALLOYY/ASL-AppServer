"""Gunicorn configuration for Render deployment.

Uses eventlet async worker for WebSocket support.
"""

import os

# Use eventlet async worker — better WebSocket compatibility with flask-sock
worker_class = "eventlet"

# Single worker to stay within Render free-tier memory limits
workers = 1

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
