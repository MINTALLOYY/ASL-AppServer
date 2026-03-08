"""Gunicorn configuration for Render deployment.

Forces gevent worker so long-lived WebSocket connections
are not killed by the default sync worker timeout.
"""

import os

# Use gevent async worker — required for WebSocket support
worker_class = "gevent"

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
