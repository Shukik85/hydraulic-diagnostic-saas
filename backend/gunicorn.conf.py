"""Gunicorn configuration for production deployment."""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = int(os.environ.get("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"  # Use "gevent" for async workloads
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True

# Timeouts
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 30))
keepalive = 2
graceful_timeout = 30

# Logging
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")
accesslog = "-"  # stdout
errorlog = "-"   # stderr
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s '
    '"%(f)s" "%(a)s" %(D)s %(L)s'
)

# Process naming
proc_name = "hydraulic-diagnostic-gunicorn"

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = os.environ.get("SSL_KEYFILE")
certfile = os.environ.get("SSL_CERTFILE")

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Application-specific settings
raw_env = [
    "DJANGO_SETTINGS_MODULE=core.settings",
]

# Hooks
def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Gunicorn server is ready. Listening on: %s", server.address)

def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    worker.log.info("Worker received SIGINT or SIGQUIT. Shutting down...")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def worker_abort(worker):
    """Called when a worker receives the SIGTERM signal."""
    worker.log.info("Worker aborted (pid: %s)", worker.pid)