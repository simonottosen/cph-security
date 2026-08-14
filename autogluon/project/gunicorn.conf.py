"""Gunicorn config for the Chronos-2 forecast service.

One worker only: Chronos-2 is memory-heavy and the retrain scheduler must be a
singleton (extra workers would each load the model and run their own scheduler).
Threads serve concurrent read-only /forecast requests while a background thread
runs the periodic retrain.
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
# Do not preload: the scheduler and model must live in the worker process.
preload_app = False


def post_worker_init(worker):
    import app

    app._initialize()
