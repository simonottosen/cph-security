"""
Prometheus metrics server for the fetchtime service.

Exposes /metrics for Prometheus scraping. Runs as a long-lived Flask process
alongside the Ofelia-managed airport CLI jobs. Because those jobs run as
separate short-lived processes, prometheus_client multiprocess mode is used:
each process writes metrics to files in PROMETHEUS_MULTIPROC_DIR, and this
server aggregates them on every scrape.

Usage:
    python metrics_server.py        # listens on 0.0.0.0:9090
"""

import os
from flask import Flask, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client import multiprocess

app = Flask(__name__)


@app.route("/metrics")
def metrics():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    data = generate_latest(registry)
    return Response(data, mimetype=CONTENT_TYPE_LATEST)


@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("METRICS_PORT", 9090))
    app.run(host="0.0.0.0", port=port)
