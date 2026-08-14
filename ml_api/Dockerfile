# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /project

COPY requirements.txt /project/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /project/requirements.txt

COPY project/app.py /project/app.py
COPY project/model.py /project/model.py
COPY project/features.py /project/features.py
COPY project/data.py /project/data.py
COPY project/train.py /project/train.py

# Serve under gunicorn (the Flask dev server is not for production). The trainer
# runs the same image but overrides the command with: python train.py
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
