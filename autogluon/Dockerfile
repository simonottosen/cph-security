# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /project

RUN pip install --no-cache-dir pipenv

# Install from the committed lockfile so CI doesn't resolve dependencies on every build.
COPY Pipfile Pipfile.lock /project/
RUN --mount=type=cache,target=/root/.cache/pip \
    pipenv install --system --deploy --ignore-pipfile

# The default Linux torch wheel now expects CUDA user-space libraries.
# Replace it with the CPU wheel so Chronos imports cleanly in a slim image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.1 torchvision==0.24.1

COPY project/app.py /project/app.py

CMD ["python", "/project/app.py", "-h", "0.0.0.0"]
