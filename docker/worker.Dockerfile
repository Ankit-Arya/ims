FROM python:3.12.14-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tesseract-ocr libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN printf '#!/bin/sh\nOMP_THREAD_LIMIT="${TESSERACT_OMP_THREAD_LIMIT:-1}" exec /usr/bin/tesseract "$@"\n' > /usr/local/bin/tesseract \
    && chmod +x /usr/local/bin/tesseract
COPY pyproject.toml alembic.ini ./
COPY migrations ./migrations
COPY src ./src
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir '.[ingest]'
RUN useradd --create-home --uid 10001 appuser && mkdir -p /data /models && chown -R appuser:appuser /data /models /srv/app
USER appuser
