FROM python:3.12.14-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv/app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 && rm -rf /var/lib/apt/lists/*
COPY inference_service ./inference_service
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    'fastapi>=0.116,<1' \
    'uvicorn[standard]>=0.35,<1' \
    'pydantic-settings>=2.10,<3' \
    'prometheus-client>=0.22,<1' \
    'sentence-transformers>=5,<6'
ARG INSTALL_OPTIMIZED_RERANK=0
RUN if [ "$INSTALL_OPTIMIZED_RERANK" = "1" ]; then \
      pip install --no-cache-dir 'sentence-transformers[onnx,openvino]>=5,<6'; \
    fi
RUN useradd --create-home --uid 10001 appuser && mkdir -p /models && chown -R appuser:appuser /models /srv/app
USER appuser
EXPOSE 8090
CMD ["uvicorn", "inference_service.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
