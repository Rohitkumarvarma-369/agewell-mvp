FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src

RUN pip install --no-cache-dir fastapi nibabel numpy pydantic python-json-logger structlog uvicorn[standard] \
    opentelemetry-api opentelemetry-instrumentation-fastapi opentelemetry-sdk

COPY src /app/src

CMD ["uvicorn", "agewell.services.qc_svc.api:app", "--host", "0.0.0.0", "--port", "8000"]
