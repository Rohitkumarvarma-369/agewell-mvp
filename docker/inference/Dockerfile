FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    opentelemetry-api==1.27.0 \
    opentelemetry-sdk==1.27.0 \
    pydantic==2.10.3 \
    pydantic-settings==2.7.0 \
    python-json-logger==2.0.7 \
    structlog==24.4.0 \
    "uvicorn[standard]==0.32.1"

EXPOSE 8000

CMD ["uvicorn", "agewell.service:app", "--host", "0.0.0.0", "--port", "8000"]
