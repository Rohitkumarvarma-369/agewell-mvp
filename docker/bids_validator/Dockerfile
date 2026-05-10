FROM node:22-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g bids-validator
RUN pip3 install --no-cache-dir fastapi pydantic python-json-logger structlog uvicorn[standard] \
    opentelemetry-api opentelemetry-instrumentation-fastapi opentelemetry-sdk

COPY src /app/src

CMD ["uvicorn", "agewell.services.bids_validator_svc.api:app", "--host", "0.0.0.0", "--port", "8000"]
