#!/usr/bin/env bash
set -euo pipefail

docker compose exec -T postgres psql -U agewell -d agewell -c \
  "SELECT datname FROM pg_database WHERE datname IN ('agewell','mlflow','prefect','agentos') ORDER BY datname;"
