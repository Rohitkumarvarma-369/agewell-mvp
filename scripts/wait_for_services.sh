#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-all}"
MODE="${2:-wait}"

check_http() {
  local name="$1"
  local url="$2"
  curl -fs "$url" >/dev/null
}

check_postgres() {
  docker compose exec -T postgres pg_isready -U agewell -d agewell >/dev/null
}

check_one() {
  local svc="$1"
  case "$svc" in
    postgres) check_postgres ;;
    minio) check_http minio "http://localhost:9000/minio/health/ready" ;;
    mlflow) check_http mlflow "http://localhost:5000/health" ;;
    prefect) check_http prefect "http://localhost:4200/api/health" ;;
    inference) check_http inference "http://localhost:8000/health" ;;
    redis) docker compose exec -T redis redis-cli ping | grep -q PONG ;;
    *) echo "unknown service: $svc" >&2; return 2 ;;
  esac
}

service_list() {
  case "$TARGET" in
    all) echo "postgres minio mlflow prefect inference redis" ;;
    *) echo "$TARGET" ;;
  esac
}

for svc in $(service_list); do
  if [[ "$MODE" == "--once" ]]; then
    if check_one "$svc"; then
      echo "$svc: ok"
    else
      echo "$svc: DOWN"
      exit 1
    fi
    continue
  fi

  printf "waiting for %s" "$svc"
  for _ in $(seq 1 60); do
    if check_one "$svc"; then
      printf " ok\n"
      break
    fi
    printf "."
    sleep 2
  done
  check_one "$svc"
done
