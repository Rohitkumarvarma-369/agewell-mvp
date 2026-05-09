# First Boot Runbook

This runbook verifies the Phase 0 foundation on a local machine or the future cloud VM.

## Prerequisites

- Docker 24+ and Docker Compose v2.
- `uv` installed.
- Network access for first-time Python package and container image pulls.
- The project checked out at `/home/rohit/kaggle-iisc/agewell-mvp`.

The host may have Python 3.12. The project pins Python `>=3.11,<3.12`; `uv` provisions
Python 3.11 from `.python-version`.

## Commands

```bash
cd /home/rohit/kaggle-iisc/agewell-mvp
make bootstrap
make up
make health
make test
```

## Expected Services

| Service | URL or port | Expected result |
|---|---:|---|
| Postgres + pgvector | `localhost:5532` | `SELECT 1` works |
| MinIO API | `http://localhost:9000/minio/health/ready` | `200 OK` |
| MinIO console | `http://localhost:9001` | login with `agewell` / `agewell-minio-pass` |
| MLflow | `http://localhost:5000` | UI loads |
| Prefect | `http://localhost:4200` | UI loads |
| Inference stub | `http://localhost:8000/health` | JSON status payload |

## Common Fixes

- If `uv sync` cannot find Python 3.11, run `uv python install 3.11`.
- Before starting Phase 1, run `uv sync --group dev --extra mvp` or
  `make bootstrap INCLUDE_MVP=1` to install the full ML/agent toolchain. The
  Phase 0 default install is intentionally minimal.
- If a port is already taken, change the host-side port in `docker-compose.yml`.
- If MinIO buckets are missing, run `./scripts/seed_minio.sh`.
- If Postgres init did not run, remove the Compose volume with `docker compose down -v`
  and run `make up` again.

## Exit Criteria

Phase 0 is green when `make lint`, `make test-unit`, `make up`, `make health`, and
`make test-integration` all pass.
