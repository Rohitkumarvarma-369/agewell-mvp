.PHONY: bootstrap up down logs health test test-unit test-integration lint format clean \
        build_master train_teacher train_student calibrate evaluate demo baseline \
        agent serve dashboard

SHELL := /bin/bash

bootstrap:
	@./scripts/bootstrap.sh

up:
	docker compose up -d --build
	@./scripts/wait_for_services.sh postgres minio
	@./scripts/seed_minio.sh
	@./scripts/wait_for_services.sh all

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

health:
	@./scripts/wait_for_services.sh all --once

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit -q

test-integration:
	uv run pytest tests/integration -q -m integration

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/agewell

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage outputs

build_master:
	uv run python -m agewell.scripts.build_master

train_teacher:
	uv run python -m agewell.scripts.train_teacher

train_student:
	uv run python -m agewell.scripts.train_student

calibrate:
	uv run python -m agewell.scripts.calibrate

evaluate:
	uv run python -m agewell.scripts.evaluate

baseline: build_master train_teacher train_student calibrate evaluate

demo:
	uv run python -m agewell.scripts.demo

agent:
	uv run python -m agewell.agent.serve

serve:
	uv run uvicorn agewell.service:app --host 0.0.0.0 --port 8000

dashboard:
	uv run streamlit run src/agewell/streamlit/app.py
