.PHONY: bootstrap up down logs health phase2-up phase2-health test test-unit test-integration lint format clean \
        build_master split eda data-gates dvc-init dvc-pin phase1 \
        phase2-smoke phase2-dvc-pin \
        train_teacher train_student calibrate evaluate demo baseline \
        agent serve dashboard

SHELL := /bin/bash
INCLUDE_MVP ?= 0

bootstrap:
	@INCLUDE_MVP=$(INCLUDE_MVP) ./scripts/bootstrap.sh

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

phase2-up:
	docker compose --profile imaging up -d --build
	@./scripts/wait_for_services.sh all
	@./scripts/wait_for_services.sh imaging

phase2-health:
	@./scripts/wait_for_services.sh all --once
	@./scripts/wait_for_services.sh imaging --once

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

split:
	uv run python -m agewell.scripts.split

eda:
	uv run python -m agewell.scripts.eda

data-gates:
	uv run python -m agewell.scripts.data_gates

dvc-init:
	@test -d .dvc || uv run dvc init

dvc-pin: dvc-init
	uv run dvc add data/master.parquet data/splits/train.parquet data/splits/calib.parquet data/splits/test.parquet

phase1: build_master split eda data-gates dvc-pin

phase2-smoke:
	uv run python -m agewell.scripts.run_imaging --cohort ADNI_NIFTI --limit 1

phase2-dvc-pin: dvc-init
	uv run dvc add models/brainiac-v1-simclr.pt models/brainiac/atlases/nihpd_asym_13.0-18.5_t1w.nii models/brainiac/hdbet/0.model
	uv run dvc add data/master.parquet
	@test ! -d data/derivatives/brainiac || uv run dvc add data/derivatives/brainiac
	@test ! -d data/derivatives/brainiac_preprocess || uv run dvc add data/derivatives/brainiac_preprocess

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
