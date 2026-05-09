# AgeWell-IN MVP

AgeWell-IN is a sparse-modality multimodal research platform for healthy brain aging,
MCI, dementia, and Alzheimer's research. This repository starts with the Phase 0
foundation: project tooling, service infrastructure, configuration, health checks, and
test scaffolding.

The public-data baseline reads Kaggle datasets from:

```bash
/home/rohit/kaggle-iisc/kaggle-cli/downloads
```

Phase 0 does not implement ML, adapters, imaging pipelines, or agents. Those are added
phase by phase after the foundation gates are green.

## First Boot

```bash
make bootstrap
make up
make health
make test
```

See `docs/runbooks/00_first_boot.md` for details.
