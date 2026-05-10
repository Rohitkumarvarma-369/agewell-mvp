"""Tests for canonical FreeSurfer column alignment."""

from agewell.config import load_cfg
from agewell.data.freesurfer_columns import canonical_freesurfer_columns


def test_canonical_freesurfer_column_list_matches_model_config() -> None:
    """The Phase 3 mri_vol encoder feature count matches the Phase 1 column list."""
    columns = canonical_freesurfer_columns()
    cfg = load_cfg()

    assert len(columns) == 328
    assert len(columns) == len(set(columns))
    assert int(cfg.model.encoders.mri_vol.n_features) == len(columns)
