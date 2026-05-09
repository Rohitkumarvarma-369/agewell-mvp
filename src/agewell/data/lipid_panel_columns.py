"""Fixed-order public lipid panel columns.

The current public Kaggle lipidomics CSV has no plasma lipid panel columns.
It contributes CSF amyloid/tau biomarkers and APOE4 under the existing
`lipid` modality name so later CBR adapters can add true plasma lipidomics
without changing downstream encoder names.
"""

LIPID_PANEL_COLUMNS: tuple[str, ...] = ()
