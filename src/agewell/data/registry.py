"""Phase 1 adapter registry."""

from __future__ import annotations

from agewell.data.adapters._base import BaseAdapter
from agewell.data.adapters.adni_nifti import ADNINiftiAdapter
from agewell.data.adapters.adni_tabular import ADNITabularAdapter
from agewell.data.adapters.brsdincer import BrsdincerAdapter
from agewell.data.adapters.ixi import IXIAdapter
from agewell.data.adapters.lipidomics import LipidomicsAdapter
from agewell.data.adapters.oasis_cross import OASISCrossAdapter
from agewell.data.adapters.oasis_long import OASISLongAdapter
from agewell.data.adapters.rabieelkharoua import RabieElKharouaAdapter

ADAPTERS: dict[str, type[BaseAdapter]] = {
    "adni_tabular": ADNITabularAdapter,
    "adni_nifti": ADNINiftiAdapter,
    "oasis_cross": OASISCrossAdapter,
    "oasis_long": OASISLongAdapter,
    "brsdincer": BrsdincerAdapter,
    "rabieelkharoua": RabieElKharouaAdapter,
    "lipidomics": LipidomicsAdapter,
    "ixi": IXIAdapter,
}
