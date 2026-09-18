"""SolvDock Solvation Engine: Plug-and-play biophysical continuum solvation package."""

from solvdock.solvation.api import (
    SolvationResult,
    predict_solvation,
    batch_predict_solvation,
)
from solvdock.solvation.applicability import (
    ApplicabilityReport,
    check_applicability_domain,
    is_push_pull_nitroaromatic,
)
from solvdock.solvation.born import (
    compute_born_ion_correction,
    compute_effective_born_radius,
)

__all__ = [
    "SolvationResult",
    "predict_solvation",
    "batch_predict_solvation",
    "ApplicabilityReport",
    "check_applicability_domain",
    "is_push_pull_nitroaromatic",
    "compute_born_ion_correction",
    "compute_effective_born_radius",
]
