"""FreeSolvE: Solvent-Mediated Differentiable Pose Refinement and Biophysical Inductive Bias.

High-speed continuum solvation PDE + articulated kinematics engine for PyTorch.
"""

from solvdock import (
    dock,
    relax,
    DockingResult,
    extract_pocket,
    FlexibleRefiner,
)
from solvdock.nn import (
    SolvDockPhysicsLoss,
    FreeSolvEPhysicsLoss,
)
from solvdock.pipeline.refiner import SolvDockRefiner, SolvDockRefiner as PoseRefiner

__version__ = "0.1.0"
__all__ = [
    "dock",
    "relax",
    "DockingResult",
    "extract_pocket",
    "FlexibleRefiner",
    "PoseRefiner",
    "SolvDockRefiner",
    "SolvDockPhysicsLoss",
    "FreeSolvEPhysicsLoss",
]
