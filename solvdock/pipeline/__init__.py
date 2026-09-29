"""SolvDock pipeline package."""

from solvdock.pipeline.energy import CombinedPotential


def __getattr__(name: str):
    if name == "SolvDockRefiner":
        from solvdock.pipeline.refiner import SolvDockRefiner
        return SolvDockRefiner
    if name == "FlexibleRefiner":
        from solvdock.pipeline.flexible_refiner import FlexibleRefiner
        return FlexibleRefiner
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["CombinedPotential", "SolvDockRefiner", "FlexibleRefiner"]


