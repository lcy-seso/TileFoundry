"""TileFoundry runtime layer — ``RuntimeModule`` container + loader.

type returned by ``tilefoundry.build`` / ``tilefoundry.compile``. It is directly
callable — ``rm(a)`` for auto-alloc, ``rm(a, out)`` for pre-alloc.
"""
from __future__ import annotations

from .module import (
    CallableType,
    KernelInfo,
    LaunchConfig,
    ParamABI,
    RuntimeFunction,
    RuntimeModule,
)

__all__ = [
    "CallableType",
    "KernelInfo",
    "LaunchConfig",
    "ParamABI",
    "RuntimeFunction",
    "RuntimeModule",
]
