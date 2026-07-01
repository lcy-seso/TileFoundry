from __future__ import annotations

import importlib
import pkgutil

from .function import Function
from .grid_region import GridRegionExpr


def _auto_import(pkg_name: str) -> None:
    pkg = importlib.import_module(pkg_name)
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg_name + "."):
        importlib.import_module(modname)


_auto_import("tilefoundry.ir.hir")

# Retrofit GLOBAL access_relation handlers onto existing primitives.
from tilefoundry.visitor_registry import access_relation_primitives  # noqa: E402, F401

__all__ = ["Function", "GridRegionExpr"]
