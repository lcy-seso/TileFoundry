"""Per-op evaluator registry.

A handler registered with ``@register_eval(OpClass)`` computes the value
semantics of one HIR Op. The registry is local to the evaluator; it reuses
the shared ``AnalysisRegistry`` container but is not a module-level instance
of the visitor-registry analyses.

Spec: evaluator.md §3
"""
from __future__ import annotations

from typing import Callable

from tilefoundry.visitor_registry.registries import AnalysisRegistry

eval_registry: AnalysisRegistry = AnalysisRegistry("eval")


def register_eval(op_cls: type):
    """Register *fn* as the evaluator for ``op_cls``."""

    def decorator(fn: Callable) -> Callable:
        eval_registry.register(op_cls, fn)
        return fn

    return decorator
