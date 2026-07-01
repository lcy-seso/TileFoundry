from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.types.shape_dim import ShapeDim

# Spec: hir.md §4


@dataclass(frozen=True)
class GridRegionExpr(Expr):
    """Loop-phi-shaped structured SSA folding a tile loop into one Expr.

    ``induction_var`` ranges over ``range(start, extent, step)``. On the first
    iteration each ``carried_args`` phi is bound to the matching ``init_args``
    value; each iteration evaluates ``body`` and feeds ``yield_values`` back
    into ``carried_args``. ``init_args`` / ``carried_args`` / ``yield_values``
    have equal length (all empty for a no-carry loop).

    ``start`` / ``extent`` / ``step`` are a ``ShapeDim`` (a static ``int`` or a
    ``DimVar`` / dim ``Expr``); a symbolic value is resolved to a concrete
    ``int`` by the evaluator from the call's argument-shape bindings. ``start``
    defaults to ``0`` (the common ``tile(extent)`` / ``range(extent)`` form);
    the ``range(start, extent, step)`` surface sets it explicitly. ``start`` and
    ``extent`` are the half-open ``[start, extent)`` Python-range endpoints
    (``extent`` is the stop value, not a count).

    The DSL surfaces ``for i in tile(...)`` and ``for i in range(...)`` both
    lower to this node — they share the loop domain; ``tile`` additionally binds
    its loop variable as a slice (``x[:, t]``) while ``range`` binds a scalar.
    """

    induction_var: Var
    carried_args: tuple[Var, ...]
    init_args: tuple[Expr, ...]
    body: Expr
    yield_values: tuple[Expr, ...]
    extent: ShapeDim
    step: ShapeDim
    start: ShapeDim = 0


__all__ = ["GridRegionExpr"]
