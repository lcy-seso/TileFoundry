"""HIR ``Tuple`` — explicit tuple construction.

``Tuple((a, b))``: value-form tuple of expressions. The type is
``TupleType(fields=(a.type, b.type))``. Not a registered Op — this is
an IR-level construct emitted by the parser for ``return (a, b)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Expr


@dataclass(frozen=True)
class Tuple(Expr):
    """Value-form explicit tuple construction."""
    elements: tuple[Expr, ...]


__all__ = ["Tuple"]
