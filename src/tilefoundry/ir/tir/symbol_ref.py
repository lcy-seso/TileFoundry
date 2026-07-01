"""SymbolRef — a module-symbol reference to a callee PrimFunction.

Spec: tir.md §9
"""
from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.expr import Expr


@dataclass(frozen=True)
class SymbolRef(Expr):
    """Reference to a callee ``PrimFunction`` by its canonical module name.

    Carried as the ``callable`` of an ``Evaluate(SymbolRef, args)`` function
    invocation and as a ``Launch`` callee. ``type`` is the callee's
    ``CallableType``, built at construction; resolution is module-level
    (``Module.lookup``), not local typeinfer. ``nested`` is reserved for a
    nested module-symbol path and MUST be empty under the current module
    model, which holds only top-level functions.
    """

    name: str
    nested: tuple[str, ...] = ()


def symbol_call(callee, args) -> "Evaluate":  # noqa: F821 -- lazy Evaluate
    """Build ``Evaluate(SymbolRef(callee), args)`` — a Stmt-position call of a
    callee ``PrimFunction`` by symbol. ``SymbolRef.type`` is the callee's
    ``CallableType``, fixed at construction.
    """
    from tilefoundry.ir.tir.stmts import Evaluate  # noqa: PLC0415
    from tilefoundry.ir.types import callable_type_for_prim_function  # noqa: PLC0415

    ref = SymbolRef(name=callee.name, type=callable_type_for_prim_function(callee))
    return Evaluate(callable=ref, args=tuple(args))


__all__ = ["SymbolRef", "symbol_call"]
