from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core.op import Op

from ..types.tensor_type import Type


@dataclass(frozen=True)
class Expr:
    """Typed SSA value. Base of all expression nodes (hir + tir-embedded).

    `type` is the Expr's result type (TensorType for single-output, TupleType
    for multi-output); `source` is optional debug info. Both kw-only so
    subclasses can declare positional fields without default-order clashes.
    """
    type: Type = field(kw_only=True)
    loc: str | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Constant(Expr):
    value: object


@dataclass(frozen=True)
class Call(Expr):
    """Call to an Op. Produces a value. Cannot be top-level Stmt in tir (§8.5)."""
    target: Op
    args: tuple[Expr, ...]


