"""Per-analysis Context dataclasses.

TypeInferContext is the type-of-cache + unified error helper. VerifyContext
extends it with a mesh scope stack. CostContext is a placeholder.

The concrete CUDA CodegenContext lives in tilefoundry.codegen.cuda.context —
this module only needs the generic contract, so codegen-side context is
imported indirectly (no cycle).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NoReturn, Union

from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.expr import Call, Constant, Expr, Var
from tilefoundry.ir.core.stmt import Stmt
from tilefoundry.ir.types.shard.layout import EMPTY_LAYOUT
from tilefoundry.ir.types.tensor_type import DType, TensorType, Type

from .registries import typeinfer_registry


def _constant_type(value: object) -> TensorType:
    if isinstance(value, bool):
        dtype = DType.bool
    elif isinstance(value, int):
        dtype = DType.i64
    elif isinstance(value, float):
        dtype = DType.f32
    else:
        raise VerifyError(f"Constant: unsupported value type {type(value).__name__}")
    return TensorType(shape=(), dtype=dtype, layout=EMPTY_LAYOUT, storage=None)


@dataclass
class TypeInferContext:
    """Lazy type-of cache. Spec §4.

    ``mesh_scope`` carries the enclosing ``MeshScope`` stack into a registered
    ``verify_stmt`` handler (the stmt walk sets it before dispatch), so a
    mesh-scoped op (``Mma`` atom-scope, ``Sync``) can verify against its
    enclosing meshes without the generic verify importing those op classes.
    """

    module: Any = None
    cache: dict[Expr, Type] = field(default_factory=dict)
    mesh_scope: tuple = ()

    def type_of(self, expr: Expr) -> Type:
        cached = self.cache.get(expr)
        if cached is not None:
            return cached
        computed = self._compute(expr)
        self.cache[expr] = computed
        return computed

    def _compute(self, expr: Expr) -> Type:
        if isinstance(expr, Constant):
            declared = getattr(expr, "type", None)
            if declared is not None:
                return declared
            return _constant_type(expr.value)
        if isinstance(expr, Var):
            return expr.type
        if isinstance(expr, Call):
            op_cls = type(expr.target)
            fn = typeinfer_registry.lookup(op_cls)
            if fn is None:
                self.error(expr, f"no typeinfer registered for {op_cls.__name__}")
            return fn(expr, self)
        declared = getattr(expr, "type", None)
        if declared is not None:
            return declared
        self.error(expr, f"no typeinfer rule for Expr subclass {type(expr).__name__}")

    def error(self, node: Union[Expr, Stmt], msg: str) -> NoReturn:
        if isinstance(node, Call):
            name = type(node.target).__name__
        else:
            name = type(node).__name__
        loc = getattr(node, "loc", None)
        where = f"\n  at {loc}" if loc else ""
        raise VerifyError(f"{name}: {msg}{where}")


@dataclass
class VerifyContext(TypeInferContext):
    """Extends TypeInferContext with a mesh scope stack.

    VerifyVisitor pushes/pops the enclosing `MeshScope.mesh` as it traverses,
    so per-stmt verify handlers can check that any `ShardLayout.mesh`
    referenced at the current point is in scope (see tir spec §6.6).
    """

    mesh_stack: list = field(default_factory=list)


@dataclass
class CostContext(TypeInferContext):
    """Cost-model context. MVP placeholder — no handlers registered."""


@dataclass
class Cost:
    """Placeholder cost record. Populated by future costmodel handlers."""

    flops: int = 0
    bytes: int = 0


__all__ = [
    "TypeInferContext",
    "VerifyContext",
    "CostContext",
    "Cost",
]
