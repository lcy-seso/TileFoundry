"""Derived Visitors — TypeInferVisitor / VerifyVisitor / CodegenVisitor / CostVisitor.

`AnalysisRegistry` instance with a traversal skeleton from
tilefoundry.ir.visitor.

The `registry` is exposed as an advanced constructor param (default: the
canonical module-level registry for that analysis). Default path uses the
module-level registry directly; passing a custom one is an advanced
extension point for sandbox tests or grouped dispatch.
"""
from __future__ import annotations

from tilefoundry.ir.core.expr import Call, Expr, Var
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope
from tilefoundry.ir.types.tensor_type import Type, UnitType
from tilefoundry.ir.visitor import ExprVisitor, StmtVisitor

from .contexts import Cost, CostContext, TypeInferContext, VerifyContext
from .registries import (
    AnalysisRegistry,
    codegen_cpu_registry,
    codegen_cuda_registry,
    costmodel_registry,
    verify_stmt_registry,
)


class TypeInferVisitor(ExprVisitor[Type]):
    """Walks hir Function.body / tir Stmt Expr fields filling type cache.

    Practical usage: call ``ctx.type_of(expr)`` directly — that already
    drives dispatch through ``typeinfer_registry``. This class exists so
    code that wants to walk a whole Expr tree in visitor-style (e.g.
    collecting types for every sub-expression) has an explicit entry.

    **Registry ownership**: unlike VerifyVisitor / CodegenVisitor, the
    typeinfer registry is consulted by ``TypeInferContext._compute``, not
    by this visitor. Callers that need a custom registry subclass
    ``TypeInferContext`` and override ``_compute`` (or inject the registry
    there). No ``registry=`` constructor param exists here so the API
    doesn't mislead into thinking it would take effect.
    """

    def __init__(self, ctx: TypeInferContext) -> None:
        self.ctx = ctx

    def visit_Call(self, call: Call) -> Type:
        return self.ctx.type_of(call)

    def visit_Var(self, var: Var) -> Type:
        return var.type

    def visit_Constant(self, c) -> Type:
        return self.ctx.type_of(c)


class VerifyVisitor(StmtVisitor[None]):
    """Dispatch verify_stmt per Stmt subclass.

    Unregistered Stmt subclasses (typically control-flow: For/While/If/
    Assign/MeshScope) fall through to the StmtVisitor default traversal,
    which recurses into children without raising. That is intentional —
    control-flow stmts whose semantics are fully captured by structure need
    no custom verify.
    """

    def __init__(
        self,
        ctx: VerifyContext,
        registry: AnalysisRegistry = verify_stmt_registry,
    ) -> None:
        # 默认路径:registry 参数不传,用模块级 verify_stmt_registry。
        # 显式传 registry 是高级扩展点(比如 sandbox 测试、分组 dispatch),
        # 日常 verify pass 不需要动它。
        self.ctx = ctx
        self.registry = registry

    def generic_visit(self, stmt: Stmt) -> None:
        if isinstance(stmt, Evaluate):
            # Effect-ful Op invocation in Stmt position: dispatch verify on the
            # Op class. The handler ABI is Call-based, so feed it a Call built
            # from the Op and its args.
            op = stmt.callable
            fn = self.registry.lookup(type(op))
            if fn is not None:
                call = Call(type=UnitType(), target=op, args=stmt.args)
                fn(call, self.ctx)
            super().generic_visit(stmt)
            return
        fn = self.registry.lookup(type(stmt))
        if fn is not None:
            fn(stmt, self.ctx)
        super().generic_visit(stmt)

    def visit_MeshScope(self, stmt: MeshScope) -> None:
        self.ctx.mesh_stack.append(stmt.mesh)
        try:
            # Fire any custom verify handler for MeshScope (none by default),
            # then recurse into body with the scope active.
            fn = self.registry.lookup(MeshScope)
            if fn is not None:
                fn(stmt, self.ctx)
            for child in stmt.body:
                self.visit(child)
        finally:
            self.ctx.mesh_stack.pop()


class CodegenVisitor:
    """Dual-path dispatch: Op (via Call) → str fragment; Stmt → emit into ctx.

    Not a subclass of StmtVisitor/ExprVisitor — codegen's two paths return
    different types (str for Op, None for Stmt) and need different entries.
    Uses `visit_<ClassName>` lookup style for API consistency with the rest
    of the visitor family.
    """

    def __init__(
        self,
        ctx,  # CodegenContext; concrete per-target type lives with the target
        target: str,
    ) -> None:
        self.ctx = ctx
        self.target = target
        self.registry = _codegen_registry_for(target)

    def emit_stmt(self, stmt: Stmt) -> None:
        fn = self.registry.lookup(type(stmt))
        if fn is None:
            raise RuntimeError(
                f"no @register_codegen_{self.target} for Stmt {type(stmt).__name__}"
            )
        fn(stmt, self.ctx)

    def emit_expr(self, expr: Expr) -> str:
        if isinstance(expr, Call):
            fn = self.registry.lookup(type(expr.target))
            if fn is None:
                raise RuntimeError(
                    f"no @register_codegen_{self.target} for Op "
                    f"{type(expr.target).__name__}"
                )
            return fn(expr, self.ctx)
        # Leaf Expr nodes (Var / Constant) are emitted by the target's
        # CodegenContext helpers (e.g. ctx.name_for / ctx.literal). Callers
        # that want a generic fallback should override emit_expr.
        raise RuntimeError(
            f"CodegenVisitor.emit_expr: leaf Expr {type(expr).__name__} "
            "has no default emission; handle via target ctx helpers."
        )


class CostVisitor(ExprVisitor[Cost]):
    """Placeholder. MVP does not implement a cost model; the class / registry
    are in place so future handlers can register without a spec churn."""

    def __init__(
        self,
        ctx: CostContext,
        registry: AnalysisRegistry = costmodel_registry,
    ) -> None:
        self.ctx = ctx
        self.registry = registry

    def visit_Call(self, call: Call) -> Cost:
        fn = self.registry.lookup(type(call.target))
        if fn is None:
            return Cost()
        return fn(call, self.ctx)


def _codegen_registry_for(target: str) -> AnalysisRegistry:
    if target == "cuda":
        return codegen_cuda_registry
    if target == "cpu":
        return codegen_cpu_registry
    raise ValueError(f"unknown codegen target: {target!r}")


__all__ = [
    "TypeInferVisitor",
    "VerifyVisitor",
    "CodegenVisitor",
    "CostVisitor",
]
