"""Emitter for ``tir.LetStmt``.

LetStmt wraps a value binding: `auto %var = <emit(value)>` then the scoped
body. Dispatch is two-step:

1. Look up an emitter keyed on the ``Op`` class of ``value.target`` — each
   TIR-owned Expr Op (``tir.memory.AllocTensor``, ``tir.memory.PtrOf``,
   ``tir.memory.TensorView``, etc.) registers its own emitter with
   signature ``(let: LetStmt, ctx) -> None``.
2. Recurse into the Sequential body.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import (
    CodegenContext,
    lookup,
    register_codegen_cuda,
)
from tilefoundry.ir.core import Call
from tilefoundry.ir.tir.stmts import LetStmt


@register_codegen_cuda(LetStmt)
def _emit(node: LetStmt, ctx: CodegenContext) -> None:
    if not isinstance(node.value, Call):
        raise RuntimeError(
            f"LetStmt.value must be a Call (TIR-owned Expr Op), "
            f"got {type(node.value).__name__}"
        )
    op = node.value.target
    handler = lookup(type(op))
    if handler is None:
        raise RuntimeError(
            f"no @register_codegen_cuda for Op {type(op).__name__} "
            f"(LetStmt value target)"
        )
    handler(node, ctx)
    ctx.emit_node(node.body)
