"""Emitter for `tir.Return`."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.stmts import Return


@register_codegen_cuda(Return)
def _emit(node: Return, ctx: CodegenContext) -> None:
    ctx.emit("return;")
