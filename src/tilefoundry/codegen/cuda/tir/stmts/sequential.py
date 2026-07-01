"""Emitter for ``tir.Sequential`` — packs a list of Stmts in order."""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.stmts import Sequential


@register_codegen_cuda(Sequential)
def _emit(node: Sequential, ctx: CodegenContext) -> None:
    for stmt in node.body:
        ctx.emit_node(stmt)
