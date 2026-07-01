"""Emitter for ``tir.For`` — C-style for loop."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.core import Constant
from tilefoundry.ir.tir.stmts import For


@register_codegen_cuda(For)
def _emit(node: For, ctx: CodegenContext) -> None:
    iv_name = ctx.name_for(node.induction_var)
    start = node.start.value if isinstance(node.start, Constant) else 0
    stop = node.stop.value if isinstance(node.stop, Constant) else 1
    step = node.step.value if isinstance(node.step, Constant) else 1

    ctx.emit(f"for (int {iv_name} = {start}; {iv_name} < {stop}; {iv_name} += {step}) {{")
    ctx.indent()
    ctx.emit_node(node.body)
    ctx.dedent()
    ctx.emit("}")
