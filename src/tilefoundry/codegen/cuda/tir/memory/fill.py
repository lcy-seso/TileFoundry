from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.core import Constant
from tilefoundry.ir.tir.memory import Fill
from tilefoundry.ir.types.shape_helpers import shape_runtime_total


# Element-count semantics: static dims contribute their literal extent;
# ``DimVar`` axes resolve through ``shape_runtime_total`` to the
# dispatch-emitted ``<param>_shape_<axis>`` scalar so the runtime fill
# count tracks the actual tensor extent. Static allocations elsewhere
# still size to the envelope ``hi`` (see ``shape_helpers.upper_bound``).
@register_codegen_cuda(Fill)
def _emit(call, ctx: CodegenContext) -> None:
    tensor, value = call.args[0], call.args[1]
    dst_n = ctx.name_for(tensor)
    val = value.value if isinstance(value, Constant) else 0.0
    N = shape_runtime_total(tensor.type.shape, ctx._dim_var_runtime)
    ctx.emit(f"tilefoundry::ops::fill({dst_n}, {val}f, {N});")
