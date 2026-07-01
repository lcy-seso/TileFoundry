from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType


@register_op
class Gather(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    indices = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)
def _norm_axis(axis: int, rank: int) -> int:
    a = axis + rank if axis < 0 else axis
    if a < 0 or a >= rank:
        raise TypeError(f"Gather: axis {axis} out of range for rank {rank}")
    return a


@register_typeinfer(Gather)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    idx_ty = ctx.type_of(call.args[1])
    axis = _norm_axis(call.target.axis, len(x_ty.shape))
    new_shape = list(x_ty.shape)
    # Replace axis-th dim with gathered indices' shape dims (flatten).
    new_shape = new_shape[:axis] + list(idx_ty.shape) + new_shape[axis + 1:]
    return TensorType(
        shape=tuple(new_shape), dtype=x_ty.dtype, layout=x_ty.layout, storage=x_ty.storage
    )


@register_eval(Gather)
def _eval_gather(ctx):
    x = ctx.args[0].data
    indices = ctx.args[1].data
    axis = _norm_axis(ctx.op.axis, x.dim())
    out = torch.index_select(x, axis, indices.reshape(-1).long())
    new_shape = x.shape[:axis] + tuple(indices.shape) + x.shape[axis + 1:]
    return TensorValue(data=out.reshape(new_shape), type=ctx.result_type)
