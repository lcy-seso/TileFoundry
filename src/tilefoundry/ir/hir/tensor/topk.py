"""Top-K selection HIR primitive.

SGLang baseline kernel K11 (TopK Gating Softmax) emits both the top-k values
and their indices. We model just the selection step here; the upstream Softmax
remains a separate node.

"""
from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, TupleValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    register_access_relation,
)


@register_op
class TopK(Op):
    """Multi-output (values, indices). Reduces the ``axis`` dim to length ``k``."""
    x = ParamDef(kind="input", pattern=Tensor)
    k = ParamDef(kind="attribute", annotation=int)
    axis = ParamDef(kind="attribute", annotation=int, default=-1)
@register_typeinfer(TopK)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
    x_ty = ctx.type_of(call.args[0])
    if not x_ty.shape:
        raise TypeError("TopK: x must be at least rank-1")
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise TypeError(f"TopK: axis {call.target.axis} out of range for rank {rank}")
    out_shape = list(x_ty.shape)
    out_shape[axis] = call.target.k
    out_shape = tuple(out_shape)
    values_ty = TensorType(
        shape=out_shape, dtype=x_ty.dtype, layout=x_ty.layout, storage=x_ty.storage
    )
    indices_ty = TensorType(
        shape=out_shape, dtype=DType.i32, layout=x_ty.layout, storage=x_ty.storage
    )
    return TupleType(fields=(values_ty, indices_ty))

@register_access_relation(TopK)
def _topk_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """GLOBAL level.

    The reduction axis is data-dependent (top-k indices come from sort), so the
    input access relation is an isl.map "scans the whole axis" rather than a
    multi_aff. Output values/indices are leading-dims identity with a new
    independent topk axis.
    """
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    in_dims = ", ".join(f"i{i}" for i in range(rank))
    out_dims = ", ".join(f"i{i}" if i != axis else "j" for i in range(rank))
    # Input relation: every output position [.., j, ..] depends on the entire
    # axis range of the input. Express as map dropping the axis dim.
    leading = ", ".join(f"i{i}" for i in range(rank) if i != axis)
    if leading:
        in_rel = isl.map(f"{{ [{out_dims}] -> [{in_dims}] }}")
    else:
        in_rel = isl.map(f"{{ [j] -> [i{axis}] }}")
    # Output identity: trivial map from output to itself.
    out_id = isl.multi_aff(f"{{ [{out_dims}] -> [{out_dims}] }}")
    return AccessRelations(inputs=(in_rel,), outputs=(out_id, out_id))

@register_eval(TopK)
def _eval_topk(ctx):
    vals, idx = torch.topk(ctx.args[0].data, ctx.op.k, dim=ctx.op.axis)
    return TupleValue(
        elements=(
            TensorValue(
                data=vals.to(to_torch_dtype(ctx.result_type.fields[0].dtype)),
                type=ctx.result_type.fields[0],
            ),
            TensorValue(
                data=idx.to(to_torch_dtype(ctx.result_type.fields[1].dtype)),
                type=ctx.result_type.fields[1],
            ),
        )
    )


__all__ = ["TopK"]
