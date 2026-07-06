"""HIR generic Reduce op with kind enum.

Spec: hir.md §2.2
"""

from __future__ import annotations

import isl

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    build_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.relation_build import build_domain
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout

__all__ = ["ReduceKind", "Reduce"]

@register_op
class Reduce(Op):
    """Axis reduction over ``x`` (``mean`` / ``sum`` / ``abs_max`` / ``max``).

    Spec: hir.md §2.2

    ``Reduce(x, axes=(0,), keepdim=True, kind=ReduceKind.MEAN)`` lowers to TIR
    ``Reduce`` (whose hardware dispatch is derived by codegen + runtime from the
    operand ``ShardLayout`` / ``Mesh``).

    Output layout contract. When ``x`` is sharded
    (``x.type.layout: ShardLayout``), reducing over an axis that is ``Split``
    across mesh axes produces a result every participant sees identically. The
    contract is the natural "project to local layout, take default strides":

    1. Project the input ``ShardLayout`` to its local layout under the current
       device's shard view: every cute position bound to a mesh axis via a
       ``Split`` attr shrinks to size 1 (the mesh handles per-shard dispatch).
    2. Reduce the local layout: every cute position that falls within a reduced
       tensor axis collapses to size 1.
    3. Strides follow the row-major default for the resulting local shape:
       size-1 positions carry stride 0; other positions carry the default
       contiguous stride for the surviving dimension(s).
    4. Attrs: every ``Split(axis=L)`` whose cute position ``L`` falls within a
       reduced tensor axis becomes ``Broadcast()``. Non-reduced mesh axes
       preserve their attr (``Split`` / ``Partial`` / ``Broadcast`` /
       ``Dynamic``).
    5. The logical ``TensorType.shape`` follows numpy semantics: a reduced axis
       becomes 1 when ``keepdim=True``, otherwise it is removed.
    6. ``storage`` is preserved.

    If ``x.type.layout`` is plain (non-``ShardLayout``) or ``None``, the output
    layout passes through unchanged. The default-contiguous-stride rule applies
    only when the input ``ShardLayout`` is itself in default-stride form; a
    non-default-stride (transposed / permuted) input MUST carry explicit strides
    from its producer, else verify / typeinfer rejects it.

    Cute position → tensor axis mapping uses the left-to-right product
    convention: each tensor axis ``k`` claims as many cute positions as needed
    to accumulate to ``tensor_shape[k]``; trailing cute positions attach to the
    last tensor axis; a singleton tensor axis claims exactly one cute position.

    Worked examples. rmsnorm ``(1, 1536) → (1, 1)`` with every mesh axis
    covering the reduced last axis: every cute position ends up
    size-1-non-reduced (outer axis 0) or reduced, so the output is
    ``shape=(1,1,1,1) strides=(0,0,0,0) attrs=(Broadcast, Broadcast)``. A
    partial reduce ``(M, N) → (M, 1)`` with the mesh covering only the reduced
    axis keeps outer axis 0's stride (it still indexes distinct rows); only the
    reduced positions go to size 1 stride 0.
    """
    x = ParamDef(kind="input", pattern=Tensor)
    axes = ParamDef(kind="attribute", annotation=tuple)
    keepdim = ParamDef(kind="attribute", annotation=bool, default=True)
    kind = ParamDef(kind="attribute", annotation=ReduceKind, default=ReduceKind.MEAN)

def _reduced_axes(call: "Call", rank: int) -> tuple:
    return tuple(a % rank if a < 0 else a for a in call.target.axes)


@register_type_relation(Reduce)
def _reduce_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward relation for Reduce: an identity input map; the output map keeps
    every axis (keepdim) or drops the reduced axes (no keepdim). The reduced
    axes are reported as completely-reduced dims, so a Split on them collapses
    to Broadcast and their cute positions collapse to size 1.
    """
    (x,) = input_types
    rank = len(x.shape)
    reduced = _reduced_axes(call, rank)
    dims = [f"d{i}" for i in range(rank)]
    src = "[" + ", ".join(dims) + "]"
    in_map = isl.map(f"{{ {src} -> [{', '.join(dims)}] }}")
    out_dims = (
        dims if call.target.keepdim else [dims[i] for i in range(rank) if i not in reduced]
    )
    out_map = isl.map(f"{{ {src} -> [{', '.join(out_dims)}] }}")
    return AccessRelationResult(domain=build_domain(x.shape), maps=(in_map, out_map))


@register_typeinfer(Reduce)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    keepdim = call.target.keepdim
    rank = len(x_ty.shape)
    reduced = _reduced_axes(call, rank)

    new_shape = list(x_ty.shape)
    for a in sorted(reduced, reverse=True):
        if keepdim:
            new_shape[a] = 1
        else:
            new_shape.pop(a)
    out_shape = tuple(new_shape)

    new_layout = x_ty.layout
    if isinstance(x_ty.layout, ShardLayout):
        relation = build_relation(call, (x_ty,), ctx)
        derived = derive_output_shard_layout(
            (x_ty,),
            relation,
            out_shape,
            complete_reduction_dims=frozenset(reduced),
            fresh_strides=True,
        )
        if derived is not None:
            new_layout = derived

    return TensorType(
        shape=out_shape,
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=x_ty.storage,
    )


@register_eval(Reduce)
def _eval_reduce(ctx):
    x = ctx.args[0].data
    axes = tuple(ctx.op.axes)
    keepdim = ctx.op.keepdim
    kind = ctx.op.kind
    if kind is ReduceKind.MEAN:
        out = x.mean(dim=axes, keepdim=keepdim)
    elif kind is ReduceKind.SUM:
        out = x.sum(dim=axes, keepdim=keepdim)
    elif kind is ReduceKind.ABS_MAX:
        out = x.abs().amax(dim=axes, keepdim=keepdim)
    elif kind is ReduceKind.MAX:
        out = x.amax(dim=axes, keepdim=keepdim)
    else:
        raise ValueError(f"evaluator: unsupported ReduceKind {kind}")
    return TensorValue(data=out, type=ctx.result_type)
