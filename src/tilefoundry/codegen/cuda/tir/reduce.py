"""Codegen for generic Reduce TIR stmt — tag-dispatched.

The 3-arg form ``tilefoundry::ops::reduce_sharded<Op, Axes>(src, dst, workspace)``
is the sharded-reduce entry. Forwarded directly when the TIR call
carries an optional ``workspace`` 3rd input (lowering emitted an
``AllocTensor(storage=smem)`` for cross-warp staging); the runtime
derives the reduction level and its warps_per_group from the operand
layouts. The 2-arg ``reduce<Op, Axes>(src, dst)`` form remains for
intra-warp / unsharded reductions; the legacy ``reduce_1d`` /
``reduce(M, K, ...)`` rank-aware kernels are kept as a fallback for
the non-sharded code path.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.reduce import Reduce, ReduceKind
from tilefoundry.ir.types.shard.shard_layout import ShardLayout

_REDUCE_TAG = {
    ReduceKind.MEAN: "tilefoundry::ops::mean_op",
    ReduceKind.SUM: "tilefoundry::ops::sum_op",
    ReduceKind.ABS_MAX: "tilefoundry::ops::absmax_op",
}


def _axes_pack_typename(axes: tuple) -> str:
    """Render the HIR ``axes`` tuple as a ``cute::tuple<cute::Int<i>...>``
    template type for the runtime entry point. Using cute's native
    tuple keeps the reduce dispatch idiomatic with the rest of the
    codegen."""
    args = ", ".join(f"cute::Int<{int(a)}>" for a in axes)
    return f"cute::tuple<{args}>"


@register_codegen_cuda(Reduce)
def _emit(call, ctx: CodegenContext) -> None:
    src, dst = call.args[0], call.args[1]
    src_n = ctx.name_for(src)
    dst_n = ctx.name_for(dst)
    op_tag = _REDUCE_TAG[call.target.kind]
    src_ty = src.type
    is_sharded = isinstance(getattr(src_ty, "layout", None), ShardLayout)

    if is_sharded:
        # Two codegen-facing entries: the intra-warp ``reduce<>(src, dst)`` when
        # no workspace is needed, and the workspace-carrying
        # ``reduce_sharded<>(src, dst, ws)`` otherwise. The lowering
        # (``_analyze_cross_warp_workspace``) sizes the workspace capacity and
        # decides which entry applies (``workspace_size == 0`` → 2-arg); the
        # runtime ``reduce_sharded`` selects the cross-warp tier and its
        # warps_per_group from the operand layouts. Cross-CTA reduce is rejected
        # upstream, not emitted here.
        axes_t = _axes_pack_typename(call.target.axes)
        if len(call.args) >= 3:
            ws_n = ctx.name_for(call.args[2])
            # The runtime derives the reduction level (intra-warp-folded vs
            # cross-warp-only) and its warps_per_group from the operand
            # ShardLayouts (see runtime ``reduce_sharded``); codegen emits only
            # the uniform entry plus the operands.
            ctx.emit(
                f"tilefoundry::ops::reduce_sharded<{op_tag}, {axes_t}>"
                f"({src_n}, {dst_n}, {ws_n});"
            )
        else:
            ctx.emit(
                f"tilefoundry::ops::reduce<{op_tag}, {axes_t}>"
                f"({src_n}, {dst_n});"
            )
        return

    # Legacy non-sharded rank-aware fallback.
    src_shape = src_ty.shape
    if len(src_shape) == 1:
        N = int(src_shape[0])
        ctx.emit(f"tilefoundry::ops::reduce_1d({src_n}, {dst_n}, {N}, {op_tag}{{}});")
    else:
        M = int(src_shape[0])
        K = int(src_shape[1])
        ctx.emit(f"tilefoundry::ops::reduce({src_n}, {dst_n}, {M}, {K}, {op_tag}{{}});")
