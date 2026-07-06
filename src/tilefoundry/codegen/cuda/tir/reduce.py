"""Codegen for the Reduce TIR stmt — emits the uniform runtime reduce call.

A sharded reduce emits ``reduce_sharded<Op, Axes>(src, dst, workspace)`` when the
lowering provided a workspace, else the 2-arg ``reduce<Op, Axes>(src, dst)``; a
non-sharded reduce emits the rank-aware ``reduce_1d`` / ``reduce`` fallback.
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
        # 3-arg ``reduce_sharded`` when the lowering sized a workspace
        # (``_analyze_cross_warp_workspace``), else the 2-arg ``reduce``.
        axes_t = _axes_pack_typename(call.target.axes)
        if len(call.args) >= 3:
            ws_n = ctx.name_for(call.args[2])
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
