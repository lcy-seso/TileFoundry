"""Codegen for generic Reduce TIR stmt — tag-dispatched.

The 3-arg form ``tilefoundry::ops::reduce<Op, Axes>(src, dst, workspace)``
is the sharded-reduce entry. Forwarded directly when the TIR call
carries an optional ``workspace`` 3rd input (lowering emitted an
``AllocTensor(storage=smem)`` for cross-warp staging). The
2-arg form remains for intra-warp / unsharded reductions; the
legacy ``reduce_1d`` / ``reduce(M, K, ...)`` rank-aware kernels
are kept as a fallback for the non-sharded code path.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.reduce import Reduce, ReduceKind
from tilefoundry.ir.types.shard.shard_layout import Broadcast, ShardLayout, Split

_WARP_SIZE = 32

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


def _lane_reduced(src_ty, dst_ty) -> bool:
    """Whether the reduction folds an intra-warp (lane) mesh axis, derived from
    the operand layouts alone.

    The reduced mesh axes are exactly those the HIR Reduce typeinfer collapsed:
    a ``Split`` in the source that became a ``Broadcast`` in the (reduced)
    destination, matched by mesh-axis index (both operands share the mesh). The
    lane axes are the rightmost warp-sized (<= 32) suffix under the ``thread``
    topology — the lanes a single hardware warp's ``__shfl`` butterfly folds.
    When a reduced axis is a lane axis the intra-warp path
    (``reduce_intra_cta``) is correct; otherwise the reduction crosses warps
    only and ``reduce_cross_warp`` must be used.
    """
    src_l = getattr(src_ty, "layout", None)
    dst_l = getattr(dst_ty, "layout", None)
    if not isinstance(src_l, ShardLayout) or not isinstance(dst_l, ShardLayout):
        return True
    src_attrs, dst_attrs = src_l.attrs, dst_l.attrs
    reduced = {
        i
        for i in range(min(len(src_attrs), len(dst_attrs)))
        if isinstance(src_attrs[i], Split) and isinstance(dst_attrs[i], Broadcast)
    }
    if not reduced:
        return True
    mesh_shape = tuple(src_l.mesh.layout.shape)
    topologies = list(src_l.mesh.topologies)
    thread_axes = 0
    if topologies and getattr(topologies[-1], "name", "") == "thread":
        prod = 1
        for extent in reversed(mesh_shape):
            if not isinstance(extent, int):
                break
            if prod * extent > _WARP_SIZE:
                break
            prod *= extent
            thread_axes += 1
    lane_axes = set(range(len(mesh_shape) - thread_axes, len(mesh_shape)))
    return bool(reduced & lane_axes)


@register_codegen_cuda(Reduce)
def _emit(call, ctx: CodegenContext) -> None:
    src, dst = call.args[0], call.args[1]
    src_n = ctx.name_for(src)
    dst_n = ctx.name_for(dst)
    op_tag = _REDUCE_TAG[call.target.kind]
    src_ty = src.type
    is_sharded = isinstance(getattr(src_ty, "layout", None), ShardLayout)

    if is_sharded:
        # Three tier entry points distinguish how the reduce spans the
        # mesh:
        #   - tier-1 ``reduce<>(src, dst)``           — intra-warp only
        #   - tier-2 ``reduce_intra_cta<>(...)``     — cross-warp, intra-CTA
        #   - tier-3 ``reduce_cross_cta<>(...)``     — cross-CTA (placeholder)
        # ``_analyze_cross_warp_workspace`` selects tier-1 vs tier-2
        # by returning ``workspace_size == 0`` (tier-1) or > 0
        # (tier-2). Tier-3 has no analysis path yet — cross-CTA reduce
        # is rejected upstream rather than emitted here.
        axes_t = _axes_pack_typename(call.target.axes)
        if len(call.args) >= 3:
            ws_n = ctx.name_for(call.args[2])
            wpg = int(call.target.warps_per_group)
            # tier-2a vs tier-2b is a pure function of the operand layouts: a
            # reduced lane axis folds within the warp (intra_cta); a reduce that
            # crosses warps only, with each lane holding its own cells, needs the
            # cross_warp path.
            fn = "reduce_intra_cta" if _lane_reduced(src_ty, dst.type) else "reduce_cross_warp"
            ctx.emit(
                f"tilefoundry::ops::{fn}<{op_tag}, {axes_t}>"
                f"({src_n}, {dst_n}, {ws_n}, {wpg});"
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
