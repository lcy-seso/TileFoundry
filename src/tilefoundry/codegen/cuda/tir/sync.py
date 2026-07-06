"""Emitter for the ``tir.Sync`` op — a mesh-scoped barrier.

The barrier kind is derived from the participating thread set (shared with
verify via ``classify`` / ``participation`` so codegen cannot disagree):

- whole CTA → ``__syncthreads()`` (or ``__syncwarp()`` when the block is one
  warp);
- a contiguous lane subset inside one warp → ``__syncwarp(mask)`` guarded by a
  participant predicate;
- a warp-aligned contiguous multi-warp subset → ``bar.sync <id>, <count>``
  guarded by a participant predicate, with ``<id>`` allocated implicitly per
  kernel.

Non-participant threads never execute the barrier; every participant executes
the same id/count.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.sync import SyncBarrier, Sync, classify, participation

_SYNC = "tilefoundry::ops::sync"
_KIND = "tilefoundry::ops::SyncKind"


@register_codegen_cuda(Sync)
def _emit(call, ctx: CodegenContext) -> None:
    mesh = call.target.mesh
    barrier = classify(mesh)

    # One uniform runtime entry per barrier: the runtime runs the participant
    # predicate and dispatches to the hardware impl; codegen only passes the
    # barrier kind and the codegen-static participant geometry as compile-time
    # template parameters (the grid counter is the sole runtime argument).
    if barrier is SyncBarrier.GRID:
        # The counter has internal linkage and is defined once per generated
        # module source (see the module template), so a header include never
        # introduces a shared/duplicated global symbol across translation units.
        ctx.emit(f"{_SYNC}<{_KIND}::grid>(tilefoundry::tf_grid_bar_state);")
        return

    p = participation(mesh)

    if barrier is SyncBarrier.SYNCTHREADS:
        ctx.emit(f"{_SYNC}<{_KIND}::syncthreads>();")
        return

    if barrier is SyncBarrier.SYNCWARP:
        if p.full_cta:
            # The whole block is a single warp — every lane participates.
            ctx.emit(f"{_SYNC}<{_KIND}::syncwarp_full>();")
            return
        # A contiguous lane subset of one warp: the runtime predicate keeps
        # non-participant lanes out of the masked warp sync.
        ctx.emit(
            f"{_SYNC}<{_KIND}::syncwarp_masked, {p.base}, {p.count}, "
            f"0x{p.lane_mask:08x}u>();"
        )
        return

    # BAR_SYNC: a warp-aligned multi-warp subset uses a named barrier; the
    # runtime participant predicate keeps non-participants out of it.
    bid = ctx.alloc_barrier_id()
    ctx.emit(
        f"{_SYNC}<{_KIND}::bar_sync, {p.base}, {p.count}, 0u, {bid}>();"
    )
