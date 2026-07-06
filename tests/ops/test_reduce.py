"""HIR Reduce typeinfer.

The output ``ShardLayout`` of ``Reduce`` collapses every Split that lives on a
reduced tensor axis into ``Broadcast`` and shrinks the matching cute layout
positions to size 1 with stride 0 (broadcast view); a Split on a non-reduced
axis is preserved. An unsharded input passes through.
"""
from __future__ import annotations

import pytest
import torch

from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.codegen.cuda.tir.reduce import _lane_reduced
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.tir.reduce import Reduce as TirReduce
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard import Topology
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Split,
    layout_axis_to_tensor_axis,
)
from tilefoundry.passes.transforms.hir_to_tir import _analyze_cross_warp_workspace
from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    TypeInferCase,
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
)

_RMEM = StorageKind.RMEM
_BF = DType.bf16
# Two-axis mesh; the reduce cases reuse it for input and expectation so the
# preserved mesh compares equal.
_M = mesh((6, 32), ("w", "t"))

_MEAN_LAST = Reduce(axes=(-1,), keepdim=True, kind=ReduceKind.MEAN)

CASES = [
    # Reduced-axis Splits become Broadcast; cute positions on the reduced axis
    # shrink to size 1 / stride 0 (broadcast view input).
    TypeInferCase(
        "reduced_axis_splits_become_broadcast",
        _MEAN_LAST,
        (
            sharded(
                (1, 1536), (Split(1), Split(2)), _M,
                cute=(1, 6, 32, 8), strides=(0, 0, 0, 1), dtype=_BF, storage=_RMEM,
            ),
        ),
        sharded(
            (1, 1), (Broadcast(), Broadcast()), _M,
            cute=(1, 1, 1, 1), strides=(0, 0, 0, 0), dtype=_BF, storage=_RMEM,
        ),
    ),
    # Same, but the input cute carries a global (non-zero) stride view: reduced
    # positions are still zeroed.
    TypeInferCase(
        "zeroes_reduced_positions_for_global_view",
        _MEAN_LAST,
        (
            sharded(
                (1, 1536), (Split(1), Split(2)), _M,
                cute=(1, 6, 32, 8), strides=(1536, 256, 8, 1), dtype=_BF, storage=_RMEM,
            ),
        ),
        sharded(
            (1, 1), (Broadcast(), Broadcast()), _M,
            cute=(1, 1, 1, 1), strides=(0, 0, 0, 0), dtype=_BF, storage=_RMEM,
        ),
    ),
    # A Split on the non-reduced axis is preserved; the reduced axis -> Broadcast.
    TypeInferCase(
        "preserves_non_reduced_axis_split",
        Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM),
        (sharded((16, 32), (Split(0), Split(1)), _M, dtype=_BF, storage=_RMEM),),
        sharded(
            (16, 1), (Split(0), Broadcast()), _M,
            cute=(16, 1), strides=(1, 0), dtype=_BF, storage=_RMEM,
        ),
    ),
    # keepdim=False pops the reduced axis from the shape; the layout still
    # broadcasts the reduced positions.
    TypeInferCase(
        "keepdim_false_pops_shape",
        Reduce(axes=(1,), keepdim=False, kind=ReduceKind.MEAN),
        (
            sharded(
                (1, 1536), (Split(1), Split(2)), _M,
                cute=(1, 6, 32, 8), strides=(1536, 256, 8, 1), dtype=_BF, storage=_RMEM,
            ),
        ),
        sharded(
            (1,), (Broadcast(), Broadcast()), _M,
            cute=(1, 1, 1, 1), strides=(0, 0, 0, 0), dtype=_BF, storage=_RMEM,
        ),
    ),
    # Unsharded input passes through (no layout).
    TypeInferCase(
        "unsharded_passes_through",
        Reduce(axes=(0,), keepdim=True, kind=ReduceKind.SUM),
        (ten((8, 16), DType.f32, storage=_RMEM),),
        ten((1, 16), DType.f32, storage=_RMEM),
    ),
    # Implicit (None) strides reduce to a fresh C-order output (no None indexing).
    TypeInferCase(
        "implicit_strides_fresh_output",
        Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM),
        (sharded((16, 32), (Split(0), Split(1)), _M, strides=None, dtype=_BF, storage=_RMEM),),
        sharded(
            (16, 1), (Split(0), Broadcast()), _M,
            cute=(16, 1), strides=(1, 0), dtype=_BF, storage=_RMEM,
        ),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reduce_typeinfer(case):
    run_typeinfer_case(case)


def test_layout_axis_to_tensor_axis_factorized() -> None:
    # tensor (1, 1536) with cute (1, 6, 32, 8): cute pos 0 -> axis 0; 1/2/3 -> axis 1.
    assert layout_axis_to_tensor_axis((1, 6, 32, 8), (1, 1536)) == [0, 1, 1, 1]


def test_layout_axis_to_tensor_axis_one_to_one() -> None:
    assert layout_axis_to_tensor_axis((16, 32), (16, 32)) == [0, 1]


@pytest.mark.parametrize(
    "op,ref,atol",
    [
        (
            Reduce(axes=(1,), keepdim=True, kind=ReduceKind.MEAN),
            lambda x: x.mean(1, keepdim=True), 1e-6,
        ),
        (
            Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM),
            lambda x: x.sum(1, keepdim=True), 1e-5,
        ),
        (
            Reduce(axes=(1,), keepdim=False, kind=ReduceKind.ABS_MAX),
            lambda x: x.abs().amax(1), 1e-6,
        ),
        (
            Reduce(axes=(1,), keepdim=True, kind=ReduceKind.MAX),
            lambda x: x.amax(1, keepdim=True), 1e-6,
        ),
    ],
    ids=["mean", "sum", "abs_max", "max"],
)
def test_reduce_evaluate(op, ref, atol):
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    run_eval_case(EvalCase("", op, (x,), ref(x), atol=atol))


def test_reduce_max_is_signed_not_abs_max():
    """``ReduceKind.MAX`` is the signed max — distinct from ``ABS_MAX`` when the
    largest-magnitude element is negative."""
    x = torch.tensor([[-5.0, 1.0, 2.0]])
    run_eval_case(
        EvalCase("", Reduce(axes=(-1,), keepdim=True, kind=ReduceKind.MAX),
                 (x,), torch.tensor([[2.0]]), atol=0.0)
    )
    run_eval_case(
        EvalCase("", Reduce(axes=(-1,), keepdim=True, kind=ReduceKind.ABS_MAX),
                 (x,), torch.tensor([[5.0]]), atol=0.0)
    )


# ── Cross-warp reduce path selection (codegen-derived, no op attribute) ──────
#
# The runtime has two sharded multi-warp templates: ``reduce_intra_cta`` (lane
# butterfly + cross-warp combine) and ``reduce_cross_warp`` (cross-warp combine
# only, each lane keeps its own output cells). Which one applies is a pure
# function of the operand layouts — a reduced Split on a lane axis vs on a
# warp-only axis — so codegen derives it from ``(src, dst)`` and the ``Reduce``
# op carries no selection attribute.

# rmsnorm-like: reduce the last axis, whose Split covers both the warp (w) and
# lane (t) mesh axes → a reduced lane axis → intra-cta.
_THREAD_A = Topology("thread", 6 * 32)
_MESH_A = mesh((6, 32), ("w", "t"), topology=_THREAD_A)
# cross-expert-like: reduce the warp axis (tk) only; the lane axis (hc) carries
# distinct output cells → no reduced lane axis → cross-warp.
_THREAD_B = Topology("thread", 4 * 32)
_MESH_B = mesh((4, 32), ("tk", "hc"), topology=_THREAD_B)


def _case_a_src_dst():
    src = sharded((1, 1536), (Split(1), Split(2)), _MESH_A,
                  cute=(1, 6, 32, 8), strides=(1536, 256, 8, 1), dtype=_BF, storage=_RMEM)
    dst = sharded((1, 1), (Broadcast(), Broadcast()), _MESH_A,
                  cute=(1, 1, 1, 1), strides=(0, 0, 0, 0), dtype=_BF, storage=_RMEM)
    return src, dst


def _case_b_src_dst():
    src = sharded((4, 32), (Split(0), Split(1)), _MESH_B,
                  cute=(4, 32), strides=(32, 1), dtype=_BF, storage=_RMEM)
    dst = sharded((1, 32), (Broadcast(), Split(1)), _MESH_B,
                  cute=(1, 32), strides=(0, 1), dtype=_BF, storage=_RMEM)
    return src, dst


def test_lane_reduced_true_when_reduced_axis_is_a_lane_axis():
    src, dst = _case_a_src_dst()
    assert _lane_reduced(src, dst) is True


def test_lane_reduced_false_when_reduction_crosses_warps_only():
    src, dst = _case_b_src_dst()
    assert _lane_reduced(src, dst) is False


def test_lane_reduced_defaults_true_without_reduced_mesh_axis():
    # A non-reduced sharded output (no Split→Broadcast) has no reduced mesh axis;
    # the intra-cta path is the safe default.
    src, _ = _case_a_src_dst()
    assert _lane_reduced(src, src) is True


def test_analyze_workspace_reports_lane_reduced_and_sizes():
    src_a, _ = _case_a_src_dst()
    ws_a, _wpg_a, _dt_a, lane_a = _analyze_cross_warp_workspace(src_a, (-1,))
    assert lane_a is True and ws_a > 0        # cross-warp over w, folded per lane
    src_b, _ = _case_b_src_dst()
    ws_b, _wpg_b, _dt_b, lane_b = _analyze_cross_warp_workspace(src_b, (0,))
    assert lane_b is False and ws_b > 0       # cross-warp only over tk


def test_tir_reduce_has_no_cross_warp_only_attribute():
    # The selection is codegen-derived; it MUST NOT leak into the TIR op schema.
    op = TirReduce(axes=(-1,), kind=ReduceKind.SUM, warps_per_group=2)
    assert not hasattr(op, "cross_warp_only")
