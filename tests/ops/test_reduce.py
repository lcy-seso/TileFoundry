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
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Split,
    layout_axis_to_tensor_axis,
)
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
