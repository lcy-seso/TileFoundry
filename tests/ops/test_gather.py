"""HIR Gather value oracle: select along ``axis`` by (multi-dim) indices."""
from __future__ import annotations

import pytest
import torch

from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.types import DType
from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
    ten,
)


def _gather_ref(x, axis, idx):
    """Reference gather: select along ``axis`` by (possibly multi-dim) ``idx``,
    expanding the indexed axis into ``idx``'s shape."""
    axis %= x.ndim
    flat = x.index_select(axis, idx.flatten().long())
    return flat.reshape(*x.shape[:axis], *idx.shape, *x.shape[axis + 1 :])


@pytest.mark.parametrize(
    "axis,x_shape,idx",
    [
        (0, (6, 3, 4), [[0, 5], [2, 3]]),  # 2-D index grid -> [2, 2, 3, 4]
        (1, (6, 3, 4), [2, 0]),  # gather along a middle axis
        (-1, (6, 3, 4), [3, 0, 1]),  # negative axis normalizes to the last axis
    ],
    ids=["axis0_2d_index", "axis1_1d_index", "neg_axis_last"],
)
def test_gather_evaluate(axis, x_shape, idx):
    torch.manual_seed(0)
    x = torch.randn(*x_shape)
    idx_t = torch.tensor(idx, dtype=torch.int32)
    run_eval_case(EvalCase("", Gather(axis=axis), (x, idx_t), _gather_ref(x, axis, idx_t)))


TYPEINFER_CASES = [
    TypeInferCase(
        "neg_axis_normalizes",
        Gather(axis=-1),
        (ten((2, 3, 4), DType.f32), ten((2,), DType.i32)),
        ten((2, 3, 2), DType.f32),
    ),
    TypeInferCase(
        "axis_out_of_range",
        Gather(axis=5),
        (ten((2, 3, 4), DType.f32), ten((2,), DType.i32)),
        ExpectedError(match="out of range", exc=TypeError),
    ),
]


@pytest.mark.parametrize("case", TYPEINFER_CASES, ids=lambda c: c.name)
def test_gather_typeinfer(case):
    run_typeinfer_case(case)
