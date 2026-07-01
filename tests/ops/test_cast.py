"""Cast typeinfer: dtype changes; shape / storage / layout pass through. A
sharded input keeps its ShardLayout (Cast's relation is the identity)."""
from __future__ import annotations

import pytest
import torch

from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, Split
from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
    ten,
)


def _mesh() -> Mesh:
    return Mesh(
        topology="gpu",
        layout=Layout(shape=(4,), strides=(1,)),
        names=("g",),
        topologies=("gpu",),
    )


_M = _mesh()

CASES = [
    TypeInferCase(
        name="unsharded_dtype_change",
        op=Cast(dtype=DType.bf16),
        inputs=(ten((4, 8), DType.f32),),
        expected=ten((4, 8), DType.bf16),
    ),
    TypeInferCase(
        name="rank0",
        op=Cast(dtype=DType.f32),
        inputs=(ten((), DType.i32),),
        expected=ten((), DType.f32),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_cast_typeinfer(case):
    run_typeinfer_case(case)


def test_cast_carries_sharded_layout():
    sl = ShardLayout(
        layout=Layout(shape=(16, 8), strides=(8, 1)),
        attrs=(Split(axis=0),),
        mesh=_M,
    )
    x = TensorType(shape=(16, 8), dtype=DType.f32, layout=sl, storage="gmem")
    out = infer_call(Cast(dtype=DType.bf16), x)
    assert out.dtype == DType.bf16
    assert out.shape == (16, 8)
    assert out.layout == sl  # identity relation -> same ShardLayout


def test_cast_evaluate():
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    run_eval_case(
        EvalCase("to_bf16", Cast(dtype=DType.bf16), (x,), x.to(torch.bfloat16), atol=2e-2, rtol=2e-2)
    )
