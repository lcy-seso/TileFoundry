"""Unary typeinfer: shape / dtype / layout / storage pass through the input,
including a sharded input's ``ShardLayout``.
"""
from __future__ import annotations

import pytest
import torch

from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, Split
from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
    ten,
    tensor_grid,
)

_NEG = Unary(kind=UnaryKind.NEG)
_NOT = Unary(kind=UnaryKind.NOT)


CASES = [
    TypeInferCase(name="passthrough", op=_NEG, inputs=(t,), expected=t)
    for t in tensor_grid((4, 8), DType.f32)
] + [
    TypeInferCase(
        name="not_requires_bool",
        op=_NOT,
        inputs=(ten((4, 8), DType.f32),),
        expected=ExpectedError(match="bool"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_unary_typeinfer(case):
    run_typeinfer_case(case)


def test_unary_passes_sharded_layout_through():
    mesh = Mesh(
        topology="gpu",
        layout=Layout(shape=(4,), strides=(1,)),
        names=("g",),
        topologies=("gpu",),
    )
    sl = ShardLayout(
        layout=Layout(shape=(16, 8), strides=(8, 1)),
        attrs=(Split(0),),
        mesh=mesh,
    )
    x = TensorType(shape=(16, 8), dtype=DType.f32, layout=sl, storage="gmem")
    out = infer_call(_NEG, x)
    assert out.layout is sl
    assert out.shape == (16, 8)


@pytest.mark.parametrize(
    "kind,ref",
    [
        (UnaryKind.NEG, lambda x: -x),
        (UnaryKind.ABS, lambda x: x.abs()),
        (UnaryKind.SQUARE, lambda x: x.square()),
    ],
    ids=["neg", "abs", "square"],
)
def test_unary_evaluate(kind, ref):
    torch.manual_seed(0)
    x = torch.randn(4)
    run_eval_case(EvalCase(kind.name.lower(), Unary(kind=kind), (x,), ref(x)))
