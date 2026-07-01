"""TopK typeinfer: returns (values, indices); the topk axis shrinks to k."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
    ten,
)

_BF = DType.bf16

CASES = [
    TypeInferCase(
        "default_axis_last",
        TopK(k=8),
        (ten((1, 128), _BF),),
        TupleType(fields=(ten((1, 8), _BF), ten((1, 8), DType.i32))),
    ),
    TypeInferCase(
        "explicit_axis",
        TopK(k=2, axis=1),
        (ten((4, 8, 16), _BF),),
        TupleType(fields=(ten((4, 2, 16), _BF), ten((4, 2, 16), DType.i32))),
    ),
    TypeInferCase(
        "axis_out_of_range",
        TopK(k=2, axis=5),
        (ten((4,), _BF),),
        ExpectedError(match="out of range", exc=TypeError),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_topk_typeinfer(case):
    run_typeinfer_case(case)


def test_topk_evaluate():
    torch.manual_seed(0)
    x = torch.randn(5, 16)
    ref_vals, ref_idx = torch.topk(x, 4, dim=-1)

    param = Var(
        type=TensorType(shape=(5, 16), dtype=DType.f32, layout=None, storage="gmem"),
        name="x",
    )
    call = Call(type=param.type, target=TopK(k=4, axis=-1), args=(param,))
    result_type = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=result_type)
    fn = Function.build(name="topk_case", params=(param,), body=call, return_type=result_type)
    vals, idx = evaluate(fn, x, device="cpu")
    torch.testing.assert_close(vals, ref_vals)
    torch.testing.assert_close(idx.long(), ref_idx)
