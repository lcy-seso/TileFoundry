"""Rsqrt evaluator value oracle."""
from __future__ import annotations

import torch

from tilefoundry.ir.hir.math.rsqrt import Rsqrt
from tests.ops.eval_utils import EvalCase, run_eval_case


def test_rsqrt_evaluate():
    torch.manual_seed(0)
    x = torch.rand(4) + 0.5
    run_eval_case(EvalCase("rsqrt", Rsqrt(), (x,), torch.rsqrt(x), atol=1e-6))
