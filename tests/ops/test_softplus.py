"""Softplus evaluator value oracle."""
from __future__ import annotations

import torch

from tilefoundry.ir.hir.math.softplus import Softplus
from tests.ops.eval_utils import EvalCase, run_eval_case


def test_softplus_evaluate():
    torch.manual_seed(0)
    x = torch.rand(4) + 0.5
    run_eval_case(EvalCase("softplus", Softplus(), (x,), torch.nn.functional.softplus(x), atol=1e-6))
