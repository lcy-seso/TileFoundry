"""Exp evaluator value oracle."""
from __future__ import annotations

import torch

from tilefoundry.ir.hir.math.exp import Exp
from tests.ops.eval_utils import EvalCase, run_eval_case


def test_exp_evaluate():
    torch.manual_seed(0)
    x = torch.rand(4) + 0.5
    run_eval_case(EvalCase("exp", Exp(), (x,), torch.exp(x), atol=1e-6))
