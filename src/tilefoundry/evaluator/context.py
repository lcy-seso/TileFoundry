"""Eval context handed to a ``@register_eval`` handler.

Spec: evaluator.md §3
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalContext:
    """Operands + op + result type for one Op evaluation.

    ``args`` are the already-evaluated operands in ``Call.args`` order; an
    operand is a ``Value`` — concretely a ``TensorValue`` (single output) or
    a ``TupleValue`` (multi-output). Op attributes are read off ``op`` as
    fields (e.g. ``ctx.op.kind``).
    """

    op: Any
    args: tuple[Any, ...]
    result_type: Any
    device: str = "cpu"
