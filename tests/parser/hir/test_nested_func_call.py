"""Nested ``@func`` call support.

Locks the DSL surface for nested ``@func`` → ``@func`` calls, the
``@register_typeinfer(Function)`` arg-contract handler, and the
viewer's inline-expansion of ``Call(target=hir.Function)``:

1. **Parser** produces ``Call(target=hir.Function, args=...)`` for a
   nested ``@func`` call site (no fixture-only inline fallback, no
   placeholder Op).
2. **TypeInfer** validates arg count + each arg's type against the
   callee's parameter types, and returns the callee's
   ``return_type``.
3. **Viewer** inline-expands the callee subgraph into the caller graph.

No GPU, no codegen, no runtime.
"""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.dsl.tf import (  # noqa: F401 — binds bare ``add``, ``mul``
    add,
    mul,
)
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.hir.function import Function as HirFunction

# ---------------------------------------------------------------------------
# Fixtures — two ``@func``s where the outer one calls the inner one.
# ---------------------------------------------------------------------------


N = DimVar("N", 1, 64)


@func
def _inner_double(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
    return add(x, x)  # noqa: F821 — bound via ``from tilefoundry.dsl.tf import *``


@func
def _outer_call_inner(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
    return _inner_double(x)


# ---------------------------------------------------------------------------
# Parser produces ``Call(target=hir.Function)``.
# ---------------------------------------------------------------------------


def test_parser_emits_call_with_hir_function_target() -> None:
    outer_ir = _outer_call_inner
    body = outer_ir.body
    # The outer body is the call expression directly (single-return form).
    assert isinstance(body, Call)
    assert isinstance(body.target, HirFunction), (
        f"expected Call.target to be hir.Function, got "
        f"{type(body.target).__name__}"
    )
    # The callee is the same canonical Function instance the inner
    # ``@func`` produced (no clone, no surrogate).
    inner_ir = _inner_double
    assert body.target is inner_ir


# ---------------------------------------------------------------------------
# TypeInfer threads callee return_type and enforces arg contract.
# ---------------------------------------------------------------------------


def test_call_type_matches_callee_return_type() -> None:
    outer_ir = _outer_call_inner
    inner_ir = _inner_double
    assert outer_ir.body.type == inner_ir.return_type


def test_arity_mismatch_rejected_at_parse_time() -> None:
    # The parser enforces the arity hard so we don't even reach
    # typeinfer with a malformed Call.
    with pytest.raises(VerifyError, match="arity mismatch"):

        @func
        def _bad_arity(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
            return _inner_double(x, x)  # type: ignore[call-arg]  # noqa: F841


def test_arg_type_mismatch_rejected_at_typeinfer() -> None:
    # Callee declares ``Tensor[(N,), "f32"]`` but caller passes
    # ``Tensor[(N,), "bf16"]`` — typeinfer must surface the
    # parameter-type mismatch.
    with pytest.raises(VerifyError, match="type mismatch"):

        @func
        def _bad_dtype(x: Tensor[(N,), "bf16"]) -> Tensor[(N,), "f32"]:
            return _inner_double(x)  # noqa: F841
