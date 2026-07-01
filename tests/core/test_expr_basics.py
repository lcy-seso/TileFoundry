"""Spec 001 core-ir — Expr / Call leaf invariants + Op singleton cache."""

from __future__ import annotations

from tilefoundry.ir.core import Call, Constant, Op, Var
from tilefoundry.ir.types import DType, TensorType


def _t() -> TensorType:
    return TensorType.scalar(DType.f32)


def test_expr_leaf_and_call_construct() -> None:
    """Var carries name+type, Constant carries value+type, Call wraps Op+args."""
    class _OpA(Op):
        pass

    v = Var(type=_t(), name="x")
    c = Constant(type=_t(), value=1.0)
    call = Call(type=_t(), target=_OpA(), args=(v,))
    assert v.name == "x" and c.value == 1.0
    assert isinstance(call.target, _OpA) and call.args == (v,)


def test_op_attribute_singleton_cache() -> None:
    """No-attribute Ops are cached — ``Foo() is Foo()`` (spec 001)."""
    class _OpB(Op):
        pass

    assert _OpB() is _OpB()
