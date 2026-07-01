"""Verify hard constraints on LetStmt."""
from __future__ import annotations

import pytest

from tilefoundry.ir.core import Call, Var, VerifyError
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.memory.ptr_of import PtrOf
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import LetStmt, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, TensorType


def _t() -> TensorType:
    return TensorType(shape=(4,), dtype=DType.f32, layout=None, storage="rmem")


def _alloc_call(t: TensorType) -> Call:
    return Call(type=t, target=AllocTensor(tensor_type=t), args=())


def test_letstmt_rejects_reused_var_nested():
    """Binding the same Var object in an outer+inner Let must raise."""
    v = Var(type=_t(), name="v")
    inner_let = LetStmt(
        var=v,
        value=_alloc_call(_t()),
        body=Sequential(body=(Return(),)),
    )
    outer_let = LetStmt(
        var=v,
        value=_alloc_call(_t()),
        body=Sequential(body=(inner_let,)),
    )
    pf = PrimFunction(
        name="fn",
        params=(),
        body=Sequential(body=(outer_let,)),
    )
    with pytest.raises(VerifyError, match="fresh Var"):
        verify_prim_function(pf)


def test_letstmt_rejects_reused_var_sibling():
    """Two sibling LetStmts in the same Sequential rebinding the
    same Var instance must also raise — fresh-Var applies across the whole
    function, not merely within the current lexical scope."""
    v = Var(type=_t(), name="v")
    first_let = LetStmt(
        var=v,
        value=_alloc_call(_t()),
        body=Sequential(body=(Return(),)),
    )
    second_let = LetStmt(
        var=v,
        value=_alloc_call(_t()),
        body=Sequential(body=(Return(),)),
    )
    pf = PrimFunction(
        name="fn",
        params=(),
        body=Sequential(body=(first_let, second_let)),
    )
    with pytest.raises(VerifyError, match="fresh Var"):
        verify_prim_function(pf)


def test_letstmt_rejects_type_mismatch():
    """var.type must equal type_of(value)."""
    t_reg = _t()
    t_shared = TensorType(shape=(4,), dtype=DType.f32, layout=None, storage="smem")
    v = Var(type=t_shared, name="v")  # declared shared
    let = LetStmt(
        var=v,
        value=_alloc_call(t_reg),  # value type is reg
        body=Sequential(body=(Return(),)),
    )
    pf = PrimFunction(
        name="fn",
        params=(),
        body=Sequential(body=(let,)),
    )
    with pytest.raises(VerifyError, match="!= value.type"):
        verify_prim_function(pf)


def test_letstmt_rejects_alloc_nested_in_other_expr():
    """Call(AllocTensor, ...) may only appear directly as
    LetStmt.value, never nested inside another Expr operand."""
    # Build a PtrOf Call with an AllocTensor Call as its input — illegal.

    t_scalar = TensorType.scalar(DType.f32)
    v = Var(type=t_scalar, name="v")
    # Illegal nesting: PtrOf(AllocTensor(...)).
    nested = Call(
        type=t_scalar,
        target=PtrOf(),
        args=(Call(type=t_scalar, target=AllocTensor(tensor_type=t_scalar), args=()),),
    )
    let = LetStmt(
        var=v,
        value=nested,
        body=Sequential(body=(Return(),)),
    )
    pf = PrimFunction(
        name="fn",
        params=(),
        body=Sequential(body=(let,)),
    )
    with pytest.raises(VerifyError, match="AllocTensor"):
        verify_prim_function(pf)
