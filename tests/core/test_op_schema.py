"""OpSchema — pure / effect-ful contract."""

from __future__ import annotations

from tilefoundry.ir.core.op_schema import OpEffect, OpSchema
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor


class _FakeOp:
    pass


def test_opschema_pure_vs_effectful() -> None:
    pd_a = ParamDef(kind="input", pattern=Tensor)
    pd_b = ParamDef(kind="input", pattern=Tensor)

    pure = OpSchema(
        name="add", dialect="tf", category="math",
        signature=(pd_a, pd_b), builder=_FakeOp, op_class=_FakeOp,
    )
    assert pure.is_pure and pure.effects == ()

    eff = OpSchema(
        name="relu", dialect="T", category="nn",
        signature=(pd_a, pd_b), builder=_FakeOp, op_class=_FakeOp,
        effects=(OpEffect(kind="write", param_index=1),),
    )
    assert not eff.is_pure
