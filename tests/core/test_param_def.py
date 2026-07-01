"""ParamDef descriptor — minimal contract + override semantics."""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef


def test_paramdef_field_semantics() -> None:
    """Required vs default vs optional axes are independent."""
    required = ParamDef(kind="input")
    assert required.is_required and not required.has_default

    optional_required = ParamDef(kind="input", optional=True)
    assert optional_required.is_required  # nullable type, still required at call-site

    omittable = ParamDef(kind="attribute", default=0)
    assert not omittable.is_required and omittable.has_default

    with pytest.raises(ValueError):
        ParamDef(kind="output")  # type: ignore[arg-type]


def test_paramdef_subclass_field_override_wins() -> None:
    """Derived ``ParamDef`` redeclaration overrides base."""

    class _Base(Op):
        a = ParamDef(kind="input", annotation=int)
        b = ParamDef(kind="attribute", annotation=int)

    class _Child(_Base):
        a = ParamDef(kind="input", annotation=float)
        c = ParamDef(kind="attribute", annotation=str)

    names = [p.name for p in _Child.params()]
    assert names == ["a", "b", "c"]  # base field positions preserved
    types = {p.name: p.type for p in _Child.params()}
    assert types["a"] is float and types["b"] is int and types["c"] is str
