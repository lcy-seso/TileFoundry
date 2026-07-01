"""Tests for ``@register_alias`` — DSL surface alias schema.

Kinded sugar names (``add`` / ``sub`` / ...) resolve to a single alias
schema; there are no per-name legacy Op classes. Each kinded sugar
name has *exactly one* schema in the registry — the alias.

These tests lock in:

- alias schemas register with ``op_class=None`` and no legacy Op class;
- the alias builder constructs the right kinded ``Binary`` / ``Unary`` op;
- a bare / ``tf.*`` op-name in a ``@func`` body parses through the alias
  builder to the kinded IR op.
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import add as _tf_add  # noqa: F401  -- closure capture
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.op_registry import (
    _first_schema,
    get_op_by_name,
    get_schemas,
)
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.types import DType

# ── Registry shape ──────────────────────────────────────────────────────


def test_kinded_alias_registers_one_schema_no_legacy_op() -> None:
    """A kinded sugar name resolves to exactly one schema — the alias —
    with ``op_class=None``, a callable builder, and no legacy Op class."""
    schemas = get_schemas("tf", "add")
    assert len(schemas) == 1
    assert schemas[0].op_class is None

    s = _first_schema("tf", "add")
    assert s is not None
    assert s.op_class is None
    assert callable(s.builder)

    assert get_op_by_name("add") is None


def test_alias_builder_constructs_kinded_op() -> None:
    """The alias builder constructs the right kinded op, reusing the
    static ParamDef references for its signature — binary ``add`` and
    unary ``neg``."""
    binary = _first_schema("tf", "add")
    assert binary.signature == (Binary.lhs, Binary.rhs)
    add_inst = binary.builder()
    assert isinstance(add_inst, Binary)
    assert add_inst.kind is BinaryKind.ADD

    unary = _first_schema("tf", "neg")
    assert unary is not None and unary.op_class is None
    assert unary.signature == (Unary.x,)
    neg_inst = unary.builder()
    assert isinstance(neg_inst, Unary)
    assert neg_inst.kind is UnaryKind.NEG


def test_all_20_kinded_aliases_registered() -> None:
    """All 16 binary + 4 unary surface aliases land in the registry."""
    binary_names = (
        "add", "sub", "mul", "div", "floor_div", "mod", "min", "max",
        "cmp_eq", "cmp_ne", "cmp_lt", "cmp_le", "cmp_gt", "cmp_ge",
        "logical_and", "logical_or",
    )
    unary_names = ("neg", "abs", "logical_not", "square")
    for n in binary_names + unary_names:
        s = _first_schema("tf", n)
        assert s is not None, f"alias {n!r} not registered"
        assert s.op_class is None, f"alias {n!r} should have op_class=None"


# ── Parser end-to-end ──────────────────────────────────────────────────


@func
def _alias_call(
    a: Tensor[(8,), DType.f32], b: Tensor[(8,), DType.f32],
) -> Tensor[(8,), DType.f32]:
    # Bare ``add`` is bound by ``from tilefoundry.dsl.tf import add`` at
    # the top of this test file (closure capture).
    return _tf_add(a, b)


def test_bare_add_routes_through_alias() -> None:
    body = _alias_call.body
    assert isinstance(body, Call)
    assert isinstance(body.target, Binary)
    assert body.target.kind is BinaryKind.ADD
