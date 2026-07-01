"""``ParamDef.annotation``-driven layout-sugar dispatch.

Sugar dispatch for a call-arg is annotation-driven: the receiving
``ParamDef.annotation`` decides which sugar parser (if any) fires, and
the annotation hint wins over the legacy ``attr_name == "layout"``
heuristic.

NOTE — internal-dispatch coverage, not surface-reachable.
The uniform parser-test format drives ``parse_script("...")`` and asserts
the resulting IR. These cases cannot use that path: the behavior under
test is the ``annotation=Layout`` dispatch branch of
``_eval_static_or_sugar``, and **no real Op in non-test source declares
``annotation=Layout``** (the only Layout-family annotation on a live Op
is ``Reshard.layout`` with ``annotation=ShardLayout``). Driving these
from a real ``@func`` source would change *which* annotation is
exercised — i.e. alter the assertion intent. Per the conversion
guidance, the internal dispatch path is exercised directly through
synthetic OpSchemas and kept with this comment so coverage of the
annotation-over-name priority rule is not lost.
"""

from __future__ import annotations

import ast

import pytest

from tilefoundry.ir.core.op_registry import _register_schema, _schemas_by_dialect_name
from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.parser.base import BaseExprVisitor
from tilefoundry.parser.symtab import LexicalEnv


class _DummyVisitor(BaseExprVisitor):
    token = "hir"  # type: ignore[assignment]


@pytest.fixture
def visitor() -> _DummyVisitor:
    return _DummyVisitor(env=LexicalEnv(), closure={})


@pytest.fixture
def tmp_op_schema():
    """Factory: register an OpSchema for the test, restore on teardown."""
    snapshots: list[tuple[tuple[str, str], list]] = []

    def _make(*, name: str, signature: tuple, attr_names: tuple[str, ...]):
        for pd, n in zip(signature, attr_names):
            pd._attr_name = n

        class _Builder:
            def __init__(self, **kw): ...

        op_cls = type(name, (), {"dialect": "tf", "name": name})
        schema = OpSchema(
            name=name, dialect="tf", category="test",
            signature=signature, builder=_Builder, op_class=op_cls,
        )
        # Mirror schema onto the class so ``_lookup_param_annotation``
        # resolves through ``op_cls._op_schema``.
        op_cls._op_schema = schema
        key = ("tf", name)
        snapshots.append((key, list(_schemas_by_dialect_name.get(key, ()))))
        _register_schema(schema)
        return op_cls

    yield _make
    for key, prev in snapshots:
        if prev:
            _schemas_by_dialect_name[key] = prev
        else:
            _schemas_by_dialect_name.pop(key, None)


def test_annotation_drives_sugar_on_non_layout_attr_name(visitor, tmp_op_schema) -> None:
    """A ParamDef named ``target`` with ``annotation=Layout`` triggers
    layout sugar (sugar dispatch is annotation-driven, not name-driven)."""
    op_cls = tmp_op_schema(
        name="anno_drives_sugar",
        signature=(
            ParamDef(kind="input"),
            ParamDef(kind="attribute", annotation=Layout),
        ),
        attr_names=("x", "target"),
    )
    node = ast.parse("(1, 1536)", mode="eval").body
    result = visitor._eval_static_or_sugar("target", node, op_cls=op_cls)
    assert isinstance(result, Layout)
    assert result.shape == (1, 1536)


def test_annotation_path_overrides_legacy_layout_name_heuristic(
    visitor, tmp_op_schema
) -> None:
    """When ``annotation=Layout`` is set on an attribute literally named
    ``layout``, the annotation-driven path wins (plain ``Layout``,
    not the legacy ``ShardLayout`` shortcut)."""
    op_cls = tmp_op_schema(
        name="anno_priority",
        signature=(
            ParamDef(kind="input"),
            ParamDef(kind="attribute", annotation=Layout),
        ),
        attr_names=("x", "layout"),
    )
    node = ast.parse("(1, 1536)", mode="eval").body
    result = visitor._eval_static_or_sugar("layout", node, op_cls=op_cls)
    assert type(result) is Layout  # not ShardLayout


def test_no_annotation_no_sugar_for_non_layout_name(visitor, tmp_op_schema) -> None:
    """No annotation hint + attr name isn't ``layout`` → plain tuple stays a tuple."""
    op_cls = tmp_op_schema(
        name="anno_no_sugar",
        signature=(
            ParamDef(kind="input"),
            ParamDef(kind="attribute"),  # default annotation=object
        ),
        attr_names=("x", "target"),
    )
    node = ast.parse("(1, 1536)", mode="eval").body
    assert visitor._eval_static_or_sugar("target", node, op_cls=op_cls) == (1, 1536)
