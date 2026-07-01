"""Round-trip tests: print → parse → structural equality."""

import pytest

from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Constant, Op, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import TensorType
from tilefoundry.parser.hir_parser import parse_script
from tests.fixtures.demo_ir import build_demo


def _is_op(obj) -> bool:
    """Check if obj is a tilefoundry Op instance."""
    return isinstance(obj, Op)


def _attr_equal(a, b, path="") -> bool:
    """Compare two attribute values (may be nested dataclasses or Ops without __eq__)."""
    if a is b:
        return True
    if type(a) is not type(b):
        print(f"MISMATCH attr type at {path}: {type(a).__name__} vs {type(b).__name__}")
        return False
    # Handle tuples (e.g. attrs)
    if isinstance(a, tuple):
        if len(a) != len(b):
            print(f"MISMATCH tuple len at {path}: {len(a)} vs {len(b)}")
            return False
        return all(_attr_equal(aa, bb, f"{path}[{i}]") for i, (aa, bb) in enumerate(zip(a, b)))
    # Handle Op instances — compare via params()
    if _is_op(a):
        for pi in type(a).params():
            av = getattr(a, pi.name, None)
            bv = getattr(b, pi.name, None)
            if not _attr_equal(av, bv, f"{path}.{pi.name}"):
                return False
        return True
    # Handle dataclass-like objects (check all public fields)
    if hasattr(a, "__dataclass_fields__"):
        for f_name in a.__dataclass_fields__:
            av = getattr(a, f_name, None)
            bv = getattr(b, f_name, None)
            if not _attr_equal(av, bv, f"{path}.{f_name}"):
                return False
        return True
    # Fallback to equality
    return a == b


def _structural_equal(a, b, path="") -> bool:
    """Compare two HIR expressions for structural equality."""
    if type(a) is not type(b):
        print(f"MISMATCH type at {path}: {type(a).__name__} vs {type(b).__name__}")
        return False

    # Handle Function
    if isinstance(a, Function):
        if a.name != b.name:
            print(f"MISMATCH name at {path}: {a.name} vs {b.name}")
            return False
        if len(a.params) != len(b.params):
            return False
        for i, (pa, pb) in enumerate(zip(a.params, b.params)):
            if not _structural_equal(pa, pb, f"{path}.params[{i}]"):
                return False
        if not _structural_equal(a.body, b.body, f"{path}.body"):
            return False
        if not _structural_equal(a.return_type, b.return_type, f"{path}.return_type"):
            return False
        return True

    if isinstance(a, Var):
        if a.name != b.name:
            print(f"MISMATCH Var.name at {path}: {a.name} vs {b.name}")
            return False
        return _structural_equal(a.type, b.type, f"{path}.type")

    if isinstance(a, Constant):
        return a.value == b.value

    if isinstance(a, Call):
        if type(a.target) is not type(b.target):
            print(f"MISMATCH target at {path}: {type(a.target).__name__} vs {type(b.target).__name__}")
            return False
        if len(a.args) != len(b.args):
            print(f"MISMATCH arg count at {path}: {len(a.args)} vs {len(b.args)}")
            return False
        for i, (aa, bb) in enumerate(zip(a.args, b.args)):
            if not _structural_equal(aa, bb, f"{path}.args[{i}]"):
                return False
        # Compare op attributes (Op dataclasses may not have __eq__)
        for pi in type(a.target).params():
            if pi.kind == "attribute":
                av = getattr(a.target, pi.name, None)
                bv = getattr(b.target, pi.name, None)
                if not _attr_equal(av, bv, f"{path}.attr.{pi.name}"):
                    return False
        return _structural_equal(a.type, b.type, f"{path}.type")

    if isinstance(a, TensorType):
        if a.shape != b.shape:
            print(f"MISMATCH shape at {path}: {a.shape} vs {b.shape}")
            return False
        if a.dtype != b.dtype:
            print(f"MISMATCH dtype at {path}: {a.dtype} vs {b.dtype}")
            return False
        if a.storage != b.storage:
            print(f"MISMATCH storage at {path}: {a.storage} vs {b.storage}")
            return False
        if not _attr_equal(a.layout, b.layout, f"{path}.layout"):
            return False
        return True

    return a == b


class TestRoundTrip:
    def test_demo_ir_roundtrip_with_keyword_attrs(self):
        """print(build_demo()) → parse_script → structural equal."""
        fn, _, _ = build_demo()
        src = as_script(fn)
        fn2 = parse_script(src)
        assert _structural_equal(fn, fn2), "round-trip mismatch with keyword attrs"

    def test_demo_ir_roundtrip_with_positional_attrs(self):
        """reshard(a, shared_layout, storage) without layout= keyword."""
        src = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "from tilefoundry.ir.types.shard import (\n"
            "    B, S, P, Layout, Mesh, Layout, ShardLayout, Topology,\n"
            ")\n"
            "shared_layout = ShardLayout(\n"
            "    layout=Layout((1, 1536), (1536, 1)),\n"
            "    attrs=(),\n"
            '    mesh=Mesh(Topology("cta", 128), Layout((128,), (1,))),\n'
            ")\n"
            "\n"
            "@func\n"
            'def test_pos(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:\n'
            '    b = reshard(a, shared_layout, "smem")\n'
            "    return b\n"
        )
        fn = parse_script(src)
        assert fn.name == "test_pos"
        # Check that the body is a reshard with correct args
        body = fn.body
        assert isinstance(body, Call)
        assert isinstance(body.target, Reshard)
        assert body.target.storage == StorageKind.SMEM
        # layout attribute should be set
        assert body.target.layout is not None

    def test_positional_and_keyword_equivalent(self):
        """reshard(a, shared_layout) ≡ reshard(a, layout=shared_layout)."""
        src_pos = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "from tilefoundry.ir.types.shard import (\n"
            "    B, S, P, Layout, Mesh, Layout, ShardLayout, Topology,\n"
            ")\n"
            "sl = ShardLayout(\n"
            "    layout=Layout((1, 1536), (1536, 1)),\n"
            "    attrs=(),\n"
            '    mesh=Mesh(Topology("cta", 128), Layout((128,), (1,))),\n'
            ")\n"
            "\n"
            "@func\n"
            'def pos(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:\n'
            "    b = reshard(a, sl)\n"
            "    return b\n"
        )
        src_kw = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "from tilefoundry.ir.types.shard import (\n"
            "    B, S, P, Layout, Mesh, Layout, ShardLayout, Topology,\n"
            ")\n"
            "sl = ShardLayout(\n"
            "    layout=Layout((1, 1536), (1536, 1)),\n"
            "    attrs=(),\n"
            '    mesh=Mesh(Topology("cta", 128), Layout((128,), (1,))),\n'
            ")\n"
            "\n"
            "@func\n"
            'def kw(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:\n'
            "    b = reshard(a, layout=sl)\n"
            "    return b\n"
        )
        fn_pos = parse_script(src_pos)
        fn_kw = parse_script(src_kw)
        # Compare bodies structurally (ignore function name)
        assert _structural_equal(fn_pos.body, fn_kw.body), \
            "positional and keyword should produce equivalent IR"

    def test_too_many_positional_args_errors(self):
        """reshard(a, sl, 'smem', 1, 'extra') — 5 positional, only 4 params."""

        src = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "from tilefoundry.ir.types.shard import *\n"
            "sl = ShardLayout(layout=Layout((1,), (1,)), attrs=(), mesh=Mesh(Topology('c',1), Layout((1,),(1,))))\n"
            "@func\n"
            'def f(a: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:\n'
            '    b = reshard(a, sl, "smem", 1, "extra")\n'
            "    return b\n"
        )
        with pytest.raises(VerifyError, match="too many positional"):
            parse_script(src)

    def test_duplicate_positional_and_keyword_errors(self):
        """reshard(a, sl, layout=sl2) — duplicate binding for layout."""

        src = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "from tilefoundry.ir.types.shard import *\n"
            "sl = ShardLayout(layout=Layout((1,), (1,)), attrs=(), mesh=Mesh(Topology('c',1), Layout((1,),(1,))))\n"
            "@func\n"
            'def f(a: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:\n'
            "    b = reshard(a, sl, layout=sl)\n"
            "    return b\n"
        )
        with pytest.raises(VerifyError, match="duplicate binding"):
            parse_script(src)

    def test_wrong_attr_type_errors(self):
        """reshard(a, 123) — int is not a ShardLayout, fails in typeinfer."""

        src = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "@func\n"
            'def f(a: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:\n'
            "    b = reshard(a, 123)\n"
            "    return b\n"
        )
        with pytest.raises(VerifyError, match="ShardLayout"):
            parse_script(src)

    def test_missing_input_errors(self):
        """reshard() — missing required input x and layout, TypeError."""
        src = (
            "from __future__ import annotations\n"
            "from tilefoundry import func\n"
            "from tilefoundry.dsl.tf import *\n"
            "from tilefoundry.dsl import Tensor\n"
            "@func\n"
            'def f(a: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:\n'
            "    b = reshard()\n"
            "    return b\n"
        )
        with pytest.raises((TypeError,)):
            parse_script(src)
