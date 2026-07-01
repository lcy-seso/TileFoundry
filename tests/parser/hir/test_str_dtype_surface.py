"""String dtype / reduce-kind authoring surface (parser.md §2.4).

The DSL surface accepts the string form (`dtype="f32"`, `kind="sum"`); the
parser normalizes it to the IR-canonical enum at the call boundary, and an
unknown string raises a clear error. The string and enum forms parse to the
same IR.
"""

from __future__ import annotations

import textwrap

import pytest

from tilefoundry.inspection import as_script
from tilefoundry.ir.core import VerifyError
from tilefoundry.parser.hir_parser import parse_script


def _dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


_HEADER = """
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *
"""


def test_string_dtype_parses() -> None:
    src = _HEADER + """
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
    return cast(x, dtype="bf16")
"""
    fn = parse_script(_dedent(src))
    assert fn.body is not None


def test_string_reduce_kind_parses() -> None:
    src = _HEADER + """
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind="sum")
"""
    fn = parse_script(_dedent(src))
    assert fn.body is not None


def test_string_and_enum_forms_are_equivalent() -> None:
    str_form = _HEADER + """
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind="sum")
"""
    enum_form = _HEADER + """
from tilefoundry.ir.core.kinds import ReduceKind
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind=ReduceKind.SUM)
"""
    assert as_script(parse_script(_dedent(str_form))) == as_script(
        parse_script(_dedent(enum_form))
    )


def test_invalid_dtype_string_raises() -> None:
    src = _HEADER + """
@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
    return cast(x, dtype="float32")
"""
    with pytest.raises(VerifyError, match=r"DType: unknown value 'float32'"):
        parse_script(_dedent(src))


def test_invalid_reduce_kind_string_raises() -> None:
    src = _HEADER + """
@func
def g(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
    return reduce(x, axes=(0,), keepdim=True, kind="plus")
"""
    with pytest.raises(VerifyError, match=r"ReduceKind: unknown value 'plus'"):
        parse_script(_dedent(src))
