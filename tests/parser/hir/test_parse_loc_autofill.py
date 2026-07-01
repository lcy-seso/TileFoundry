"""``Call.loc`` auto-fill — LHS-name vs explicit override."""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem

# ── Single-Name LHS sets `loc` to the LHS name ----------------------------


@func
def _single_name(
    a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"],
) -> Tensor[(1, 4), "f32"]:
    sum_ab = add(a, b)
    return sum_ab


# ── Tuple-unpack: parent Call gets DSL callable name, TGI gets LHS name ----


@func
def _tuple_unpack(x: Tensor[(1, 1536), "bf16"]) -> Tensor[(1, 1536), "fp8e4m3"]:
    x_fp8, x_scale = quant(x)
    return x_fp8


# ── Explicit `loc=` wins over LHS-name auto-fill ---------------------------


@func
def _explicit_loc(
    a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"],
) -> Tensor[(1, 4), "f32"]:
    summed = add(a, b, loc="custom_tag")
    return summed


def test_loc_autofill_lhs_name_and_explicit_override() -> None:
    """Single-Name LHS sets ``loc`` to the LHS name; explicit ``loc=`` wins.

    For tuple-unpack the parent Call gets the DSL callable name
    (``"quant"``) and each ``TupleGetItem`` gets its LHS name.
    """
    # Single-Name LHS
    fn = _single_name
    body = fn.body
    assert isinstance(body, Call) and body.loc == "sum_ab"

    # Tuple unpack: parent loc == "quant", TupleGetItem loc == LHS name.
    fn = _tuple_unpack
    body = fn.body
    assert isinstance(body, Call) and isinstance(body.target, TupleGetItem)
    assert body.loc == "x_fp8"
    parent = body.args[0]
    assert isinstance(parent, Call) and parent.loc == "quant"

    # Explicit `loc=` wins over LHS-name auto-fill.
    fn = _explicit_loc
    body = fn.body
    assert isinstance(body, Call) and body.loc == "custom_tag"
