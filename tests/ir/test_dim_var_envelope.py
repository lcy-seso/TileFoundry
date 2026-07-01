"""HIR ``verify_function`` enforcement of ``DimVar`` envelope rules.

Covers:

- ``DimVarRangePat`` ⊆ ``DimVar`` envelope.
- ``DimVarRangePat`` referencing an unknown ``DimVar`` name.
- ``_DimVarMeta`` no longer raises for same-name distinct-bounds
  construction (conflict scoped to a single signature, surfaced
  only in ``verify_function``).
- Same-name ``DimVar`` instances within one signature must agree on
  bounds.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.tensor_type import TupleType


def _tensor(shape) -> TensorType:
    return TensorType(shape=shape, dtype=DType.f32, layout=None, storage="gmem")


def _identity_fn(
    *,
    name: str,
    params: tuple[Var, ...],
    specializations: tuple = (),
) -> HirFunction:
    return HirFunction.build(
        name=name,
        params=params,
        body=params[0],
        return_type=params[0].type,
        specializations=specializations,
    )


def test_specialization_outside_envelope_raises() -> None:
    s = DimVar(name="S_env_out", lo=1, hi=8)
    x = Var(type=_tensor((s,)), name="x")
    fn = _identity_fn(
        name="f",
        params=(x,),
        specializations=(DimVarRangePat("S_env_out", 0, 100),),
    )
    with pytest.raises(VerifyError, match="not contained in DimVar envelope"):
        verify_function(fn)


def test_specialization_unknown_dim_var_raises() -> None:
    s = DimVar(name="S_env_known", lo=1, hi=8)
    x = Var(type=_tensor((s,)), name="x")
    fn = _identity_fn(
        name="f",
        params=(x,),
        specializations=(DimVarRangePat("OTHER", 1, 4),),
    )
    with pytest.raises(VerifyError, match="references unknown DimVar"):
        verify_function(fn)


def test_dim_var_cross_function_same_name_distinct_bounds_OK() -> None:
    # Construction must not raise even with same-name distinct-bounds.
    a = DimVar(name="S_cross", lo=1, hi=8)
    b = DimVar(name="S_cross", lo=100, hi=200)
    assert a is not b

    xa = Var(type=_tensor((a,)), name="x")
    xb = Var(type=_tensor((b,)), name="x")
    fn_a = _identity_fn(
        name="fa",
        params=(xa,),
        specializations=(DimVarRangePat("S_cross", 1, 4),),
    )
    fn_b = _identity_fn(
        name="fb",
        params=(xb,),
        specializations=(DimVarRangePat("S_cross", 150, 200),),
    )
    verify_function(fn_a)
    verify_function(fn_b)


def test_dim_var_within_function_inconsistent_bounds_raises() -> None:
    s_small = DimVar(name="S_inc", lo=1, hi=8)
    s_large = DimVar(name="S_inc", lo=4, hi=16)
    x = Var(type=_tensor((s_small,)), name="x")
    y = Var(type=_tensor((s_large,)), name="y")
    fn = _identity_fn(name="f", params=(x, y))
    with pytest.raises(VerifyError, match="inconsistent DimVar bounds for 'S_inc'"):
        verify_function(fn)


def test_return_type_dim_var_inconsistent_with_params_raises() -> None:
    """Same-name DimVar in return_type with different bounds must also raise.

    The signature-conflict check scans params AND return_type. A
    function whose return_type carries a DimVar with bounds that
    disagree with the same-name DimVar in the params is rejected at
    parse/verify time, not silently accepted.
    """
    s_params = DimVar(name="S_ret", lo=1, hi=8)
    s_return = DimVar(name="S_ret", lo=4, hi=16)
    x = Var(type=_tensor((s_params,)), name="x")
    fn = HirFunction.build(
        name="f",
        params=(x,),
        body=x,
        return_type=_tensor((s_return,)),
    )
    with pytest.raises(VerifyError, match="inconsistent DimVar bounds for 'S_ret'"):
        verify_function(fn)


def test_return_only_specialization_raises() -> None:
    """Specialization must anchor to a DimVar in params, not return only.

    DispatchCall.subject lowers to ShapeOf(param, axis); a DimVar
    that appears only in the return_type cannot be referenced at
    runtime.
    """
    r = DimVar(name="R_ret_only", lo=1, hi=8)
    static_x = Var(type=_tensor((4,)), name="x")
    fn = HirFunction.build(
        name="f",
        params=(static_x,),
        body=static_x,
        return_type=_tensor((r,)),
        specializations=(DimVarRangePat("R_ret_only", 1, 4),),
    )
    with pytest.raises(VerifyError, match="references unknown DimVar"):
        verify_function(fn)


def _dispatch_proto(name: str, env, ranges):
    """A dispatch prototype over ``DimVar(name, *env)`` with one variant per
    ``(lo, hi)`` in *ranges*. Used to exercise the half-open-interval partition
    verifier (`_verify_partition`)."""
    s = DimVar(name=name, lo=env[0], hi=env[1])
    ty = _tensor((s,))
    base = HirFunction.build(name="g", params=(Var(type=ty, name="x"),),
                             body=None, return_type=ty)
    for lo, hi in ranges:
        x = Var(type=ty, name="x")
        base.add_variant(HirFunction.build(
            name="g", params=(x,), body=x, return_type=ty,
            specializations=(DimVarRangePat(name, lo, hi),),
        ))
    return base


def test_partition_complete_and_disjoint_ok() -> None:
    # [1,5) + [5,8) exactly tile the half-open envelope [1,8).
    verify_function(_dispatch_proto("S_par_ok", (1, 8), [(1, 5), (5, 8)]))


def test_partition_single_point_variant_ok() -> None:
    # A single-point variant [4,5) (= the value 4) is a legal half-open range.
    verify_function(_dispatch_proto("S_par_pt", (1, 8), [(1, 4), (4, 5), (5, 8)]))


def test_partition_overlap_raises() -> None:
    # [1,5) and [4,8) overlap at 4 (next must start at 5).
    with pytest.raises(VerifyError, match="gap or overlap at 4"):
        verify_function(_dispatch_proto("S_par_ov", (1, 8), [(1, 5), (4, 8)]))


def test_partition_gap_raises() -> None:
    # [1,3) then [5,8) leaves 3,4 uncovered.
    with pytest.raises(VerifyError, match="gap or overlap at 5"):
        verify_function(_dispatch_proto("S_par_gap", (1, 8), [(1, 3), (5, 8)]))


def test_partition_incomplete_raises() -> None:
    # [1,5) + [5,7) stop at 7 but the envelope reaches 8.
    with pytest.raises(VerifyError, match=r"cover \[1, 7\) but the envelope is \[1, 8\)"):
        verify_function(_dispatch_proto("S_par_inc", (1, 8), [(1, 5), (5, 7)]))


def test_dim_var_range_pat_match_is_half_open() -> None:
    # [1, 4) matches 1..3 and rejects 4 (hi exclusive).
    pat = DimVarRangePat("c", 1, 4)
    assert pat.match(1) and pat.match(3)
    assert not pat.match(4)
    assert not pat.match(0)


def test_dim_var_range_pat_rejects_empty_range() -> None:
    # lo == hi is an empty half-open range; a single point is [k, k+1).
    with pytest.raises(ValueError, match="requires lo < hi"):
        DimVarRangePat("c", 4, 4)


def test_dim_var_rejects_empty_envelope() -> None:
    with pytest.raises(ValueError, match="require lo < hi"):
        DimVar(name="S_empty", lo=4, hi=4)


def test_return_tuple_type_inconsistent_dim_var_bounds_raises() -> None:
    """Same-name DimVar inside a TupleType field of return_type with
    different bounds is rejected (recursive consistency scan)."""
    s_params = DimVar(name="S_tup", lo=1, hi=8)
    s_return = DimVar(name="S_tup", lo=4, hi=16)
    x = Var(type=_tensor((s_params,)), name="x")
    tup_return = TupleType(fields=(_tensor((s_return,)),))
    fn = HirFunction.build(
        name="f",
        params=(x,),
        body=x,
        return_type=tup_return,
    )
    with pytest.raises(VerifyError, match="inconsistent DimVar bounds for 'S_tup'"):
        verify_function(fn)
