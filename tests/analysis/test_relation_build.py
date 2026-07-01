"""Forward relation domain builder — static / dynamic / affine / arity."""
from __future__ import annotations

import isl
import pytest

from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimFloorDiv, DimMul, DimVar
from tilefoundry.visitor_registry.access_relation import AccessRelationResult
from tilefoundry.visitor_registry.relation_build import (
    build_domain,
    shape_from_relation,
    validate_output_map_arity,
)

_I64 = TensorType.scalar(DType.i64)


def test_build_domain_static_constant_constraints():
    dom = build_domain((8, 4))
    # Static extents become constant upper bounds; no isl parameters.
    assert dom.dim(isl.dim_type.PARAM) == 0
    assert dom.dim(isl.dim_type.SET) == 2
    # The whole [0,8)x[0,4) box is 32 points.
    assert int(dom.count_val().num_si()) == 32


def test_build_domain_dynamic_dimvar_is_param():
    M = DimVar("M", 1, 4096)
    dom = build_domain((M, 128))
    # The dynamic extent becomes one isl parameter carrying its bound.
    assert dom.dim(isl.dim_type.PARAM) == 1
    assert dom.dim(isl.dim_type.SET) == 2


def test_build_domain_affine_extent():
    M = DimVar("M", 1, 4096)
    dom = build_domain((M + 1,))
    assert dom.dim(isl.dim_type.PARAM) == 1
    assert dom.dim(isl.dim_type.SET) == 1


def test_build_domain_rank0():
    dom = build_domain(())
    assert dom.dim(isl.dim_type.SET) == 0


def test_build_domain_non_affine_extent_raises():
    M = DimVar("M", 1, 4096)
    floordiv = Call(type=_I64, target=DimFloorDiv(), args=(M, Constant(type=_I64, value=4)))
    with pytest.raises(NotImplementedError, match="DimFloorDiv"):
        build_domain((floordiv,))


def test_build_domain_dimvar_times_const_is_affine():
    M = DimVar("M", 1, 4096)
    mul = Call(type=_I64, target=DimMul(), args=(M, Constant(type=_I64, value=4)))
    dom = build_domain((mul,))
    assert dom.dim(isl.dim_type.PARAM) == 1
    assert dom.dim(isl.dim_type.SET) == 1


def test_build_domain_symbol_times_symbol_raises():
    M = DimVar("M", 1, 4096)
    N = DimVar("N", 1, 4096)
    mul = Call(type=_I64, target=DimMul(), args=(M, N))
    with pytest.raises(NotImplementedError, match="symbolic"):
        build_domain((mul,))


def test_build_domain_same_name_conflicting_bounds_raises():
    with pytest.raises(ValueError, match="conflicting bounds"):
        build_domain((DimVar("S", 1, 8), DimVar("S", 1, 16)))


def test_validate_output_map_arity():
    om = isl.map("{ [m, k, n] -> [m, n] }")
    validate_output_map_arity(om, (1, 1))  # ok
    with pytest.raises(ValueError, match="range rank"):
        validate_output_map_arity(om, (1, 1, 1))


# ─── shape_from_relation ──────────────────────────────────────────────────────


def _ten(shape):
    return TensorType(shape=shape, dtype=DType.f32, layout=None, storage="gmem")


def _relation(extents, out_dst):
    dims = [f"d{i}" for i in range(len(extents))]
    src = "[" + ", ".join(dims) + "]"
    out_map = isl.map(f"{{ {src} -> [{out_dst}] }}")
    return AccessRelationResult(domain=build_domain(extents), maps=(out_map,))


def test_shape_from_relation_static():
    rel = _relation((16, 8), "d0, d1")
    assert shape_from_relation((_ten((16, 8)),), rel) == (16, 8)


def test_shape_from_relation_dimvar_param():
    n = DimVar("N", 1, 64)
    rel = _relation((16, n), "d0, d1")
    # The dynamic axis resolves back to the same DimVar by parameter name.
    assert shape_from_relation((_ten((16, n)),), rel) == (16, n)


def test_shape_from_relation_broadcast_constant_axis():
    # A constant output result is a size-1 axis.
    rel = _relation((16, 8), "d0, 0")
    assert shape_from_relation((_ten((16, 8)),), rel) == (16, 1)


def test_shape_from_relation_rank0():
    rel = _relation((), "")
    assert shape_from_relation((_ten(()),), rel) == ()


def test_shape_from_relation_non_projection_fails_closed():
    rel = _relation((16, 8), "d0 + d1")
    with pytest.raises(ValueError, match="pure projection"):
        shape_from_relation((_ten((16, 8)),), rel)
