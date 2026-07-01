"""Pattern — minimal contract."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tilefoundry.ir.core.pattern import (
    AndPat,
    DimVarRangePat,
    Pattern,
    Scalar,
    Tensor,
    TensorPat,
    TypePattern,
)


@dataclass(frozen=True)
class FakeTy:
    shape: tuple[int, ...]
    dtype: str = "f32"


def test_pattern_match_contract() -> None:
    """Singletons + parametric patterns + And combinator share one contract."""
    assert Scalar.match(FakeTy(shape=()))
    assert not Scalar.match(FakeTy(shape=(3,)))
    assert Tensor.match(FakeTy(shape=(3, 4)))
    assert not Tensor.match(FakeTy(shape=()))

    rank2_bf16 = TensorPat(rank=2, dtype="bf16")
    assert rank2_bf16.match(FakeTy(shape=(3, 4), dtype="bf16"))
    assert not rank2_bf16.match(FakeTy(shape=(3,), dtype="bf16"))
    assert not rank2_bf16.match(FakeTy(shape=(3, 4), dtype="f32"))

    combined = AndPat(parts=(TensorPat(rank=2), TensorPat(dtype="f16")))
    assert combined.match(FakeTy(shape=(3, 4), dtype="f16"))
    assert not combined.match(FakeTy(shape=(3,), dtype="f16"))
    assert AndPat(parts=()).match(FakeTy(shape=()))  # empty AND is a tautology


# --- DimVarRangePat (dynamic-shape v0) -----------------------------------


def test_dim_var_range_pat_match_half_open() -> None:
    """``DimVarRangePat("S", 1, 4)`` matches int in the half-open [1, 4)."""
    p = DimVarRangePat("S", 1, 4)
    # in-range (lo inclusive, hi exclusive)
    assert p.match(1)
    assert p.match(2)
    assert p.match(3)
    # out of range
    assert not p.match(4)
    assert not p.match(0)
    assert not p.match(5)
    assert not p.match(-1)


def test_dim_var_range_pat_single_point() -> None:
    """A single-point range ``[k, k+1)`` matches exactly ``k``."""
    p = DimVarRangePat("S", 3, 4)
    assert p.match(3)
    assert not p.match(2)
    assert not p.match(4)


def test_dim_var_range_pat_rejects_non_int() -> None:
    p = DimVarRangePat("S", 1, 4)
    assert not p.match(2.0)
    assert not p.match("2")
    assert not p.match(None)
    # bool subclasses int but is not a valid shape value
    assert not p.match(True)
    assert not p.match(False)


def test_dim_var_range_pat_requires_lo_lt_hi() -> None:
    # Half-open interval: a single point is [k, k+1); lo >= hi is empty/invalid.
    DimVarRangePat("S", 4, 5)  # single value 4 — no raise
    with pytest.raises(ValueError, match="lo < hi"):
        DimVarRangePat("S", 4, 4)  # empty range
    with pytest.raises(ValueError, match="lo < hi"):
        DimVarRangePat("S", 5, 4)


def test_dim_var_range_pat_requires_non_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty str"):
        DimVarRangePat("", 1, 4)


def test_dim_var_range_pat_rejects_non_int_bounds() -> None:
    with pytest.raises(TypeError, match="lo must be int"):
        DimVarRangePat("S", 1.0, 4)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="hi must be int"):
        DimVarRangePat("S", 1, 4.0)  # type: ignore[arg-type]


def test_dim_var_range_pat_is_hashable() -> None:
    """Frozen dataclass; identical args produce equal, hashable instances."""
    p1 = DimVarRangePat("S", 1, 4)
    p2 = DimVarRangePat("S", 1, 4)
    assert p1 == p2
    assert hash(p1) == hash(p2)
    # Distinct args produce distinct instances.
    assert p1 != DimVarRangePat("S", 4, 8)
    assert p1 != DimVarRangePat("H", 1, 4)


# --- Back-compat alias ----------------------------------------------------


def test_type_pattern_alias_resolves_to_pattern() -> None:
    """``TypePattern`` is kept as a back-compat alias for one release cycle."""
    assert TypePattern is Pattern
    # Existing subclasses are still recognised through the alias.
    assert issubclass(TensorPat, TypePattern)
    assert issubclass(AndPat, TypePattern)
    assert issubclass(DimVarRangePat, TypePattern)
