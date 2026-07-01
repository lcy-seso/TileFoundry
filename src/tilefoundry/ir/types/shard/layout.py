from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from tilefoundry.ir.types.shape_dim import ShapeDim


@dataclass(frozen=True)
class Layout:
    """Cute-style layout: shape + per-axis cute strides.

    A shape / stride entry is a ``ShapeDim`` — a static ``int`` or a symbolic
    / dynamic dim (a ``DimVar`` or dim ``Expr``) — and may also be ``None`` for
    a launch-provided (dynamic) extent (the dynamic-CTA mesh layout
    ``Layout(shape=(None,), strides=(1,))``). Consumers that need a concrete
    integer (``Mesh.__getitem__``, ``T.sync`` participation) require static
    ``int`` entries and fail closed on a symbolic / dynamic one.

    ``strides`` MAY be ``None`` (the whole tuple) to signal an *un-materialized*
    layout coming from parser sugar (``docs/spec/shard.md §7.1.2`` +
    ``docs/spec/hir.md §3``). ``Reshard`` typeinfer fills the concrete tuple in
    based on the storage-level direction rule.

    Invariant: after ``Reshard`` typeinfer has run on a value, the
    ``strides`` reachable from that value's type is a concrete
    tuple; the un-materialized form is an intermediate-only signal
    that lowering / codegen / runtime never see.

    ``strides=()`` keeps its rank-0 scalar meaning (``shape=()``); it
    is NOT overloaded as a sentinel.
    """

    shape: tuple["ShapeDim | None", ...]
    strides: Optional[tuple["ShapeDim", ...]] = None


@dataclass(frozen=True)
class ComposedLayout:
    """CuTe composed layout: ``image(c) = inner(offset + outer(c))``.

    Field order + names mirror CuTeDSL ``make_composed_layout(inner, offset,
    outer)`` (``third_party/cutlass/python/CuTeDSL/cutlass/cute/core.py``):

    - ``outer`` — applied **first** (domain / input side); the domain shape and
      axis numbering of the composition come from ``outer``, so a binding
      ``ShardLayout``'s ``Split(k)`` references ``outer``'s domain axis.
    - ``offset`` — intermediate scalar offset added before ``inner``.
    - ``inner`` — applied **last** (codomain / output side).

    The left inverse reverses the composition (see CuTe
    ``layout_composed.hpp`` ``left_inverse``):
    ``image⁻¹(t) = outer⁻¹(inner⁻¹(t) − offset)``.
    """

    inner: "LayoutLike"
    offset: int
    outer: "LayoutLike"


# Forward ref resolved after shard_layout import
LayoutLike = Union[Layout, ComposedLayout, "ShardLayout"]  # noqa: F821

EMPTY_LAYOUT = Layout(shape=(), strides=())


__all__ = ["Layout", "ComposedLayout", "LayoutLike", "EMPTY_LAYOUT"]
