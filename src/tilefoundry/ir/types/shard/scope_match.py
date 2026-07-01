"""Match an enclosing mesh scope against an op's required thread scope.

Spec: tir.md §3.7

An op (e.g. an MMA atom) declares a ``required_scope`` — the thread
participation contract it needs (SM80 mma = 32 lanes arranged ``(4, 8)``;
WGMMA = 128 lanes / 4 warps). At a use site the question is **not** "is the
caller's mesh object equal to the atom's mesh" (that is brittle: binding-var
name, axis names, and topology aliases would all leak in). It is: *does the
current enclosing mesh scope, inverse-projected back to its thread topology,
provide the same thread participation the op requires?*

We answer that with the flat CuTe layout algebra (:mod:`layout_algebra`):

- the topology must be the same program level (``thread``) — a ``cta`` scope is
  not a warp scope even if its layout happens to carry ``(4, 8)``;
- the mesh must be self-consistent (topology domain == layout extent) and an
  admissible (inverse-projectable) execution scope;
- the thread-value decomposition (layout shape + strides) must match the
  required one. The atom's fragment ``Split`` attrs index the mesh axes, so the
  scope must provide the same multi-axis lane layout — e.g. SM80 needs the
  2-axis ``(4, 8):(1, 4)`` warp, not a flat ``(32,)``. Mesh object identity and
  axis names are *not* compared.

The same predicate is reused by ``T.mma`` verify for the operand/scope
check, so the fragment use point and the instruction cannot drift.
"""
from __future__ import annotations

from .layout import Layout
from .layout_algebra import is_inverse_projectable, size
from .mesh import Mesh


def _as_layout(mesh: Mesh) -> Layout:
    return Layout(shape=tuple(mesh.layout.shape), strides=tuple(mesh.layout.strides))


def _topology_domain(mesh: Mesh) -> "int | None":
    """Total thread count = product of the mesh's topology extents; ``None`` if
    any extent is dynamic (launch-provided)."""
    topos = mesh.topologies or (mesh.topology,)
    domain = 1
    for t in topos:
        if not isinstance(t.size, int):
            return None
        domain *= t.size
    return domain


def mesh_scope_matches_required_scope(current: Mesh, required: Mesh) -> bool:
    """True iff ``current`` provides the thread participation ``required`` needs.

    Name/identity-independent: only the program topology level, the static
    thread count, and the exact required thread-value layout (shape + strides)
    are compared.
    """
    # Same program topology level — a `cta` scope is never a `thread`/warp scope.
    if current.topology.name != required.topology.name:
        return False

    cur_domain = _topology_domain(current)
    req_domain = _topology_domain(required)
    if cur_domain is None or req_domain is None:
        return False

    cur_layout = _as_layout(current)
    req_layout = _as_layout(required)

    # Self-consistent mesh: topology domain == layout extent (Mesh does not
    # enforce this, so a `thread(64)` carrying a 32-element layout is rejected).
    if cur_domain != size(cur_layout) or req_domain != size(req_layout):
        return False

    # Must be an admissible execution scope (injective, compact-ordered).
    if not is_inverse_projectable(cur_layout):
        return False

    # Same thread-value decomposition: the fragment's Split attrs index the mesh
    # axes, so the lane layout must match exactly (shape + strides) — a flat or
    # differently-shaped 32-lane scope cannot host a 2-axis (4, 8) fragment.
    return cur_layout.shape == req_layout.shape and cur_layout.strides == req_layout.strides


__all__ = ["mesh_scope_matches_required_scope"]
