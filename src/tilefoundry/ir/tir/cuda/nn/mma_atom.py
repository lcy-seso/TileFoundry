"""CUDA MMA op / atom model (CuTe ``MMA_Op`` → ``MMA_Atom``).

The explicit instruction descriptor / atom model is target-owned: an MMA atom
fixes a concrete hardware instruction and its fragment layouts, so it lives
under the CUDA target surface alongside the concrete instructions
(``ir/tir/cuda/nn/mma.py``).

- :class:`MmaOpSpec` — a fully-specified **named instruction** (CuTe
  ``MMA_Op``). dtype / shape / source-layout are baked into ``name``; it
  surfaces no layout knowledge and is constructed without a dtype.
- :class:`MmaAtom` — the realized atom (CuTe ``MMA_Atom``). It holds the
  A / B / C fragment ``ShardLayout``\\s (the atom's thread-value layout) plus
  the ``required_scope`` — the thread participation contract the atom needs.
  ``atom.A/B/C`` are layout contracts returned as-is; whether the *enclosing*
  mesh scope can host the atom is a separate **check** (see
  ``shard.scope_match``), not a rebind of the layout onto the caller's mesh.

Spec: tir.md §3.7
"""
from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard import Mesh, ShardLayout


@dataclass(frozen=True)
class MmaOpSpec:
    """A named, fully-specified MMA instruction (CuTe ``MMA_Op``).

    dtype / shape / source-layout live in ``name``; the explicit fields mirror
    that name so verify and codegen do not re-parse the string. Constructed
    without a dtype argument — everything is fixed by the instruction.
    """
    name: str
    shape_mnk: tuple[int, int, int]
    dtype_a: DType
    dtype_b: DType
    dtype_c: DType
    operand_layout: str  # e.g. "TN" (A row-major, B col-major)


@dataclass(frozen=True)
class MmaAtom:
    """Realized MMA atom (CuTe ``MMA_Atom``) — op + fragment layouts + scope.

    ``A`` / ``B`` / ``C`` are the per-operand fragment ``ShardLayout``\\s (the
    atom's lane→value layout contract). ``required_scope`` is the atom's thread
    participation contract — the thread mesh the atom needs. It is *not* the
    caller's mesh: whether a given enclosing scope can host the atom is checked
    with
    :func:`tilefoundry.ir.types.shard.scope_match.mesh_scope_matches_required_scope`.
    """
    op: MmaOpSpec
    A: ShardLayout
    B: ShardLayout
    C: ShardLayout
    required_scope: Mesh


__all__ = ["MmaOpSpec", "MmaAtom"]
