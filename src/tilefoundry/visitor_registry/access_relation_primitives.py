"""GLOBAL-level access relation handlers for existing HIR primitives.

This module retrofits ``access_relation`` handlers onto built-in primitives
(``Binary``, ``MatMul``, ``RMSNorm``, ``Cast``, ``Reshape``, ``Transpose``,
``Slice``, ``Gather``, ``SoftMax``, ``TupleGetItem``) so the SGLang baseline
graph can iterate every Call and find a handler.

All handlers cover only the GLOBAL black-box level:

- Pointwise / shape-preserving ops (``Binary``, ``RMSNorm``, ``Cast``,
  ``SoftMax``) → identity multi_aff per input, identity per output.
- Shape-rearrangement ops (Reshape, Transpose, Slice, Gather) → identity at
  GLOBAL level (each output element corresponds to a single input element via
  a deterministic remap; the exact remap is encoded in the op attributes and
  not duplicated in the relation at this level).
- MatMul → input access is "scan reduction axis" (isl.map), output identity.
- TupleGetItem → identity multi_aff (structural extractor).

"""
from __future__ import annotations

import isl

from tilefoundry.ir.core import Call
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.hir.nn.softmax import SoftMax
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem

from .access_relation import (
    OPAQUE,
    AccessRelations,
    register_access_relation,
)


def _identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return isl.multi_aff("{ [] -> [] }")
    dims = ", ".join(f"i{i}" for i in range(rank))
    return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")


# ── Elementwise binary (kinded ``Binary``) ─────────────────────────────


@register_access_relation(Binary)
def _elementwise_binary(call: Call, ctx) -> AccessRelations:
    out_ty = ctx.type_of(call)
    rank = len(out_ty.shape)
    ident = _identity(rank)
    return AccessRelations(inputs=(ident, ident), outputs=(ident,))


# ── RMSNorm: x identity, weight identity (broadcast along last dim treated
#   as identity at GLOBAL black-box; reduction is internal to the op). ──


@register_access_relation(RMSNorm)
def _rms_norm_relation(call: Call, ctx) -> AccessRelations:
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    return AccessRelations(
        inputs=(_identity(rank), _identity(1)),
        outputs=(_identity(rank),),
    )


# ── Softmax: input axis-scan, output identity. ───────────────────────


@register_access_relation(SoftMax)
def _softmax_relation(call: Call, ctx) -> AccessRelations:
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    # For now, identity at GLOBAL black-box (the per-axis reduction is internal).
    ident = _identity(rank)
    return AccessRelations(inputs=(ident,), outputs=(ident,))


# ── MatMul: lhs / rhs share reduction over K dim (isl.map at GLOBAL). ──


@register_access_relation(MatMul)
def _matmul_relation(call: Call, ctx) -> AccessRelations:
    lhs_ty = ctx.type_of(call.args[0])
    rhs_ty = ctx.type_of(call.args[1])
    out_ty = ctx.type_of(call)
    out_rank = len(out_ty.shape)
    lhs_rank = len(lhs_ty.shape)
    rhs_rank = len(rhs_ty.shape)
    # Output is identity over its own iteration domain; for inputs, GLOBAL
    # black-box — declare identity multi_aff (the K-dim reduction is internal
    # to the op semantics at this level).
    return AccessRelations(
        inputs=(_identity(lhs_rank), _identity(rhs_rank)),
        outputs=(_identity(out_rank),),
    )


# ── Shape-rearrangement ops: identity at GLOBAL black-box. ────────────


@register_access_relation(Cast)
def _cast_relation(call: Call, ctx) -> AccessRelations:
    rank = len(ctx.type_of(call).shape)
    return AccessRelations(inputs=(_identity(rank),), outputs=(_identity(rank),))


@register_access_relation(Reshape)
def _reshape_relation(call: Call, ctx) -> AccessRelations:
    in_rank = len(ctx.type_of(call.args[0]).shape)
    out_rank = len(ctx.type_of(call).shape)
    return AccessRelations(
        inputs=(_identity(in_rank),), outputs=(_identity(out_rank),)
    )


@register_access_relation(Transpose)
def _transpose_relation(call: Call, ctx) -> AccessRelations:
    rank = len(ctx.type_of(call).shape)
    return AccessRelations(inputs=(_identity(rank),), outputs=(_identity(rank),))


@register_access_relation(Slice)
def _slice_relation(call: Call, ctx) -> AccessRelations:
    in_rank = len(ctx.type_of(call.args[0]).shape)
    out_rank = len(ctx.type_of(call).shape)
    return AccessRelations(
        inputs=(_identity(in_rank),), outputs=(_identity(out_rank),)
    )


@register_access_relation(Gather)
def _gather_relation(call: Call, ctx) -> AccessRelations:
    """Gather pulls per-index slices; access pattern is data-dependent on the
    indices arg → input data is OPAQUE; indices identity; output identity."""
    in_rank = len(ctx.type_of(call.args[0]).shape)
    idx_rank = len(ctx.type_of(call.args[1]).shape)
    out_rank = len(ctx.type_of(call).shape)
    del in_rank  # data input is OPAQUE
    return AccessRelations(
        inputs=(OPAQUE, _identity(idx_rank)),
        outputs=(_identity(out_rank),),
    )


# ── TupleGetItem: structural extractor. ───────────────────────────────


@register_access_relation(TupleGetItem)
def _tuple_get_item_relation(call: Call, ctx) -> AccessRelations:
    out_ty = ctx.type_of(call)
    rank = len(out_ty.shape)
    return AccessRelations(
        inputs=(_identity(rank),), outputs=(_identity(rank),)
    )


__all__: list[str] = []
