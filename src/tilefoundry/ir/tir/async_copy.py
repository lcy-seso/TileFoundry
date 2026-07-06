"""Effect-ful TIR ops for asynchronous (``cp.async``) gmem→smem staging.

Spec: tir.md §3.8

The Ampere-class async-copy building blocks: a producer issues ``CopyAsync``
(a non-blocking DMA into shared memory), groups the in-flight copies with
``CpAsyncCommit``, and a consumer blocks on ``CpAsyncWait(n)`` until all but
the ``n`` most-recent groups have landed. ``CopyAsync`` mirrors
``tir.memory.Copy`` (same operand shape) but lowers to ``cp.async`` PTX
instead of a synchronous ``cute::copy``; the commit / wait ops carry no
operands (they are fences over the in-flight async-group queue, like
``Sync``).
"""
from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer, register_verify_stmt
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import UnitType


@register_op(dialect="T", category="async", name="copy_async")
class CopyAsync(Op):
    """Async gmem→smem copy (``cp.async.cg.shared.global``); non-blocking.

    Spec: tir.md §3.8

    Same operand shape as ``tir.memory.Copy`` (``source`` → ``destination``),
    but the copy is issued asynchronously: it returns before the data lands,
    and the caller MUST order a later read of ``destination`` with a
    ``CpAsyncCommit`` + ``CpAsyncWait``. ``destination`` MUST be smem and
    ``source`` gmem — the only direction ``cp.async`` supports — and both MUST
    share a dtype (the fast path issues a byte copy, no cast).
    """
    source = ParamDef(kind="input", pattern=Tensor)
    destination = ParamDef(kind="input", pattern=Tensor)


@register_typeinfer(CopyAsync)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(CopyAsync)
def _(call: "Call", ctx: "VerifyContext") -> None:
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    if dst.storage != StorageKind.SMEM:
        ctx.error(call, f"CopyAsync destination must be smem, got {dst.storage}")
    if src.storage != StorageKind.GMEM:
        ctx.error(call, f"CopyAsync source must be gmem, got {src.storage}")
    if src.dtype != dst.dtype:
        ctx.error(call, f"CopyAsync dtype mismatch: {src.dtype} vs {dst.dtype}")


@register_op(dialect="T", category="async", name="cp_async_commit")
class CpAsyncCommit(Op):
    """Close the current in-flight ``cp.async`` group (``commit_group``).

    Spec: tir.md §3.8

    A fence with no operands: it snapshots every ``CopyAsync`` issued since the
    previous commit into one async group so a later ``CpAsyncWait`` can count
    groups.
    """


@register_typeinfer(CpAsyncCommit)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(CpAsyncCommit)
def _(call: "Call", ctx: "VerifyContext") -> None:
    return None


@register_op(dialect="T", category="async", name="cp_async_wait")
class CpAsyncWait(Op):
    """Block until all but the ``n`` newest committed groups have arrived
    (``cp.async.wait_group n``).

    Spec: tir.md §3.8

    A fence with no operands. ``n`` is a non-negative compile-time count of the
    most-recent groups allowed to stay in flight; ``n=0`` drains every
    outstanding group. After the wait, the drained groups' ``destination``
    tensors are safe to read.
    """
    n = ParamDef(kind="attribute", annotation=int, default=0)


@register_typeinfer(CpAsyncWait)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(CpAsyncWait)
def _(call: "Call", ctx: "VerifyContext") -> None:
    n = call.target.n
    if not isinstance(n, int) or n < 0:
        ctx.error(call, f"CpAsyncWait.n must be a non-negative int, got {n!r}")


__all__ = ["CopyAsync", "CpAsyncCommit", "CpAsyncWait"]
