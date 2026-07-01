"""Emitter for ``tir.DispatchCall`` — nested ``if`` chain over case patterns.

v0 strategy: emit one ``if`` / ``else if`` clause per case in source order
(using the half-open ``lo <= S && S < hi`` predicate built from each
case's ``DimVarRangePat`` against the ``ShapeOf`` subject), then a
trailing ``else { fallback }``. The fallback is ``Sequential((Abort(),))``
per the v0 verifier rule and lowers to ``assert(false);`` (works in
both host wrapper context — where the dispatch entry lives — and
device context).

Arguments inside each case call are rendered positionally; ``Var``
args use the codegen context's bound name, and ``ShapeOf(param, axis)``
args use the canonical shape kernel param name
``<param.name>_shape_<axis>`` matching the host-wrapper plumbing.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.core import Var
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.shape import ShapeOf, shape_var_name
from tilefoundry.ir.tir.stmts import Abort, Evaluate, Sequential


def _render_subject(subject) -> str:
    """Render ``DispatchCall.subjects[0]`` as a C expression.

    v0 verifier guarantees ``ShapeOf(param, axis)``; the canonical
    kernel scalar name matches the host wrapper plumbing.
    """
    if isinstance(subject, ShapeOf):
        return shape_var_name(subject.param.name, subject.axis)
    raise NotImplementedError(
        f"DispatchCall emitter: subject of type {type(subject).__name__!r} "
        f"is not supported (v0 expects ShapeOf)."
    )


def _render_arg(arg, ctx: CodegenContext) -> str:
    if isinstance(arg, ShapeOf):
        return shape_var_name(arg.param.name, arg.axis)
    if isinstance(arg, Var):
        return ctx.name_for(arg)
    raise NotImplementedError(
        f"DispatchCall emitter: case-call arg of type "
        f"{type(arg).__name__!r} is not supported (v0 expects Var / ShapeOf)."
    )


def _render_case_predicate(pat: DimVarRangePat, subject_text: str) -> str:
    """Build ``((lo <= <subject>) && (<subject> < hi))`` literally.

    ``DimVarRangePat`` is a half-open range ``[lo, hi)``, so the upper bound
    uses ``<``. Mirrors the textual form ``render_scalar_predicate`` would
    produce for the equivalent ``AND(LE(lo, subject), LT(subject, hi))``
    Call — constructed directly to avoid building a throwaway Expr AST just
    for predicate rendering.
    """
    return (
        f"(({pat.lo} <= ({subject_text})) "
        f"&& (({subject_text}) < {pat.hi}))"
    )


def _emit_symbol_call(call: Evaluate, ctx: CodegenContext) -> None:
    callee_name = call.callable.name
    # The dispatch op emits from a host-wrapper context. The callee's
    # C++ wrapper is published under an internal sanitized symbol (see
    # ``prim_function._internal_wrapper_symbol``) — call that, not the
    # user-facing name (which is only a TVM FFI export key and may even
    # collide with ``::main`` for entries literally named ``main``).
    from tilefoundry.codegen.cuda.tir.prim_function import (  # noqa: PLC0415
        _internal_wrapper_symbol,
    )
    # The callee is the variant's host wrapper. Its signature only
    # carries user-visible params (tvm::ffi::Tensor + non-hidden scalar
    # params); hidden ``<param>_shape_<axis>`` scalars are extracted
    # locally inside the wrapper from each tensor arg's runtime
    # ``shape()``. Forwarding them again from the dispatch entry would
    # mismatch the wrapper signature, so we drop ``ShapeOf`` args here.
    visible_args = tuple(a for a in call.args if not isinstance(a, ShapeOf))
    args_text = ", ".join(_render_arg(a, ctx) for a in visible_args)
    ctx.emit(f"{_internal_wrapper_symbol(callee_name)}({args_text});")


def _emit_fallback(fallback: Sequential, ctx: CodegenContext) -> None:
    # v0 verifier guarantees Sequential((Abort(),)). The dispatch op is
    # emitted from the entry's host wrapper, where ``__trap()`` (a
    # ``__device__``-only intrinsic) does not compile. ``assert(false)``
    # works in both host and device contexts and aborts loudly.
    for stmt in fallback.body:
        if isinstance(stmt, Abort):
            ctx.emit("assert(false);")
        else:
            raise NotImplementedError(
                f"DispatchCall emitter: fallback stmt of type "
                f"{type(stmt).__name__!r} is not supported (v0 expects "
                f"Sequential((Abort(),)))."
            )


@register_codegen_cuda(DispatchCall)
def _emit(node: DispatchCall, ctx: CodegenContext) -> None:
    subject_text = _render_subject(node.subjects[0])

    for i, (patterns, call) in enumerate(zip(node.case_patterns, node.case_calls)):
        pat = patterns[0]
        if not isinstance(pat, DimVarRangePat):
            raise NotImplementedError(
                f"DispatchCall emitter: case pattern of type "
                f"{type(pat).__name__!r} is not supported (v0 expects "
                f"DimVarRangePat)."
            )
        pred = _render_case_predicate(pat, subject_text)
        prefix = "if" if i == 0 else "} else if"
        ctx.emit(f"{prefix} ({pred}) {{")
        ctx.indent()
        _emit_symbol_call(call, ctx)
        ctx.dedent()

    if node.case_calls:
        ctx.emit("} else {")
    else:
        ctx.emit("{")
    ctx.indent()
    _emit_fallback(node.fallback, ctx)
    ctx.dedent()
    ctx.emit("}")
