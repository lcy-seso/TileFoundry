from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.target import CudaTarget, Target
from tilefoundry.ir.types import CallableType, TensorType, Type, callable_type_for
from tilefoundry.ir.types.shard.mesh import Topology
from tilefoundry.visitor_registry.contexts import TypeInferContext


def _callable_type_for(params: tuple[Var, ...], return_type: Type) -> CallableType:
    """Project ``Function.params`` + ``return_type`` into the IR-level
    ``CallableType``.
    """
    return callable_type_for(params, return_type)


@dataclass(frozen=True)
class Function(Expr):
    """hir function container. body is a single Expr (SSA-as-DAG).

    ``Function`` is an ``Expr`` subclass; its ``Expr.type`` is the
    IR-level ``CallableType`` projected from ``params`` + ``return_type``.
    Construction sites SHOULD use :meth:`Function.build` so the ``type``
    projection stays consistent.

    ``topologies`` is the single-function convenience path: declared
    ``Topology`` values available for name resolution in
    ``with Mesh(topology="cta", ...)``. When a standalone ``Function``
    enters ``compile`` / ``jit``, its topologies lift to the enclosing
    ``Module``.
    """
    name: str
    params: tuple[Var, ...]
    body: Expr | None                       # None for a dispatch prototype (DSL ``pass``)
    return_type: Type
    topologies: tuple[Topology, ...] = field(default_factory=tuple)
    specializations: tuple[Pattern, ...] = field(default_factory=tuple)
    variants: tuple["Function", ...] = field(default_factory=tuple)
    target: Target = field(default_factory=CudaTarget)

    @classmethod
    def build(
        cls,
        *,
        name: str,
        params: tuple[Var, ...],
        body: Expr | None,
        return_type: Type,
        topologies: tuple[Topology, ...] = (),
        specializations: tuple[Pattern, ...] = (),
        variants: tuple["Function", ...] = (),
        target: Target | None = None,
        loc: str | None = None,
    ) -> "Function":
        """Construct a Function with the canonical CallableType."""
        return cls(
            name=name,
            params=params,
            body=body,
            return_type=return_type,
            topologies=tuple(topologies),
            specializations=tuple(specializations),
            variants=tuple(variants),
            target=target if target is not None else CudaTarget(),
            type=_callable_type_for(params, return_type),
            loc=loc,
        )

    def add_variant(self, variant: "Function") -> None:
        """Append a specialization ``variant`` during authoring.

        ``Function`` is a frozen, hashable dataclass and ``variants`` is in
        eq/hash, so accumulation uses controlled authoring-phase mutation
        (``object.__setattr__``). It is legal only before the base enters a
        ``Module``: ``Module`` construction seals the base (sets ``_sealed``)
        and any later ``add_variant`` raises. A base accumulating variants
        MUST NOT be hashed until it is sealed.
        """
        if getattr(self, "_sealed", False):
            raise RuntimeError(
                f"hir Function {self.name!r}: cannot add a specialization "
                f"variant after the function has entered a Module (sealed)"
            )
        object.__setattr__(self, "variants", (*self.variants, variant))

    def seal(self) -> None:
        """Freeze authoring mutation: ``add_variant`` raises afterwards.

        Called by ``Module`` construction on each function it contains.
        Idempotent. Variants are sealed alongside their base.
        """
        object.__setattr__(self, "_sealed", True)
        for v in self.variants:
            v.seal()


def canonical_specialization_signature(
    specializations: tuple[Pattern, ...],
) -> str:
    """Deterministic identity string for a Function's specialization tuple.

    Same-name Functions are distinguished by this signature. For v0 the
    only allowed pattern is ``DimVarRangePat``, so the signature is
    ``"<dim_var>$<lo>_<hi>"`` joined by ``;`` in declared order.
    """

    parts: list[str] = []
    for pat in specializations:
        if isinstance(pat, DimVarRangePat):
            parts.append(f"{pat.dim_var}${pat.lo}_{pat.hi}")
        else:
            # Fall back to repr for forward-compat; v0 verifier rejects
            # non-DimVarRangePat patterns elsewhere.
            parts.append(repr(pat))
    return ";".join(parts)


def _check_arg_against_param(call, ctx, callee, i, param, arg_ty: Type) -> None:
    """Validate one call argument against a callee parameter.

    A parameter declared *without* sharding (``TensorType`` with ``layout is
    None``) is a logical tensor: its layout is unconstrained, so an argument of
    any layout (plain / replicated / split / partial) is accepted as long as
    its logical shape and dtype match. A parameter declared *with* a
    ``ShardLayout`` is an explicit layout constraint: the argument's type must
    match it exactly. Non-tensor parameters require exact type equality.
    """
    p = param.type
    if isinstance(p, TensorType) and isinstance(arg_ty, TensorType) and p.layout is None:
        if arg_ty.shape != p.shape or arg_ty.dtype != p.dtype:
            ctx.error(
                call,
                f"hir Function call {callee.name!r}: arg {i} shape/dtype "
                f"mismatch — callee param {param.name!r} expects logical "
                f"{p.shape} {p.dtype}, got {arg_ty.shape} {arg_ty.dtype}",
            )
        return
    if arg_ty != p:
        ctx.error(
            call,
            f"hir Function call {callee.name!r}: arg {i} type mismatch — "
            f"callee param {param.name!r} expects {p!r}, got {arg_ty!r}",
        )


@register_typeinfer(Function)
def _typeinfer_hir_function_call(call: Call, ctx) -> Type:
    """Typeinfer handler for ``Call(target=hir.Function, args=...)``.

    The callee body is re-derived under the *actual* argument types: each
    parameter is bound to its caller argument's type and the body is
    typeinferred in a fresh context, so a layout (sharding) supplied by the
    caller flows into a layout-unconstrained (plain) parameter and propagates
    through the body. A parameter that declares an explicit ``ShardLayout``
    constrains its argument (mismatch fails at the boundary); a plain parameter
    only constrains logical shape / dtype. The derived body type is this call's
    result, so the callee specializes per call site.

    When the body cannot express a propagated sharding (e.g. a reshape whose
    cute factorization straddles a new axis), it fails at that op rather than
    being rejected at the boundary.

    A dispatch prototype callee (``variants != ()``, ``body is None``) has no
    body to re-derive: dispatch selects a variant by the runtime argument
    shapes, and every variant shares the prototype's declared signature, so the
    call's result is the prototype's declared ``return_type``. The prototype
    body (``None``) is never inspected.
    """
    callee: Function = call.target  # type: ignore[assignment]
    expected = len(callee.params)
    got = len(call.args)
    if got != expected:
        ctx.error(
            call,
            f"hir Function call {callee.name!r}: arity mismatch — "
            f"callee declares {expected} parameter(s), call passed {got}",
        )
    if callee.variants:
        # Dispatch prototype: validate args against the shared signature and
        # return the declared return type — never typeinfer the None body.
        for i, (param, arg) in enumerate(zip(callee.params, call.args)):
            _check_arg_against_param(call, ctx, callee, i, param, ctx.type_of(arg))
        return callee.return_type
    sub = TypeInferContext(module=ctx.module)
    for i, (param, arg) in enumerate(zip(callee.params, call.args)):
        arg_ty = ctx.type_of(arg)
        _check_arg_against_param(call, ctx, callee, i, param, arg_ty)
        sub.cache[param] = arg_ty
    return sub.type_of(callee.body)


__all__ = [
    "Function",
    "_callable_type_for",
    "canonical_specialization_signature",
]
