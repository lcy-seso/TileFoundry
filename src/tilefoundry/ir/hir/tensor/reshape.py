from __future__ import annotations

from dataclasses import replace

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError, TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    ShardLayout,
    Split,
)


@register_op
class Reshape(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    new_shape = ParamDef(kind="attribute", annotation=tuple)


def _carry_sharded_reshape(layout: ShardLayout, new_shape: tuple):
    """Carry a genuine sharding across a reshape (a view) when the cute
    factorization aligns with *new_shape*.

    A reshape is a view: it inserts/removes size-1 axes and groups along
    boundaries. It can carry the sharding iff every cute position lies entirely
    within one new axis -- no position straddles a new-axis boundary -- so each
    non-size-1 new axis is the product of a contiguous run of cute positions.
    Size-1 axes hold no sharding and are inserted/dropped freely; the cute
    factorization (and any tiling) of the surviving positions is preserved, and
    each ``Split`` cute-axis reference is remapped to its new cute position
    (``Partial`` / ``Broadcast`` are mesh-axis states with no cute axis and
    carry through unchanged).

    Returns the carried ``ShardLayout``, or ``None`` when the reshape cannot
    express the sharding (the caller fails closed). All extents must be static
    to verify alignment.
    """
    cute = layout.layout
    cute_shape = cute.shape
    if not all(isinstance(d, int) and not isinstance(d, bool) for d in cute_shape):
        return None
    if not all(isinstance(d, int) and not isinstance(d, bool) for d in new_shape):
        return None

    cute_strides = cute.strides
    n_cute = len(cute_shape)

    def _next_nonunit(start: int) -> int:
        i = start
        while i < n_cute and int(cute_shape[i]) == 1:
            i += 1
        return i

    # Walk new axes; compose each non-size-1 new axis from a contiguous run of
    # non-size-1 cute positions, recording the old cute position -> new cute
    # position remap. Inserted size-1 new axes get a fresh unit position.
    new_positions: list[tuple[int, int, int | None]] = []  # (size, stride, old_pos)
    old_to_new: dict[int, int] = {}
    ci = 0
    for dim in new_shape:
        d = int(dim)
        if d == 1:
            new_positions.append((1, 0, None))
            continue
        prod = 1
        while prod < d:
            ci = _next_nonunit(ci)
            if ci >= n_cute:
                return None  # ran out of cute positions to compose this axis
            cs = int(cute_shape[ci])
            prod *= cs
            if prod > d:
                return None  # this cute position straddles a new-axis boundary
            stride = cute_strides[ci] if cute_strides is not None else 0
            old_to_new[ci] = len(new_positions)
            new_positions.append((cs, stride, ci))
            ci += 1
    if _next_nonunit(ci) < n_cute:
        return None  # leftover non-size-1 cute positions cannot be placed

    new_attrs = []
    for attr in layout.attrs:
        if isinstance(attr, Split):
            if attr.axis not in old_to_new:
                return None  # a sharded cute position was dropped -> fail closed
            new_attrs.append(replace(attr, axis=old_to_new[attr.axis]))
        else:
            # Partial / Broadcast are mesh-axis states with no cute axis; they
            # carry through the reshape unchanged.
            new_attrs.append(attr)

    out_shape = tuple(s for s, _, _ in new_positions)
    out_strides = (
        None if cute_strides is None else tuple(st for _, st, _ in new_positions)
    )
    new_cute = Layout(shape=out_shape, strides=out_strides)
    return ShardLayout(layout=new_cute, attrs=tuple(new_attrs), mesh=layout.mesh)


@register_typeinfer(Reshape)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    new_shape = tuple(call.target.new_shape)

    new_layout = None
    if isinstance(x_ty.layout, ShardLayout):
        genuine = any(not isinstance(a, Broadcast) for a in x_ty.layout.attrs)
        if not genuine:
            new_layout = None  # replicated input -> unsharded output
        else:
            new_layout = _carry_sharded_reshape(x_ty.layout, new_shape)
            if new_layout is None:
                # A genuine sharding whose cute factorization does not align with
                # the new shape cannot be expressed; fail closed rather than
                # fabricate a layout. (Re-laying-out across a misaligned reshape
                # would need an explicit Reshard.)
                ctx.error(
                    call,
                    "Reshape cannot express the sharded layout: new shape does "
                    "not align with the input cute factorization",
                )
    return TensorType(
        shape=new_shape,
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=x_ty.storage,
    )


@register_eval(Reshape)
def _eval_reshape(ctx):
    # A symbolic (DimVar / Expr) target axis is inferred from the concrete input
    # via torch's ``-1``; at most one axis may be inferred.
    shape = tuple(
        int(d) if isinstance(d, int) and not isinstance(d, bool) else -1
        for d in ctx.op.new_shape
    )
    if shape.count(-1) > 1:
        raise EvalError(
            "reshape: at most one dynamic axis can be inferred, "
            f"got new_shape={ctx.op.new_shape!r}"
        )
    return TensorValue(data=ctx.args[0].data.reshape(shape), type=ctx.result_type)
