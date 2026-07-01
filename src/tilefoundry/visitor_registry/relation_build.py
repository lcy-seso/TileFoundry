"""Forward access-relation construction helpers (input-type driven).

These build the bounded iteration ``domain`` (an ``isl.set``) for an op from
its iteration extents. A static extent becomes a constant constraint
(``0 <= d < N``); a ``DimVar`` becomes an isl parameter constrained to its
half-open ``[lo, hi)`` range; affine dim-arithmetic extents translate to isl affine
expressions over those parameters. The result feeds
``AccessRelationResult.domain``; the relation never carries a tensor shape.
"""
from __future__ import annotations

import isl

from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.types.dim import DimAdd, DimMul, DimSub, DimVar


def _is_const(node) -> bool:
    """A static (non-symbolic) dim operand."""
    if isinstance(node, bool):
        return False
    return isinstance(node, int) or isinstance(node, Constant)


def _dim_to_isl(dim, params: dict[str, tuple[int, int]]) -> str:
    """Render *dim* as an isl expression string, recording any ``DimVar`` it
    uses in *params* (name → ``(lo, hi)``).

    Only affine extents are expressible: ``+`` / ``-``, and ``*`` with at least
    one constant operand. A symbol×symbol product (or any other dim op) raises
    ``NotImplementedError`` so it never reaches libisl as a non-affine string.
    A ``DimVar`` reused under conflicting bounds raises ``ValueError``.
    """
    if isinstance(dim, bool):
        raise TypeError("ShapeDim must not be bool")
    if isinstance(dim, int):
        return str(dim)
    if isinstance(dim, Constant):
        return str(int(dim.value))
    if isinstance(dim, DimVar):
        bound = (dim.lo, dim.hi)
        prev = params.get(dim.name)
        if prev is not None and prev != bound:
            raise ValueError(
                f"DimVar {dim.name!r} used with conflicting bounds {prev} vs {bound}"
            )
        params[dim.name] = bound
        return dim.name
    if isinstance(dim, Call):
        op = type(dim.target)
        if op in (DimAdd, DimSub):
            a = _dim_to_isl(dim.args[0], params)
            b = _dim_to_isl(dim.args[1], params)
            return f"({a} {'+' if op is DimAdd else '-'} {b})"
        if op is DimMul:
            lhs, rhs = dim.args
            if not (_is_const(lhs) or _is_const(rhs)):
                raise NotImplementedError(
                    "DimMul of two symbolic dims is not affine-expressible as an isl extent"
                )
            a = _dim_to_isl(lhs, params)
            b = _dim_to_isl(rhs, params)
            return f"({a} * {b})"
        raise NotImplementedError(
            f"dim op {op.__name__} is not affine-expressible as an isl extent"
        )
    raise TypeError(f"unsupported ShapeDim {type(dim).__name__}")


def build_domain(extents: tuple) -> "isl.set":
    """Bounded iteration domain ``{ [d0, ..., dn] : 0 <= di < extent_i }``.

    Static extents are constant constraints; ``DimVar`` extents are isl
    parameters carrying their half-open ``[lo, hi)`` bound. A rank-0 op gives ``{ [] }``.
    """
    params: dict[str, tuple[int, int]] = {}
    dims = [f"d{i}" for i in range(len(extents))]
    constraints = [
        f"0 <= d{i} < {_dim_to_isl(ext, params)}" for i, ext in enumerate(extents)
    ]
    constraints += [f"{lo} <= {name} < {hi}" for name, (lo, hi) in params.items()]
    prefix = f"[{', '.join(params)}] -> " if params else ""
    if not dims:
        return isl.set(prefix + "{ [] }")
    body = f"{{ [{', '.join(dims)}] : {' and '.join(constraints)} }}"
    return isl.set(prefix + body)


def _extent_of_domain_dim(domain: "isl.set", d: int, dimvars: dict):
    """Extent of domain dim *d* = ``dim_max(d) + 1``. Static → ``int``; a bare
    isl parameter (coeff 1, offset 0 after the +1) resolves back to its
    ``DimVar`` via *dimvars*. Anything else (piecewise max, non-unit / multiple
    params, divs) is not a recoverable ShapeDim → raise (fail closed)."""
    pieces: list = []
    domain.dim_max(d).foreach_piece(lambda _s, a: pieces.append(a))
    if len(pieces) != 1:
        raise ValueError(f"domain dim {d}: dim_max is piecewise; cannot recover extent")
    aff = pieces[0]
    if aff.dim(isl.dim_type.DIV):
        raise ValueError(f"domain dim {d}: dim_max involves a div; cannot recover extent")
    n_par = aff.dim(isl.dim_type.PARAM)
    params = [
        (aff.get_dim_name(isl.dim_type.PARAM, i),
         int(aff.get_coefficient_val(isl.dim_type.PARAM, i).num_si()))
        for i in range(n_par)
    ]
    size_const = int(aff.get_constant_val().num_si()) + 1
    nonzero = [(name, c) for name, c in params if c != 0]
    if not nonzero:
        return size_const
    if len(nonzero) == 1 and nonzero[0][1] == 1 and size_const == 0:
        name = nonzero[0][0]
        if name in dimvars:
            return dimvars[name]
    raise ValueError(f"domain dim {d}: cannot recover ShapeDim from extent {aff}")


def shape_from_relation(input_types: tuple, relation) -> tuple:
    """Derive the output shape from the relation's output map + bounded domain.

    Each output map result axis is a pure projection of a domain dim (its
    extent becomes the output dim) or a constant (a size-1 output axis). The
    domain (built forward from the input types, dynamic dims as isl params) is
    the single source of bounds; ``DimVar``s are recovered by parameter name.
    A non-projection / non-constant result axis, or an extent that does not
    resolve to a ShapeDim, fails closed.
    """
    domain = relation.domain
    output_map = relation.maps[-1]
    ma = output_map.as_pw_multi_aff().as_multi_aff()
    n_out = ma.dim(isl.dim_type.OUT)
    n_in = ma.dim(isl.dim_type.IN)
    dimvars: dict = {}
    for t in input_types:
        for dim in t.shape:
            if isinstance(dim, DimVar):
                dimvars[dim.name] = dim
    shape: list = []
    for o in range(n_out):
        aff = ma.get_at(o)
        used = [
            (j, int(aff.get_coefficient_val(isl.dim_type.IN, j).num_si()))
            for j in range(n_in)
            if int(aff.get_coefficient_val(isl.dim_type.IN, j).num_si()) != 0
        ]
        if not used:
            shape.append(1)  # constant result: a size-1 output axis
        elif len(used) == 1 and used[0][1] == 1:
            shape.append(_extent_of_domain_dim(domain, used[0][0], dimvars))
        else:
            raise ValueError(
                f"output axis {o} is not a pure projection or constant; "
                "cannot infer shape"
            )
    return tuple(shape)


def validate_output_map_arity(output_map: "isl.map", output_shape: tuple) -> None:
    """Check the output access map's range rank matches the claimed output
    shape rank. The relation carries no shape, so this is the consistency
    point between the relation and the typeinfer-side output shape."""
    n_out = output_map.dim(isl.dim_type.OUT)
    if n_out != len(output_shape):
        raise ValueError(
            f"output map range rank {n_out} != output shape rank {len(output_shape)}"
        )


__all__ = ["build_domain", "shape_from_relation", "validate_output_map_arity"]
