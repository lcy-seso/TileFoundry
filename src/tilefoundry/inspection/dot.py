"""DOT graph serializer for SSA HIR Functions.

Walks a ``hir.Function`` expression tree and produces a Graphviz DOT
string. Each ``Var`` / ``Call`` / ``Constant`` gets a numbered node.
``ShardLayout`` annotations on ``TensorType.layout`` are rendered as
per-mesh-axis labels with axis names, attrs, and mesh shape.

Example label::

    q_proj
    shape=(1,4096), dtype=bf16
    mesh=(cluster,cta,warp,lane):(64,2,8,32)
    cluster:B, cta:S(1), warp:P(sum), lane:P(sum)
"""

from __future__ import annotations

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, ShardLayout, Split


def _shard_label(layout, mesh_axis_names=("cluster", "cta", "warp", "lane")) -> str:
    """Render ShardLayout as a multi-line label: mesh axes + attrs."""
    if not isinstance(layout, ShardLayout):
        return ""
    parts = []  # noqa: F841
    # Per-axis attrs
    attr_parts = []
    for i, attr in enumerate(layout.attrs):
        name = mesh_axis_names[i] if i < len(mesh_axis_names) else f"ax{i}"
        if isinstance(attr, Broadcast):
            attr_parts.append(f"{name}:B")
        elif isinstance(attr, Split):
            attr_parts.append(f"{name}:S({attr.axis})")
        elif isinstance(attr, Partial):
            attr_parts.append(f"{name}:P({attr.reduction})")
        else:
            attr_parts.append(f"{name}:?")
    # Mesh shape
    mesh_shape = layout.mesh.layout.shape if hasattr(layout.mesh, 'layout') else ()
    mesh_names = ", ".join(mesh_axis_names[:len(mesh_shape)])
    return (
        f"mesh=({mesh_names}):{mesh_shape}",
        ", ".join(attr_parts),
    )


def _type_label(ty, mesh_axis_names) -> list[str]:
    """Type info lines for a node label."""
    if isinstance(ty, TensorType):
        shape = str(ty.shape).replace(" ", "")
        dtype = ty.dtype.name if hasattr(ty.dtype, 'name') else str(ty.dtype)
        lines = [f"shape=({shape}), dtype={dtype}"]
        if isinstance(ty.layout, ShardLayout):
            mesh_line, attr_line = _shard_label(ty.layout, mesh_axis_names)
            lines.append(mesh_line)
            lines.append(attr_line)
        return lines
    return [str(ty)]


def _op_name(target) -> str:
    cls = type(target).__name__
    for suffix in ("Op", "Expr", "Stmt"):
        if cls.endswith(suffix) and cls != suffix:
            cls = cls[:-len(suffix)]
    return cls


def _escape_dot(s: str) -> str:
    """Escape a string for safe inclusion in a DOT label."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def hir_function_to_dot(fn: HirFunction,
                        mesh_axis_names=("cluster", "cta", "warp", "lane")) -> str:
    """Convert a hir.Function to a DOT digraph string.

    Args:
        fn: The HIR function to visualize.
        mesh_axis_names: Names for mesh axes in display order.

    Returns:
        A Graphviz DOT format string.
    """
    lines = [f"digraph {fn.name} {{", '  rankdir=TB;',
             '  node [shape=box, style=filled, fillcolor="#f0f0f0"];',
             '  edge [fontsize=10, fontcolor="#555555"];', '']
    _counter = [0]
    _ids = {}
    _emitted: set[int] = set()

    def _id(node):
        key = id(node)
        if key not in _ids:
            _ids[key] = f"n{_counter[0]}"
            _counter[0] += 1
        return _ids[key]

    def _emit_node(nid, label_lines, fill="#f0f0f0"):
        escaped = [_escape_dot(ln) for ln in label_lines]
        label = "\\n".join(escaped)
        lines.append(f'  {nid} [label="{label}", fillcolor="{fill}"];')

    def _emit_edge(src_id, dst_id, label=""):
        if label:
            lines.append(f'  {src_id} -> {dst_id} [label="{label}"];')
        else:
            lines.append(f'  {src_id} -> {dst_id};')

    VAR_FILL = "#d4e6f1"
    CONST_FILL = "#f9e79f"
    CALL_FILL = "#d5f5e3"
    SHARDING_FILL = "#e8daef"

    def walk(expr):
        nid = _id(expr)
        key = id(expr)
        is_new = key not in _emitted
        if is_new:
            _emitted.add(key)

        if isinstance(expr, Var):
            if is_new:
                _emit_node(nid, [
                    f"Var: {expr.name}",
                    *_type_label(expr.type, mesh_axis_names),
                ], fill=VAR_FILL)
        elif isinstance(expr, Constant):
            if is_new:
                val = f"{expr.value:.6g}" if isinstance(expr.value, float) else str(expr.value)
                _emit_node(nid, [
                    f"Const: {val}",
                    *_type_label(expr.type, mesh_axis_names),
                ], fill=CONST_FILL)
        elif isinstance(expr, Call):
            target = expr.target
            if isinstance(target, Reshard):
                if is_new:
                    header = expr.loc if expr.loc else "Reshard"
                    if expr.loc:
                        header = f"{expr.loc}\\nReshard"
                    _emit_node(nid, [
                        header,
                        *_type_label(expr.type, mesh_axis_names),
                    ], fill=SHARDING_FILL)
                    for arg in expr.args:
                        walk(arg)
                        _emit_edge(_id(arg), nid)
                return

            op_label = _op_name(target)
            # Use loc as human-readable name when available
            header = expr.loc if expr.loc else op_label
            if expr.loc:
                header = f"{expr.loc}\\n{op_label}"
            if is_new:
                _emit_node(nid, [
                    header,
                    *_type_label(expr.type, mesh_axis_names),
                ], fill=CALL_FILL)
                for i, arg in enumerate(expr.args):
                    walk(arg)
                    edge_label = f"arg[{i}]" if len(expr.args) > 1 else ""
                    _emit_edge(_id(arg), nid, edge_label)
            else:
                for arg in expr.args:
                    walk(arg)
        else:
            if is_new:
                _emit_node(nid, [type(expr).__name__], fill="#ffffff")

    walk(fn.body)
    for p in fn.params:
        walk(p)

    # Legend
    lines.append("")
    lines.append('  subgraph cluster_legend {')
    lines.append('    label="Legend";')
    lines.append('    style=dashed;')
    lines.append('    fontsize=11;')
    lines.append('    l_var [label="Var/Param", fillcolor="#d4e6f1", shape=box, style=filled];')
    lines.append('    l_const [label="Constant", fillcolor="#f9e79f", shape=box, style=filled];')
    lines.append('    l_call [label="Op", fillcolor="#d5f5e3", shape=box, style=filled];')
    lines.append('    l_shard [label="Reshard", fillcolor="#e8daef", shape=box, style=filled];')
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def module_entry_to_dot(module: Module,
                        mesh_axis_names=("cluster", "cta", "warp", "lane")) -> str:
    """Convert a Module's entry function to DOT."""
    fn = module.entry_function()
    return hir_function_to_dot(fn, mesh_axis_names)


# Legacy alias
to_dot = hir_function_to_dot
