"""``tilefoundry.dsl.tf`` — HIR op surface (``dialect='tf'``).

Names are resolved on demand against the OpSchema registry; calling
``tf.<name>(*args, **kw)`` runs first-match overload resolution and
constructs the corresponding IR node via the schema's ``builder``.

For static type completion in editors, generated ``.pyi`` stubs live
alongside this module (gitignored). Run ``tilefoundry stub regen`` to
refresh them after registering new ops.
"""

from __future__ import annotations

from typing import Any

from tilefoundry.ir.core.op_registry import get_schemas, iter_schema_names
from tilefoundry.parser.overload import resolve

_DIALECT = "tf"


def __getattr__(name: str) -> Any:
    """Resolve ``name`` against the HIR OpSchema registry.

    Special-case ``__all__`` so ``from tilefoundry.dsl.tf import *``
    sees op classes registered *after* this module was first
    imported (e.g. test-fixture custom ops registered at test
    collection time, after ``tilefoundry.dsl.tf`` first loaded).

    For single-schema names (the common case) we return the op
    **class** directly — that makes ``add = tf.add`` /
    ``from tilefoundry.dsl.tf import add`` bind ``add`` to the actual
    ``Add`` Op subclass. The parser then sees ``ast.Name("add")``,
    looks it up in the function's closure, and uses the class as the
    Op target without consulting any registry shortcut.

    For multi-schema overloads (``len(schemas) > 1``) we return a
    runtime resolver callable that picks a schema by best-effort
    arg-type matching — real parser-time overload dispatch lives in
    :mod:`tilefoundry.parser.overload`.

    Unknown names raise :class:`AttributeError` so editors and
    tooling get clean attribute-error semantics.
    """
    if name == "__all__":
        return sorted(iter_schema_names(_DIALECT))
    schemas = get_schemas(_DIALECT, name)
    if not schemas:
        raise AttributeError(
            f"tilefoundry.dsl.tf has no op named {name!r} "
            f"(did you forget to import the module that defines it?)"
        )

    # Surface aliases (``schema.op_class is None``) prepend
    # to the bucket so they win first-match. Return the alias builder
    # function — it carries ``_op_schema`` (set by ``@register_alias``)
    # so the parser's ``_resolve_call_target`` recognises it via the
    # same ``getattr(val, "_op_schema", None)`` path used for Op
    # classes. This way ``from tilefoundry.dsl.tf import add`` /
    # ``tf.add(...)`` route through the alias schema even when a
    # legacy real-Op schema with the same name is also registered
    # (a transitional state).
    first = schemas[0]
    if first.op_class is None:
        return first.builder

    if len(schemas) == 1:
        return first.op_class

    # Multi-overload (legacy): runtime-best-effort resolver for callers
    # outside the parser. Parser-time dispatch goes through OpSchema
    # directly.
    def _call(*args: Any, **kwargs: Any) -> Any:
        arg_types = tuple(getattr(a, "type", None) for a in args)
        chosen = resolve(schemas, arg_types)
        return chosen.builder(*args, **kwargs)

    _call.__name__ = name
    _call.__qualname__ = f"tilefoundry.dsl.tf.{name}"
    _call.__doc__ = first.op_class.__doc__
    return _call


def __dir__() -> list[str]:
    return sorted(iter_schema_names(_DIALECT))


# ``__all__`` is **not** set as a module-level attribute — instead the
# module ``__getattr__`` above resolves it on demand. That way star
# import sees ops registered *after* this module was first loaded
# (test-fixture custom ops, lazy-loaded modules, etc.); a frozen
# module-level ``__all__`` would miss them.
