"""``tilefoundry.dsl.T`` — TIR op surface (``dialect='T'``).

Names are resolved on demand against the OpSchema registry; calling
``T.<name>(*args, **kw)`` runs first-match overload resolution and
constructs the corresponding IR node via the schema's ``builder``.

See :mod:`tilefoundry.dsl.tf` for the HIR counterpart.
"""

from __future__ import annotations

from typing import Any

from tilefoundry.ir.core.op_registry import get_schemas, iter_schema_names
from tilefoundry.parser.overload import resolve

_DIALECT = "T"


def __getattr__(name: str) -> Any:
    """Same contract as :mod:`tilefoundry.dsl.tf`: single-schema names
    return the Op class; multi-schema names return a runtime resolver.
    ``__all__`` is resolved on demand so star-import sees ops
    registered after first module load.

    Platform sub-namespaces (``T.cuda``, later other targets) are resolved
    first: they hold compile-time instruction descriptors rather than
    callable Ops, so they bypass the schema registry.
    """
    if name == "__all__":
        return sorted(iter_schema_names(_DIALECT))
    from tilefoundry.dsl.T._platforms import PLATFORM_NAMESPACES  # noqa: PLC0415
    if name in PLATFORM_NAMESPACES:
        return PLATFORM_NAMESPACES[name]
    schemas = get_schemas(_DIALECT, name)
    if not schemas:
        raise AttributeError(
            f"tilefoundry.dsl.T has no op named {name!r} "
            f"(did you forget to import the module that defines it?)"
        )

    if len(schemas) == 1:
        return schemas[0].op_class

    def _call(*args: Any, **kwargs: Any) -> Any:
        arg_types = tuple(getattr(a, "type", None) for a in args)
        chosen = resolve(schemas, arg_types)
        return chosen.builder(*args, **kwargs)

    _call.__name__ = name
    _call.__qualname__ = f"tilefoundry.dsl.T.{name}"
    _call.__doc__ = schemas[0].op_class.__doc__
    return _call


def __dir__() -> list[str]:
    return sorted(iter_schema_names(_DIALECT))


# See :mod:`tilefoundry.dsl.tf` for the ``__all__`` lazy-resolution rationale.
