"""``@module`` decorator — collect a class of DSL functions into an IR Module.

Spec: parser.md §2.7

``@module(entry="<name>")`` turns a class whose body holds ``@func`` /
``@prim_func`` methods into a ``tilefoundry.ir.core.module.Module``. The class body
is a pure function container: every non-dunder member is a DSL function and
nothing else. The decorated name binds directly to the Module, so
``Cls.<func_name>`` resolves a kernel by attribute and ``Cls.lookup`` remains for
programmatic unique-name resolution.
"""

from __future__ import annotations


def module(cls=None, *, entry: str):
    """Collect a class body's ``@func`` / ``@prim_func`` members into a ``Module``.

    Every non-dunder class member MUST be an ``@func`` / ``@prim_func`` result (an
    ``hir.Function`` / ``tir.PrimFunction``); an undecorated method, a nested
    class, or any stray attribute is rejected. Members are collected in
    definition order. ``entry`` names the public entry function — a forward
    reference resolved against the collected members — and MUST match exactly
    one. The decorated name binds to the resulting ``Module``.
    """
    from tilefoundry.ir.core.module import Module  # noqa: PLC0415 — avoid import cycle
    from tilefoundry.ir.hir.function import Function as HirFunction  # noqa: PLC0415
    from tilefoundry.ir.tir.prim_function import PrimFunction  # noqa: PLC0415

    def _wrap(cls_inner):
        functions = []
        for name, value in vars(cls_inner).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            if not isinstance(value, (HirFunction, PrimFunction)):
                raise TypeError(
                    f"@module {cls_inner.__name__!r}: member {name!r} is a "
                    f"{type(value).__name__}, not an @func / @prim_func result; a "
                    f"@module class body may contain only DSL functions"
                )
            # Specialization variants live on their base's ``variants`` (the
            # throwaway ``@base.specialize`` def), not as standalone entries.
            if getattr(value, "specializations", ()):
                continue
            functions.append(value)
        if not functions:
            raise TypeError(
                f"@module {cls_inner.__name__!r}: no @func / @prim_func members"
            )
        names = [fn.name for fn in functions]
        # One name maps to one function (core-ir verify_module invariant): a
        # class-body alias (``y = some_func``) collects the same function twice,
        # so reject any repeated function name.
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"@module {cls_inner.__name__!r}: duplicate function name(s) "
                f"{dupes} (a class-body alias of a DSL function is not allowed; "
                f"one name maps to one function)"
            )
        if entry not in names:
            raise ValueError(
                f"@module {cls_inner.__name__!r}: entry {entry!r} names no "
                f"collected function (have {names})"
            )
        return Module(name=cls_inner.__name__, functions=tuple(functions), entry=entry)

    if cls is not None:
        return _wrap(cls)
    return _wrap


__all__ = ["module"]
