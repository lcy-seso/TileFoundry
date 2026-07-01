"""Per-target codegen emitter registry + target grouping.

Maps a function ``target`` (by its ``name``) to the emitter that lowers that
target's functions, mirroring nncase's ``Target.GetModuleCompiler(kind)`` shape:
group a module's functions by target, let each target emit its own
``LinkableModule``, then link those modules into one host-callable artifact.

This module is infrastructure only — registering / looking up emitters and
grouping functions. It does not change the default compile path.
"""
from __future__ import annotations

from typing import Callable

from tilefoundry.ir.core.module import Module, ModuleFunction
from tilefoundry.ir.target import Target

_EMITTERS: dict[str, Callable] = {}


def register_emitter(name: str, emitter: Callable) -> None:
    """Register the emitter that lowers functions whose target is *name*."""
    _EMITTERS[name] = emitter


def _ensure_defaults() -> None:
    if "cuda" not in _EMITTERS:
        # Split-pipeline per-target linkable-module emitters.
        from tilefoundry.codegen.cpu.module import (  # noqa: PLC0415
            emit_host_module,
        )
        from tilefoundry.codegen.cuda.module import (  # noqa: PLC0415
            emit_cuda_module,
        )

        _EMITTERS["cuda"] = emit_cuda_module
        _EMITTERS["cpu"] = emit_host_module


def get_emitter(target: "str | Target") -> Callable:
    """Return the emitter registered for *target* (a ``Target`` or its name)."""
    name = target.name if isinstance(target, Target) else target
    _ensure_defaults()
    try:
        return _EMITTERS[name]
    except KeyError:
        raise ValueError(
            f"no codegen emitter registered for target {name!r}; "
            f"have {sorted(_EMITTERS)}"
        ) from None


def group_functions_by_target(module: Module) -> dict[str, tuple[ModuleFunction, ...]]:
    """Group ``module.functions`` by their target name, preserving order."""
    groups: dict[str, list[ModuleFunction]] = {}
    for fn in module.functions:
        groups.setdefault(fn.target.name, []).append(fn)
    return {name: tuple(fns) for name, fns in groups.items()}


__all__ = ["get_emitter", "group_functions_by_target", "register_emitter"]
