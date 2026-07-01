"""Module — top-level compilation unit.

``entry`` names the public entry function; verify_module checks it resolves.
``metadata`` holds lowering / target configuration (e.g. target).
``topologies`` carries the module-level topology declarations; these form the
namespace against which ``with Mesh(topology="cta", ...)`` strings resolve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types.shard.mesh import Topology

ModuleFunction = Union[HirFunction, PrimFunction]


@dataclass(frozen=True)
class Module:
    """Frozen container of functions + the name of the public entry function."""

    name: str
    functions: tuple[ModuleFunction, ...]
    entry: str
    topologies: tuple[Topology, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Seal each function so authoring mutation (``add_variant`` /
        ``.specialize``) is forbidden once it belongs to a Module. Sealing is
        idempotent and only applies to functions that support it (hir
        Functions); other entries are left untouched."""
        for fn in self.functions:
            seal = getattr(fn, "seal", None)
            if callable(seal):
                seal()

    def __getattr__(self, name: str) -> ModuleFunction:
        """Attribute access forwards to the function of that name, so a module
        reads like the model it mirrors: ``decoder.self_attention``. Each name
        maps to at most one entry (specialization variants live on the
        function's ``variants``, not as separate entries). Only fires for names
        absent as real attributes; dunder/private names are never functions and
        fall through to ``AttributeError``."""
        if name.startswith("_"):
            raise AttributeError(name)
        matches = tuple(fn for fn in self.functions if fn.name == name)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise AttributeError(f"Module {self.name!r} has no function {name!r}")
        raise AttributeError(
            f"Module {self.name!r}: {name!r} resolves to {len(matches)} "
            f"entries; one name must map to one function"
        )

    def function_named(self, name: str) -> tuple[ModuleFunction, ...]:
        """Return the functions whose name matches, in source order.

        Each name maps to at most one entry, so in a verified module this is
        length 0 or 1 (specialization variants live on the function's
        ``variants``, not as separate same-name entries).
        """
        return tuple(fn for fn in self.functions if fn.name == name)

    def lookup(self, name: str) -> ModuleFunction:
        """Return the function named ``name``; raise unless exactly one matches.

        It is the module-level resolution contract for a ``SymbolRef`` callee.
        """
        matches = self.function_named(name)
        if len(matches) != 1:
            raise ValueError(
                f"Module {self.name!r}: {name!r} must resolve to exactly one "
                f"function, found {len(matches)}"
            )
        return matches[0]

    def entry_function(self) -> ModuleFunction:
        matches = self.function_named(self.entry)
        if not matches:
            raise ValueError(
                f"Module {self.name!r}: entry {self.entry!r} not in functions"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Module {self.name!r}: entry {self.entry!r} resolves to "
                f"{len(matches)} functions; entry must be a unique callable"
            )
        return matches[0]


__all__ = ["Module", "ModuleFunction"]
