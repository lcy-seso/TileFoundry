"""``Stmt`` base class.

Lives in ``tilefoundry.ir.core`` because it is referenced by
``tilefoundry.visitor_registry.contexts.VerifyContext`` (mesh scope
stack lookups, etc.). Hosting it under ``ir/core/`` keeps
``visitor_registry`` from depending on ``tilefoundry.ir.tir``, which
would close the historical ``ir.tir`` ↔ ``visitor_registry`` import
cycle.

Concrete TIR ``Stmt`` subclasses (``LetStmt`` / ``Evaluate`` /
``Sequential`` / ``MeshScope`` / ``For`` / ``While`` / ``If`` /
``Return`` / ``PrimFunction``) remain in
``tilefoundry.ir.tir.stmts``. ``tilefoundry.ir.tir.stmt`` is a thin
re-export shim that keeps ``from tilefoundry.ir.tir.stmt import Stmt``
working for back-compat.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stmt:
    """tir-only. hir does not contain Stmt nodes.

    Structural Stmts (Sequential / LetStmt / For / While / If /
    MeshScope / Return / Evaluate / PrimFunction) are not part of any
    callable registry; effect-ful TIR Ops register themselves via
    ``@register_op`` (and live in Stmt position via
    ``Evaluate(op, args)``).
    """

    loc: str | None = field(default=None, kw_only=True)
