#!/usr/bin/env python
"""Lint ``docs/spec/*.md`` against the mechanically-checkable subset of
``docs/SPEC-RULES.md``.

The check is deliberately narrow: it flags only tokens / section headers that
SPEC-RULES explicitly forbids AND that can be detected without false positives
on legitimate spec prose. Anything a term could legitimately be (a bare commit
hash looks like any hex literal; a bare ``#123`` looks like a link anchor) is
left to human review rather than guessed at.

``docs/SPEC-RULES.md`` itself is not a spec section and is not linted — it
names the forbidden tokens as examples.

Usage: ``spec_rules_lint.py <file.md> ...`` (the pre-commit hook passes the
staged ``docs/spec/*.md`` files). Exits non-zero and prints ``file:line:
message`` for each violation.
"""
from __future__ import annotations

import re
import sys

# Each rule: (compiled regex, message). The regex matches a forbidden token on
# a single line. Header-only rules are applied to heading lines separately.
_TOKEN_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bmsg=[0-9a-f]{6,}"), "chat message / thread id (msg=...)"),
    (re.compile(r"æ"), "the literal `æ` annotation marker"),
    (re.compile(r"\bM\d+[a-z]?\b"), "a milestone identifier (e.g. M0 / M1a)"),
    (re.compile(r"\btask #\d+"), "a task id (task #N)"),
    (re.compile(r"\bPR #?\d+\b"), "a pull-request number (PR #N)"),
    (re.compile(r"\b(?:Alice|Bob|ZhengQiHang)\b"), "an agent / human name"),
    (
        re.compile(r"\bV\d+\b"),
        "a version stamp (e.g. V1 / V2); if this is a product identifier, "
        "rephrase or add a documented allow",
    ),
]

# Forbidden section-header terms (matched only on heading lines, so ordinary
# prose like "in the future" is never flagged).
_HEADER_TERMS = re.compile(
    r"\b(?:Non-goals?|非目标|Future|TODO|Out of scope|Tests|Testing|"
    r"Test plan|测试要求)\b",
    re.IGNORECASE,
)
_HEADING = re.compile(r"^\s*#{1,6}\s")


def lint_text(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, message)`` for every violation in *text*."""
    violations: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, message in _TOKEN_RULES:
            if pattern.search(line):
                violations.append((lineno, message))
        if _HEADING.match(line):
            m = _HEADER_TERMS.search(line)
            if m:
                violations.append(
                    (lineno, f"a forbidden section header ({m.group(0)})")
                )
    return violations


def lint_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return [f"{path}:{ln}: {msg}" for ln, msg in lint_text(text)]


def main(argv: list[str]) -> int:
    failures: list[str] = []
    for path in argv:
        failures.extend(lint_file(path))
    if failures:
        sys.stderr.write(
            "spec_rules_lint: docs/spec violates docs/SPEC-RULES.md:\n"
        )
        for f in failures:
            sys.stderr.write(f"  {f}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
