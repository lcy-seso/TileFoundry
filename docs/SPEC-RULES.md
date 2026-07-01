# TileFoundry Spec — Rules

How `docs/spec/*.md` is written. Principle first; constraints below.
No fixed template — pick the form that fits each construct.

## Principle

**Simplicity above all.** A spec section reads like a contract, not an
implementation log. If one sentence is enough, do not write a section.

A spec MUST be written in English. Existing zh-CN passages SHOULD be
rewritten on first touch.

## Section structure

A construct that is a `@dataclass` (or has the moral equivalent —
named fields with stable identity) MUST be introduced as:

1. one short opening sentence — what it is, what role it plays
2. the `@dataclass` code block as the definition of truth
3. a multi-level heading per field, in the order the fields appear
   in the dataclass, with sub-headings for each named sub-field that
   carries its own contract
4. additional sub-sections (cross-construct invariants, examples,
   an optional design-rationale section — see below) only after the
   field walk-through, never interleaved

Each field section MUST be written in formal style: bullet rules with
`MUST` / `MAY` / `SHOULD` (RFC 2119), equations, and set / index
notation. Outside the optional design-rationale section below, avoid
explanatory prose paragraphs ("here X means …", "the reason is …"); if a
paragraph reads like commentary rather than a rule or a definition, drop it.

A field section MAY include a short worked example after the rules,
when an example pins down a corner of the contract that the rules
alone leave ambiguous.

A construct's section MAY carry one **Design rationale** subsection,
placed after the field walk-through, when the *why* behind the contract
is not obvious from the rules and is worth preserving. Keep it to a
sentence or two that point at the intent — not a
restatement of the rules, not an essay. It is the one place commentary
is allowed; it MUST still be English. A routine construct needs none.

## Constraints

A spec section MAY reference other spec sections (`<file> §X.Y`) or
external public knowledge (with a stable URL). References are inline
where they are used; do not maintain a "Related specs" or "See also"
header / footer block.

A spec section MUST NOT reference any of:

- a plan under `docs/plans/`
- a milestone identifier (`M0`, `M1a`, `(M3 sync)`, ...)
- a task ID (`task #87`, `#73`, ...)
- a commit hash, pull-request number, or other VCS coordinate
- an agent name (Alice, Bob, ...) or human name
- a chat thread / message ID
- the literal `æ` annotation marker
- a version stamp (`V1`, `V2`). Spec records the
  current contract; previous shapes are not in scope.

A spec section MUST NOT carry milestone or sync markers (`(M0 sync)`
and the like) in its title.

A spec MUST NOT carry a `Non-goals` / 非目标 / "Out of scope" /
"Future / TODO" section, nor inline prose that catalogues what the
spec deliberately does not cover ("we do not consider X", "X is left
for future work", etc.). What is not in the repo is not part of the
contract; listing exclusions only invites confusion. State the
positive contract; if a boundary needs to be drawn, draw it inline
where the relevant construct is defined.

A spec MUST NOT carry a `测试要求` / `Tests` / `Testing` /
"Test plan" section, nor a list of test names that should pass. The
spec records the contract; tests live in `tests/` and are owned by
the test suite. If a test name appears in spec text, drop it.

A spec MUST NOT embed implementation detail or a recipe that does not
belong to its own owning surface. Each construct, and each cross-layer
translation recipe, lives in exactly one owning spec. A lowering
recipe lives in the lowering-pass spec when that pass owns the
translation (e.g. the cute MMA fragment → `ShardLayout` rewrite lives
in the `HirToTirPass` spec); a target-side emission detail lives in the
target / codegen spec.

A construct has exactly one owning section. Every other spec
references it and MUST NOT restate its definition or normative rules:
a shared fact lives once, at its owner, and is linked — never copied.

A consensus Op (`add`, `relu`, ...) is one line plus a stable
external link. A custom Op spells out responsibility, fields, and
the verifier-relevant constraints. Field-by-field tables and
`ParamDef` listings stay in code; the spec records the contract.

For finite enumerations (dtype, storage class, ...) state the rule
and a representative subset; exhaustive enumeration is not required.

Use **MUST** / **MUST NOT** / **SHOULD** / **MAY** (RFC 2119) only
inside contract sentences.

## Code-side back references

A code module that implements a spec construct SHOULD carry a single
docstring line:

```
Spec: <file> §X.Y
```

The reverse index is `grep`. There is no central registry. Internal
helpers and private utilities MAY omit the anchor.

Spec drives plan, not the other way round. When implementation
reveals that a contract is wrong, the fix lands on the spec first.
