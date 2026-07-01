# TileFoundry Spec — Target

A `Target` is the back-end a verified, lowered `tir.PrimFunction` is compiled
for. It is a capability descriptor: it names the back-end and carries the
back-end's compile-time parameters and the program topology levels it admits. A
function carries a single `target`, which selects its target-specific lowering
and codegen. Emitting source and linking the artifact are owned by
[codegen](./codegen.md); loading the artifact as a `RuntimeModule` is owned by
[runtime](./runtime.md).

## 1. Role and scope

- **Input** is verified TIR. HIR Ops MUST NOT reach a `Target`.
- A `Target` describes capability; it does not emit source, link, run passes,
  load or launch device code, or own the user-facing entry points
  (`compile` / `build` / `jit`).
- A target is resolved by name: a string reflects into that back-end's default
  target object (`"cuda"` → `CudaTarget()`), and a `Target` object passes
  through unchanged. Codegen groups a module's functions by their target name
  and emits one `LinkableModule` per group
  ([codegen §1](./codegen.md#1-pipeline)).

## 2. `Target`

```python
@dataclass(frozen=True)
class Target:
    name: str
```

#### `name`
- MUST be the stable back-end identifier used for target resolution and for the
  function-target grouping in codegen.

A `Target` MUST NOT carry the linkable / linked artifact dataclasses; those are
codegen products ([codegen §4](./codegen.md#4-codegen-products)).

## 3. `CudaTarget`

CUDA is the current reference target.

```python
@dataclass(frozen=True)
class CudaTarget(Target):
    name: str = "cuda"
    arch: str = "sm_90"
    topology_levels: tuple[str, ...] = ("cta", "thread")
```

#### `name`
- MUST be `"cuda"`.

#### `arch`
- MUST name the SM architecture the device source is compiled for.

#### `topology_levels`
- MUST be the program topology level set the target admits: `{cta, thread}`.
- `warp` / `lane` / `warpgroup` MUST be expressed as axes of a thread mesh
  layout ([shard](./shard.md)), not as program topology levels.
- A function whose declared program topology levels are not a subset of this set
  MUST raise at lowering. The level set is consumed by the device program
  accessors ([codegen §6](./codegen.md#6-program-shape-and-dynamic-cta)).

## 4. `CpuTarget`

```python
@dataclass(frozen=True)
class CpuTarget(Target):
    name: str = "cpu"
```

#### `name`
- MUST be `"cpu"`.

The CPU target hosts the entry that marshals arguments and invokes the device
entry through its C-ABI launch shim
([codegen §3](./codegen.md#3-target-driven-emission)).
