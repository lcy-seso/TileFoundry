"""End-to-end cross-warp reduce.

Reducing a tensor axis whose Split maps to a warp-only mesh axis (each lane
keeps its own output cell) selects the ``reduce_cross_warp`` runtime template —
distinct from ``reduce_intra_cta`` (used when a reduced axis folds within a
warp, e.g. RMSNorm). Full GPU compile + run + numerical compare.
"""

import torch

import tilefoundry
from tilefoundry import module
from tilefoundry.dsl import *  # Tensor, tf, Mesh, Topology, ReduceKind, func, ...


@module(entry="cross_warp_sum")
class CrossWarpSumModule:
    @func(topologies=(Topology("thread", 4 * 32),))
    def cross_warp_sum(a: Tensor[(4, 32), 'f32']):
        with Mesh(Topology("thread", 4 * 32), (4, 32), ('tk', 'hc')) as m:
            # Axis 0 (tk) spans the four warps; axis 1 (hc) is the lane axis and
            # carries distinct output cells. Reducing axis 0 crosses warps only.
            a_reg = tf.reshard(a, (4 @ m.tk, 32 @ m.hc), 'rmem')
            s = tf.reduce(a_reg, (0,), True, ReduceKind.SUM)
            return tf.reshard(s, (1, 32 @ m.hc), 'gmem')


def test_cross_warp_sum_matches_torch() -> None:
    """The warp-only reduction lowers to ``reduce_cross_warp`` (selected from the
    operand layouts) and matches a torch axis-0 sum."""
    rm = tilefoundry.compile(CrossWarpSumModule, target="cuda")
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float32, device="cuda")
    out = rm(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x.sum(0, keepdim=True), rtol=1e-4, atol=1e-4)


def test_cross_warp_sum_emits_reduce_cross_warp() -> None:
    """Codegen selects the cross-warp template (not intra-cta) purely from the
    operand layouts."""
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(CrossWarpSumModule, target="cuda")
    src = emit_cuda_module(group_functions_by_target(lowered)["cuda"]).source
    assert "reduce_cross_warp" in src
    assert "reduce_intra_cta" not in src
