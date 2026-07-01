from __future__ import annotations

from ..core.registry import register_typeinfer
from .dim import (
    DimAdd,
    DimConst,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
)
from .shard.layout import EMPTY_LAYOUT
from .tensor_type import DType, TensorType


def _meta_i64() -> TensorType:
    return TensorType(shape=(), dtype=DType.i64, layout=EMPTY_LAYOUT, storage=None)


for _cls in (DimConst, DimVar, DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax):

    @register_typeinfer(_cls)
    def _(call, ctx, _cls=_cls):  # noqa: ARG001 — uniform signature
        return _meta_i64()
