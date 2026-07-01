from __future__ import annotations

import enum


class DType(enum.Enum):
    f32 = "f32"
    f16 = "f16"
    bf16 = "bf16"
    fp8e4m3 = "fp8e4m3"
    i32 = "i32"
    i64 = "i64"
    bool = "bool"


__all__ = ["DType"]
