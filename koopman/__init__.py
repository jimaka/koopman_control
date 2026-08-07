"""Deep-Koopman 船舶动力学 Python 包。

模型类按需惰性导入，避免 ``import koopman.estimation`` 等轻量路径强制依赖 torch。
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "BaseKoopmanModel",
    "HorizontalKoopmanModel",
    "HorizontalKoopmanModelV3",
    "FEATURE_DICT_ATOMS",
    "ResidualConvBlock",
    "res_mlp",
]

_LAZY_V1_V2 = {
    "BaseKoopmanModel",
    "HorizontalKoopmanModel",
    "ResidualConvBlock",
    "res_mlp",
}
_LAZY_V3 = {
    "FEATURE_DICT_ATOMS",
    "HorizontalKoopmanModelV3",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_V1_V2:
        from koopman import model_v1_v2 as m

        return getattr(m, name)
    if name in _LAZY_V3:
        from koopman import model_v3 as m

        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
