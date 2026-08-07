"""线性卡尔曼滤波参考实现。

详见 docs/卡尔曼滤波推导.md。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_col(x: ArrayLike, n: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1, 1)
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"expected length {n}, got {arr.shape[0]}")
    return arr


def _as_2d(M: ArrayLike, shape: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(M, dtype=float)
    if arr.ndim == 1 and shape[0] == shape[1] and arr.size == shape[0]:
        arr = np.diag(arr)
    arr = np.atleast_2d(arr).astype(float)
    if arr.shape != shape:
        raise ValueError(f"expected shape {shape}, got {arr.shape}")
    return arr


class LinearKalmanFilter:
    """线性离散卡尔曼滤波器（预测 / 更新 / 批量滤波）。

    模型::

        x_k = F x_{k-1} + B u_{k-1} + w,   w ~ N(0, Q)
        z_k = H x_k + v,                   v ~ N(0, R)

    默认用 Joseph 形式更新协方差；``joseph=False`` 时用朴素形式 ``(I-KH)P``。
    """

    def __init__(
        self,
        F: ArrayLike,
        H: ArrayLike,
        Q: ArrayLike,
        R: ArrayLike,
        x0: ArrayLike,
        P0: ArrayLike,
        B: Optional[ArrayLike] = None,
        *,
        joseph: bool = True,
    ) -> None:
        F_arr = np.atleast_2d(np.asarray(F, dtype=float))
        n = F_arr.shape[0]
        if F_arr.shape != (n, n):
            raise ValueError(f"F must be square, got {F_arr.shape}")

        H_arr = np.atleast_2d(np.asarray(H, dtype=float))
        p = H_arr.shape[0]
        if H_arr.shape[1] != n:
            raise ValueError(f"H columns must match state dim {n}, got {H_arr.shape}")

        self.F = F_arr
        self.H = H_arr
        self.Q = _as_2d(Q, (n, n))
        self.R = _as_2d(R, (p, p))
        self.n = n
        self.p = p
        self.joseph = bool(joseph)

        if B is None:
            self.B = None
            self.m = 0
        else:
            B_arr = np.atleast_2d(np.asarray(B, dtype=float))
            if B_arr.shape[0] != n:
                raise ValueError(f"B rows must match state dim {n}, got {B_arr.shape}")
            self.B = B_arr
            self.m = B_arr.shape[1]

        self.x = _as_col(x0, n)
        self.P = _as_2d(P0, (n, n))

    def predict(self, u: Optional[ArrayLike] = None) -> Tuple[np.ndarray, np.ndarray]:
        """时间更新：先验均值与协方差。"""
        if self.B is None:
            if u is not None:
                raise ValueError("control u given but B is None")
            self.x = self.F @ self.x
        else:
            if u is None:
                u_col = np.zeros((self.m, 1))
            else:
                u_col = _as_col(u, self.m)
            self.x = self.F @ self.x + self.B @ u_col
        self.P = self.F @ self.P @ self.F.T + self.Q
        # 数值对称化
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy()

    def update(
        self, z: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """测量更新：后验均值、协方差、创新与创新协方差。"""
        z_col = _as_col(z, self.p)
        innov = z_col - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        S = 0.5 * (S + S.T)

        # K = P H^T S^{-1}，用 solve 避免显式 inv
        PHt = self.P @ self.H.T
        K = np.linalg.solve(S.T, PHt.T).T

        self.x = self.x + K @ innov
        I = np.eye(self.n)
        if self.joseph:
            L = I - K @ self.H
            self.P = L @ self.P @ L.T + K @ self.R @ K.T
        else:
            self.P = (I - K @ self.H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy(), innov.copy(), S.copy()

    def filter(
        self,
        zs: ArrayLike,
        us: Optional[ArrayLike] = None,
        *,
        predict_first: bool = True,
    ) -> Dict[str, Any]:
        """对观测序列逐拍滤波。

        Parameters
        ----------
        zs :
            形状 ``(T, p)`` 或 ``(T,)``（标量观测）。
        us :
            可选控制，形状 ``(T, m)``；第 ``k`` 步用于从 ``k-1`` 预测到 ``k``。
        predict_first :
            True（默认）时每步先 ``predict`` 再 ``update``；
            False 时第一步先 ``update`` 初值再用后续 predict/update。
        """
        zs_arr = np.asarray(zs, dtype=float)
        if zs_arr.ndim == 1:
            zs_arr = zs_arr.reshape(-1, 1)
        if zs_arr.ndim != 2 or zs_arr.shape[1] != self.p:
            raise ValueError(f"zs must have shape (T, {self.p}), got {zs_arr.shape}")
        T = zs_arr.shape[0]

        us_arr: Optional[np.ndarray] = None
        if us is not None:
            us_arr = np.asarray(us, dtype=float)
            if us_arr.ndim == 1:
                us_arr = us_arr.reshape(-1, 1)
            if us_arr.shape != (T, self.m):
                raise ValueError(f"us must have shape (T, {self.m}), got {us_arr.shape}")

        xs = np.zeros((T, self.n))
        Ps = np.zeros((T, self.n, self.n))
        innovs = np.zeros((T, self.p))
        Ss = np.zeros((T, self.p, self.p))

        for k in range(T):
            u_k = None if us_arr is None else us_arr[k]
            if predict_first:
                self.predict(u_k)
            elif k > 0:
                self.predict(u_k)
            _, _, innov, S = self.update(zs_arr[k])
            xs[k] = self.x.ravel()
            Ps[k] = self.P
            innovs[k] = innov.ravel()
            Ss[k] = S

        return {
            "x": xs,
            "P": Ps,
            "innov": innovs,
            "S": Ss,
        }
