"""线性卡尔曼滤波单测：解析一步 + Joseph 一致性 + filter 长度。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from koopman.estimation import LinearKalmanFilter


class TestLinearKalmanFilter(unittest.TestCase):
    def test_scalar_one_step_analytic(self) -> None:
        """标量系统手工推一步，与实现对比。"""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.25]])
        R = np.array([[1.0]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        u = np.array([0.0])
        B = np.array([[0.0]])
        z = np.array([2.0])

        # 手工：predict
        x_m = F @ x0  # 0
        P_m = F @ P0 @ F.T + Q  # 1.25
        # update
        S = H @ P_m @ H.T + R  # 2.25
        K = P_m @ H.T / S  # 1.25/2.25
        x_p = x_m + K * (z - H @ x_m)
        L = 1.0 - K * H
        P_joseph = L * P_m * L + K * R * K

        kf = LinearKalmanFilter(F, H, Q, R, x0, P0, B=B, joseph=True)
        kf.predict(u)
        x_hat, P_hat, innov, S_hat = kf.update(z)

        np.testing.assert_allclose(x_hat.ravel(), x_p.ravel(), atol=1e-12)
        np.testing.assert_allclose(P_hat, P_joseph, atol=1e-12)
        np.testing.assert_allclose(innov.ravel(), (z - H @ x_m).ravel(), atol=1e-12)
        np.testing.assert_allclose(S_hat, S, atol=1e-12)

    def test_2d_one_step_analytic(self) -> None:
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.diag([0.1, 0.2])
        R = np.array([[0.5]])
        x0 = np.array([1.0, 0.5])
        P0 = np.diag([2.0, 1.0])
        z = np.array([1.2])

        x_m = F @ x0
        P_m = F @ P0 @ F.T + Q
        S = H @ P_m @ H.T + R
        K = P_m @ H.T @ np.linalg.inv(S)
        innov = z - H @ x_m
        x_p = x_m + (K @ innov.reshape(-1, 1)).ravel()
        I = np.eye(2)
        L = I - K @ H
        P_j = L @ P_m @ L.T + K @ R @ K.T

        kf = LinearKalmanFilter(F, H, Q, R, x0, P0, joseph=True)
        kf.predict()
        x_hat, P_hat, innov_hat, S_hat = kf.update(z)

        np.testing.assert_allclose(x_hat.ravel(), x_p, atol=1e-10)
        np.testing.assert_allclose(P_hat, P_j, atol=1e-10)
        np.testing.assert_allclose(innov_hat.ravel(), innov, atol=1e-10)
        np.testing.assert_allclose(S_hat, S, atol=1e-10)

    def test_joseph_vs_naive_close(self) -> None:
        F = np.array([[1.0, 0.5], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.diag([0.01, 0.01])
        R = np.array([[0.1]])
        x0 = np.zeros(2)
        P0 = np.eye(2)
        z = np.array([0.3])

        kf_j = LinearKalmanFilter(F, H, Q, R, x0, P0, joseph=True)
        kf_n = LinearKalmanFilter(F, H, Q, R, x0, P0, joseph=False)
        kf_j.predict()
        kf_n.predict()
        _, Pj, _, _ = kf_j.update(z)
        _, Pn, _, _ = kf_n.update(z)
        np.testing.assert_allclose(Pj, Pn, atol=1e-10)

    def test_filter_length(self) -> None:
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[1.0]])
        zs = np.array([0.0, 1.0, 0.5, -0.2])
        kf = LinearKalmanFilter(F, H, Q, R, x0=[0.0], P0=[[1.0]])
        out = kf.filter(zs)
        self.assertEqual(out["x"].shape, (4, 1))
        self.assertEqual(out["P"].shape, (4, 1, 1))
        self.assertEqual(out["innov"].shape, (4, 1))
        self.assertEqual(out["S"].shape, (4, 1, 1))

    def test_filter_with_control(self) -> None:
        F = np.array([[1.0]])
        B = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.1]])
        zs = np.array([1.0, 2.0, 3.0])
        us = np.array([0.5, 0.5, 0.5])
        kf = LinearKalmanFilter(F, H, Q, R, x0=[0.0], P0=[[1.0]], B=B)
        out = kf.filter(zs, us)
        self.assertEqual(len(out["x"]), 3)
        # 有正控制且观测递增，终态应明显为正
        self.assertGreater(float(out["x"][-1, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
