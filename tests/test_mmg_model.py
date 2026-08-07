#!/usr/bin/env python3
"""MMG 模型与残差混合模型的单元测试。

覆盖：
1. 最小二乘辨识能从合成数据（已知参数的 MMG 前向仿真）精确恢复参数；
2. MmgModel / MmgResidualModel 的步进形状与接口；
3. 残差 MLP 零初始化 ⇒ 混合模型严格等于 MMG 基线（技术方案 §3.4 约定）；
4. PhysStepAdapter 的 evalkit 接口；
5. 参数 npz 保存/加载往返一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from koopman.mmg_model import (
    N_MMG_PARAMS,
    MmgModel,
    PhysStepAdapter,
    compute_train_stats,
    least_squares_fit,
    load_mmg_npz,
    save_mmg_npz,
)
from koopman.mmg_residual import MmgResidualModel

THETA_TRUE = np.array([
    -2.5e-3, -5.0e-3, 4.0e-4,                      # surge
    -4.0e-2, 3.0e-1, -1.5e-1, 2.0e1, -2.5e-2, -1.5e-4,  # sway
    2.0e-4, -6.0e-2, -5.0e-3, 4.0e-5, -3.0e-5, 4.0e-6,  # yaw
], dtype=np.float64)
assert len(THETA_TRUE) == N_MMG_PARAMS


def _prbs(rng: np.random.Generator, T: int, lo: float, hi: float,
          min_hold: int = 30, max_hold: int = 300) -> np.ndarray:
    """随机电报信号（PRBS 风格）：分段常值、随机切换，频谱宽、激励充分。"""
    x = np.zeros(T)
    t = 0
    while t < T:
        hold = int(rng.integers(min_hold, max_hold))
        x[t:t + hold] = rng.uniform(lo, hi)
        t += hold
    return x


def _synth_segments(theta: np.ndarray, n_seg: int = 4, T: int = 1500, dt: float = 0.1,
                    seed: int = 0):
    """用已知参数的 MMG 前向仿真生成 (states (T,6), ctrls (T,4)) 段。

    4 个控制通道用相互独立的 PRBS 激励（含换舷/倒车/满舵），尽量解相关。
    """
    rng = np.random.default_rng(seed)
    model = MmgModel(theta=theta)
    segs = []
    for _ in range(n_seg):
        cp = _prbs(rng, T, -60, 100)
        cs = _prbs(rng, T, -60, 100)
        dp = _prbs(rng, T, -35, 35, min_hold=10, max_hold=100)
        ds = _prbs(rng, T, -35, 35, min_hold=10, max_hold=100)
        ctr = np.stack([cp, dp, cs, ds], axis=1).astype(np.float32)
        dyn = torch.zeros(T, 3)
        x = torch.tensor([1.5, 0.0, 0.0])
        with torch.no_grad():
            for k in range(T):
                dyn[k] = x  # 先存当前状态：与真实数据一致，x_{k+1} = f(x_k, c_k)
                x = model.step_phys(x.view(1, 3), torch.from_numpy(ctr[k]).view(1, 4), dt).view(3)
        st = np.zeros((T, 6), dtype=np.float32)
        st[:, 3:6] = dyn.numpy()
        segs.append((st, ctr))
    return segs


def test_least_squares_recovers_synth_params() -> None:
    # 合成数据来自精确 Euler（sub_dt == data_dt），smooth=1 时差分无失真。
    # 注意：r 与 r|r| 类特征存在结构性相关（系统辨识的固有病态），逐参数
    # 精确恢复不可期望；断言 (1) 回归残差≈0（特征结构正确）、(2) 执行器
    # 增益等可辨识参数精确恢复、(3) 拟合参数与真值前向仿真预测等价。
    segs = _synth_segments(THETA_TRUE)
    theta_hat, report = least_squares_fit(segs, data_dt=0.1, smooth=1, ridge=1e-12)

    # (1) 一步回归残差 ≈ 0（模型类正确）。合成轨迹以 float32 计算/存储，
    # 量化残差下限 ~5e-7（surge 加速度幅值最大、残差最大），阈值取 1e-6
    # 仍可捕捉结构性错误（符号/漏项 ⇒ 残差 O(加速度幅值)~1e-1）。
    for ch in ("surge", "sway", "yaw"):
        assert report[ch]["rmse"] < 1e-6, f"{ch} rmse={report[ch]['rmse']:.2e}"
        assert report[ch]["r2"] > 0.9999, f"{ch} R²={report[ch]['r2']}"

    # (2) 执行器增益（独立激励下可辨识）精确恢复
    names = ["k_tx", "k_ty", "k_tn_lat", "k_tn_diff"]
    idx = [2, 8, 13, 14]
    for n, i in zip(names, idx):
        rel = abs(theta_hat[i] - THETA_TRUE[i]) / abs(THETA_TRUE[i])
        assert rel < 1e-3, f"{n}: hat={theta_hat[i]:.3e} true={THETA_TRUE[i]:.3e}"

    # (3) 预测等价：拟合参数与真值在新激励下 60s 开环 rollout 几乎一致
    segs_fresh = _synth_segments(THETA_TRUE, n_seg=1, T=600, seed=99)
    ctr = torch.from_numpy(segs_fresh[0][1])
    m_true, m_hat = MmgModel(theta=THETA_TRUE), MmgModel(theta=theta_hat)
    x_true, x_hat = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0])
    with torch.no_grad():
        for k in range(ctr.shape[0]):
            x_true = m_true.step_phys(x_true.view(1, 3), ctr[k].view(1, 4), 0.1).view(3)
            x_hat = m_hat.step_phys(x_hat.view(1, 3), ctr[k].view(1, 4), 0.1).view(3)
    dev = (x_true - x_hat).abs()
    assert dev.max() < 1e-4, f"60s rollout 偏差 {dev}"


def test_mmg_step_shapes() -> None:
    model = MmgModel(theta=THETA_TRUE)
    dyn = torch.randn(7, 3)
    ctrl = torch.randn(7, 4) * 10
    a = model.accel(dyn, ctrl)
    assert a.shape == (7, 3)
    nxt = model.step_phys(dyn, ctrl, 0.5)
    assert nxt.shape == (7, 3)
    # 零控制 + 零状态 ⇒ 零加速度
    z = torch.zeros(1, 3)
    c = torch.zeros(1, 4)
    assert torch.allclose(model.accel(z, c), z)


def _dummy_stats() -> dict:
    rng = np.random.default_rng(1)
    states = rng.normal(size=(500, 6)).astype(np.float32) * [10, 10, 3, 1.5, 0.2, 0.02]
    ctrls = rng.normal(size=(500, 4)).astype(np.float32) * [40, 20, 40, 20] + [50, 0, 50, 0]
    return compute_train_stats(states, ctrls), states, ctrls


def test_residual_zero_init_equals_baseline() -> None:
    stats, _, _ = _dummy_stats()
    mmg = MmgModel(theta=THETA_TRUE)
    hybrid = MmgResidualModel(MmgModel(theta=THETA_TRUE), stats)
    dyn = torch.randn(16, 3) * torch.tensor([2.0, 0.3, 0.02])
    ctrl = torch.rand(16, 4) * torch.tensor([100.0, 35.0, 100.0, 35.0])
    assert torch.allclose(hybrid.accel(dyn, ctrl), mmg.accel(dyn, ctrl), atol=1e-7)
    assert torch.allclose(hybrid.step_phys(dyn, ctrl, 1.0), mmg.step_phys(dyn, ctrl, 1.0), atol=1e-6)
    # MMG 默认冻结
    assert all(not p.requires_grad for p in hybrid.mmg.parameters())


def test_adapter_interface() -> None:
    stats, _, _ = _dummy_stats()
    mmg = MmgModel(theta=THETA_TRUE)
    adapter = PhysStepAdapter(lambda d, c: mmg.step_phys(d, c, 1.0), stats)
    dyn_std = torch.tensor(stats["state_std"][3:6])
    dyn_mean = torch.tensor(stats["state_mean"][3:6])
    z0 = (torch.randn(5, 3) - dyn_mean) / dyn_std
    u_n = torch.randn(5, 4)
    z1 = adapter.latent_step(adapter.encode(z0), u_n)
    assert z1.shape == (5, 3)
    rec = adapter.reconstruct_state(z1)
    assert torch.equal(rec, z1)


def test_npz_roundtrip(tmp_path: Path = Path("/tmp")) -> None:
    stats, _, _ = _dummy_stats()
    p = str(tmp_path / "mmg_test.npz")
    save_mmg_npz(p, THETA_TRUE, stats, {"surge": {"r2": 1.0}})
    theta, stats2, report = load_mmg_npz(p)
    np.testing.assert_array_equal(theta, THETA_TRUE)
    for k in stats:
        np.testing.assert_array_equal(stats[k], stats2[k])
    assert report["surge"]["r2"] == 1.0


if __name__ == "__main__":
    test_least_squares_recovers_synth_params()
    test_mmg_step_shapes()
    test_residual_zero_init_equals_baseline()
    test_adapter_interface()
    test_npz_roundtrip()
    print("all mmg tests passed")
