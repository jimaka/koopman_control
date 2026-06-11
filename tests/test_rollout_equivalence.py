"""回归测试：向量化后的 evalkit.rollout_dataset 必须与朴素实现逐位一致。

背景：rollout_dataset 的取数环节由「逐样本逐步 Python 双层循环」优化为
numpy fancy-indexing 一次 gather，并把循环内的逐步 device→host 拷贝合并为
每 batch 一次。两者都是纯拷贝/相同算子序列，结果应 bit-identical。
本测试用仓库自带 ckpt + 测试集对比两种实现，防止后续改动破坏数值一致性。

运行方式（仓库根目录）::

    python3 tests/test_rollout_equivalence.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koopman import evalkit as ek  # noqa: E402


@torch.no_grad()
def rollout_dataset_naive(
    model: nn.Module,
    states_full: np.ndarray,
    ctrls_full: np.ndarray,
    sample_global_t0: np.ndarray,
    pred_len: int,
    stats: Dict[str, np.ndarray],
    device: torch.device,
    dt: float,
    batch_size: int = 1024,
    model_stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """优化前的参考实现（逐样本逐步取数 + 逐步拷贝回 CPU）。"""
    state_mean = stats["state_mean"].astype(np.float32)
    state_std = stats["state_std"].astype(np.float32)
    ctrl_mean = stats["ctrl_mean"].astype(np.float32)
    ctrl_std = stats["ctrl_std"].astype(np.float32)

    dyn_mean_t = torch.tensor(state_mean[3:6], device=device)
    dyn_std_t = torch.tensor(state_std[3:6], device=device)
    ctrl_mean_t = torch.tensor(ctrl_mean, device=device)
    ctrl_std_t = torch.tensor(ctrl_std, device=device)

    M = sample_global_t0.shape[0]
    K = pred_len
    pred_dyn = np.empty((M, K, 3), dtype=np.float32)
    gt_dyn = np.empty((M, K, 3), dtype=np.float32)
    pred_xy = np.empty((M, K, 2), dtype=np.float32)
    gt_xy = np.empty((M, K, 2), dtype=np.float32)

    model.eval()
    for start in range(0, M, batch_size):
        end = min(M, start + batch_size)
        idx = sample_global_t0[start:end]
        b = idx.shape[0]
        x0 = states_full[idx]
        u_future = np.empty((b, K, 4), dtype=np.float32)
        gt_dyn_future = np.empty((b, K, 3), dtype=np.float32)
        ms = int(model_stride)
        for j, t0 in enumerate(idx):
            for k in range(K):
                u_future[j, k] = ctrls_full[t0 + k * ms]
                gt_dyn_future[j, k] = states_full[t0 + (k + 1) * ms, 3:6]

        x0_t = torch.from_numpy(x0).to(device)
        u_future_t = torch.from_numpy(u_future).to(device)
        u_norm = (u_future_t - ctrl_mean_t) / ctrl_std_t

        dyn0_norm = (x0_t[:, 3:6] - dyn_mean_t) / dyn_std_t
        z = model.encode(dyn0_norm)
        cur_x = x0_t[:, 0].clone()
        cur_y = x0_t[:, 1].clone()
        cur_yaw = x0_t[:, 2].clone()
        gt_x = x0_t[:, 0].clone()
        gt_y = x0_t[:, 1].clone()
        gt_yaw = x0_t[:, 2].clone()
        gt_dyn_future_t = torch.from_numpy(gt_dyn_future).to(device)

        for k in range(K):
            z = model.latent_step(z, u_norm[:, k, :])
            pred_phys = model.reconstruct_state(z) * dyn_std_t + dyn_mean_t
            up, vp, rp = pred_phys[:, 0], pred_phys[:, 1], pred_phys[:, 2]
            cur_x = cur_x + (up * torch.cos(cur_yaw) - vp * torch.sin(cur_yaw)) * dt
            cur_y = cur_y + (up * torch.sin(cur_yaw) + vp * torch.cos(cur_yaw)) * dt
            cur_yaw = cur_yaw + rp * dt
            pred_dyn[start:end, k, 0] = up.cpu().numpy()
            pred_dyn[start:end, k, 1] = vp.cpu().numpy()
            pred_dyn[start:end, k, 2] = rp.cpu().numpy()
            pred_xy[start:end, k, 0] = cur_x.cpu().numpy()
            pred_xy[start:end, k, 1] = cur_y.cpu().numpy()

            ug = gt_dyn_future_t[:, k, 0]
            vg = gt_dyn_future_t[:, k, 1]
            rg = gt_dyn_future_t[:, k, 2]
            gt_x = gt_x + (ug * torch.cos(gt_yaw) - vg * torch.sin(gt_yaw)) * dt
            gt_y = gt_y + (ug * torch.sin(gt_yaw) + vg * torch.cos(gt_yaw)) * dt
            gt_yaw = gt_yaw + rg * dt
            gt_dyn[start:end, k, 0] = ug.cpu().numpy()
            gt_dyn[start:end, k, 1] = vg.cpu().numpy()
            gt_dyn[start:end, k, 2] = rg.cpu().numpy()
            gt_xy[start:end, k, 0] = gt_x.cpu().numpy()
            gt_xy[start:end, k, 1] = gt_y.cpu().numpy()
    return gt_dyn, pred_dyn, gt_xy, pred_xy


def _run_case(ckpt: str, data: str, pred_len: int, dt: float, model_stride: int,
              max_samples: int, batch_size: int) -> None:
    device = torch.device("cpu")
    model, stats = ek.load_model_from_ckpt(ckpt, device)
    states_full, ctrls_full, _, _, t0g, _, _ = ek._flatten_segments(
        data, pred_len=pred_len, stride=1, model_stride=model_stride
    )
    if t0g.shape[0] > max_samples:
        sel = np.linspace(0, t0g.shape[0] - 1, max_samples).astype(int)
        t0g = t0g[sel]
    args = (model, states_full, ctrls_full, t0g, pred_len, stats, device, dt)
    kw = dict(batch_size=batch_size, model_stride=model_stride)
    ref = rollout_dataset_naive(*args, **kw)
    new = ek.rollout_dataset(*args, **kw)
    names = ("gt_dyn", "pred_dyn", "gt_xy", "pred_xy")
    for name, a, b in zip(names, ref, new):
        assert np.array_equal(a, b), f"{name} mismatch (ckpt={ckpt}, K={pred_len})"
    print(f"  OK ckpt={ckpt} K={pred_len} ms={model_stride} M={t0g.shape[0]} -> bit-identical")


def main() -> int:
    print("[test_rollout_equivalence] case 1: v1 ckpt, dt=0.1, K=20")
    _run_case("checkpoints/koopman_v1_best.pth", "data/koopman_test.npz",
              pred_len=20, dt=0.1, model_stride=1, max_samples=2048, batch_size=512)
    print("[test_rollout_equivalence] case 2: v4 ckpt, dt=1.0 (model_stride=10), K=20")
    _run_case("checkpoints/koopman_v4_best.pth", "data/koopman_test.npz",
              pred_len=20, dt=1.0, model_stride=10, max_samples=512, batch_size=128)
    print("[test_rollout_equivalence] ALL PASS")
    return 0


if __name__ == "__main__":
    import os

    os.chdir(REPO_ROOT)
    raise SystemExit(main())
