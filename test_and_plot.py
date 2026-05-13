import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from koopman import HorizontalKoopmanModel
from train_multistep_voyage import ExplicitKoopmanDataset

def rollout_prediction(model, dyn_init, u_seq, pred_len):
    current_state = dyn_init
    pred_states = []
    latent_norms = []

    for step in range(pred_len):
        z = model.encode(current_state)
        latent_norms.append(torch.norm(z, dim=-1).mean().item())
        z_next = model.latent_step(z, u_seq[:, step, :])
        pred_state = model.reconstruct_state(z_next)
        pred_states.append(pred_state)
        current_state = pred_state

    return torch.stack(pred_states, dim=1), latent_norms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/koopman_best.pth"
    npz_path = "koopman_test.npz"
    pred_len = 20
    batch_size = 2048
    dt = 0.1

    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到模型文件 {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats = checkpoint["stats"]

    state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32, device=device)
    state_std = torch.tensor(stats["state_std"], dtype=torch.float32, device=device)
    dyn_mean, dyn_std = state_mean[3:6], state_std[3:6]

    # 【核心更新】：加载带物理先验的 Koopman (hidden_dim=24)
    model = HorizontalKoopmanModel(state_dim=3, hidden_dim=24)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_ds = ExplicitKoopmanDataset(npz_path, pred_len=pred_len, stats=stats, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_gt_full, all_pred_full, all_latent_norms = [], [], []

    print("🚀 开始 物理先验网络 + 外部数学积分 多步推演...")
    with torch.no_grad():
        for x_t_norm, x_target_seq_norm, u_seq in test_loader:
            x_t_norm, x_target_seq_norm, u_seq = x_t_norm.to(device), x_target_seq_norm.to(device), u_seq.to(device)

            x_t_phys = x_t_norm * state_std + state_mean
            current_x, current_y, current_yaw = x_t_phys[:, 0], x_t_phys[:, 1], x_t_phys[:, 2]

            dyn_init = x_t_norm[:, 3:6]
            pred_dyn_norm, latent_norms = rollout_prediction(model, dyn_init, u_seq, pred_len)
            pred_dyn_phys = pred_dyn_norm * dyn_std + dyn_mean

            pred_full_phys = []
            for step in range(pred_len):
                u, v, r = pred_dyn_phys[:, step, 0], pred_dyn_phys[:, step, 1], pred_dyn_phys[:, step, 2]
                next_x = current_x + (u * torch.cos(current_yaw) - v * torch.sin(current_yaw)) * dt
                next_y = current_y + (u * torch.sin(current_yaw) + v * torch.cos(current_yaw)) * dt
                next_yaw = current_yaw + r * dt
                
                step_state = torch.stack([next_x, next_y, next_yaw, u, v, r], dim=1)
                pred_full_phys.append(step_state)
                current_x, current_y, current_yaw = next_x, next_y, next_yaw
                
            pred_full_seq = torch.stack(pred_full_phys, dim=1)
            gt_full_phys = x_target_seq_norm * state_std + state_mean

            all_gt_full.append(gt_full_phys.cpu().numpy())
            all_pred_full.append(pred_full_seq.cpu().numpy())
            all_latent_norms.extend(latent_norms)

    full_gt = np.concatenate(all_gt_full, axis=0)
    full_pred = np.concatenate(all_pred_full, axis=0)

    # 误差计算
    vel_error = np.sqrt((full_gt[:, :, 3] - full_pred[:, :, 3]) ** 2 + (full_gt[:, :, 4] - full_pred[:, :, 4]) ** 2)
    mean_vel_error = np.mean(vel_error, axis=0)

    gt_acc = (full_gt[:, 1:, 3:6] - full_gt[:, :-1, 3:6]) / dt
    pred_acc = (full_pred[:, 1:, 3:6] - full_pred[:, :-1, 3:6]) / dt
    acc_error = np.sqrt(np.sum((gt_acc - pred_acc) ** 2, axis=-1))
    mean_acc_error = np.mean(acc_error, axis=0)

    save_dir = "test_analysis"
    os.makedirs(save_dir, exist_ok=True)
    sample_indices = np.linspace(0, len(full_gt) - 1, 6, dtype=int)

    # Plot 1: Trajectory
    plt.figure(figsize=(18, 12))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i + 1)
        plt.plot(full_gt[idx, :, 0], full_gt[idx, :, 1], "g-o", label="GT Path", alpha=0.7)
        plt.plot(full_pred[idx, :, 0], full_pred[idx, :, 1], "r--x", label="PI-Koopman")
        plt.title(f"Sample {idx} - Trajectory")
        plt.axis('equal'); plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "trajectory_comparison.png"))

    # Plot 2: Velocity U
    plt.figure(figsize=(18, 12))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i + 1)
        plt.plot(full_gt[idx, :, 3], "g-o", label="GT Surge (u)", alpha=0.7)
        plt.plot(full_pred[idx, :, 3], "r--x", label="Pred u")
        plt.title(f"Sample {idx} - Surge Velocity (u)")
        plt.xlabel("Step"); plt.ylabel("Velocity [m/s]"); plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "u_prediction.png"))

    # Plot 3 & 4: Errors
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, pred_len + 1), mean_vel_error, marker="o", color='blue', linewidth=2)
    plt.xlabel("Step"); plt.ylabel("Velocity Error [m/s]"); plt.grid(True)
    plt.title("Velocity Error Accumulation (PI-Koopman)")
    plt.savefig(os.path.join(save_dir, "velocity_error_curve.png"), dpi=200)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, pred_len), mean_acc_error, marker="s", color='darkorange', linewidth=2)
    plt.xlabel("Step"); plt.ylabel("Acceleration Error [$m/s^2$]"); plt.grid(True)
    plt.savefig(os.path.join(save_dir, "acceleration_error_curve.png"), dpi=200)

    print("\n" + "=" * 60)
    print(f"📊 平均速度误差 : {np.mean(vel_error):.6f} m/s")
    print(f"🎯 第20步速度误差 : {mean_vel_error[-1]:.6f} m/s")
    print(f"⚡ 平均加速度误差 : {np.mean(acc_error):.6f} m/s^2")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()