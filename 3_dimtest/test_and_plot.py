import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from koopman import HorizontalKoopmanModel

class ExplicitKoopmanDataset(Dataset):
    def __init__(self, npz_path, pred_len=20, stats=None):
        self.segments = np.load(npz_path, allow_pickle=True)['datas']
        self.pred_len = pred_len
        self.indices = [(i, t) for i, seg in enumerate(self.segments) if seg['len'] > pred_len for t in range(0, seg['len'] - pred_len)]
        self.stats = stats 

    def _get_raw_state(self, seg, t):
        u, v, r = seg['Vel'][0, t], seg['Vel'][1, t], seg['pqr'][0, t]
        return np.array([u, v, r, u*abs(u), v*abs(v), r*abs(r), u*r, v*r], dtype=np.float32)

    def __getitem__(self, index):
        seg, t = self.segments[self.indices[index][0]], self.indices[index][1]
        x_t = (self._get_raw_state(seg, t) - self.stats["state_mean"]) / self.stats["state_std"]
        x_seq = (np.array([self._get_raw_state(seg, t + i + 1) for i in range(self.pred_len)]) - self.stats["state_mean"]) / self.stats["state_std"]
        u_seq = (np.array([seg['Thrusters_CMD'][:, t+i] for i in range(self.pred_len)]) - self.stats["ctrl_mean"]) / self.stats["ctrl_std"]
        return torch.FloatTensor(x_t), torch.FloatTensor(x_seq), torch.FloatTensor(u_seq)

    def __len__(self): return len(self.indices)

def evaluate_all_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path, npz_path, pred_len = "../checkpoints/koopman_best.pth", "../koopman_test.npz", 20
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = HorizontalKoopmanModel(input_dim=8, state_dim=3, control_dim=4, latent_dim=32, enc_hidden=[128, 128])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval() 
    
    stats = checkpoint['stats']
    state_mean_3d, state_std_3d = stats['state_mean'][:3], stats['state_std'][:3]
    
    test_ds = ExplicitKoopmanDataset(npz_path, pred_len=pred_len, stats=stats)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
    
    all_targets, all_preds = [], []
    
    with torch.no_grad():
        for x_t_8d, x_target_seq_8d, u_seq_norm in test_loader:
            x_t_8d, x_target_seq_8d, u_seq_norm = x_t_8d.to(device), x_target_seq_8d.to(device), u_seq_norm.to(device)
            
            z_current = model.encode(x_t_8d)
            pred_states_3d = []
            for step in range(pred_len):
                z_current = model.latent_step(z_current, u_seq_norm[:, step, :])
                pred_states_3d.append(model.reconstruct_state(z_current))
                
            x_pred_seq_3d = torch.stack(pred_states_3d, dim=1) 
            
            # 【作图逻辑修正】：把初始状态 x_t (Step 0) 也拼接到序列开头，展示绝对真实的出发点
            targs_with_init = torch.cat([x_t_8d[:, None, :3], x_target_seq_8d[:, :, :3]], dim=1)
            preds_with_init = torch.cat([x_t_8d[:, None, :3], x_pred_seq_3d], dim=1)
            
            all_targets.append(targs_with_init.cpu().numpy())
            all_preds.append(preds_with_init.cpu().numpy())

    full_targets = np.concatenate(all_targets, axis=0) # [Samples, 21, 3]
    full_preds = np.concatenate(all_preds, axis=0)     # [Samples, 21, 3]
    
    # 反归一化
    gt_u = full_targets[:, :, 0] * state_std_3d[0] + state_mean_3d[0]
    gt_v = full_targets[:, :, 1] * state_std_3d[1] + state_mean_3d[1]
    pred_u = full_preds[:, :, 0] * state_std_3d[0] + state_mean_3d[0]
    pred_v = full_preds[:, :, 1] * state_std_3d[1] + state_mean_3d[1]

    vel_error = np.sqrt((gt_u - pred_u)**2 + (gt_v - pred_v)**2)
    mean_vel_error_per_step = np.mean(vel_error, axis=0) # 长度为 21 (包含了 step 0)

    # =============== 绘图 ===============
    os.makedirs("test_analysis", exist_ok=True)
    sample_indices = np.linspace(0, full_targets.shape[0] - 1, 6, dtype=int)
    steps = np.arange(0, pred_len + 1) # X轴现在从 0 到 20
    
    def plot_metric(gt, pred, title, ylabel, filename, is_deg=False):
        plt.figure(figsize=(18, 12))
        for i, idx in enumerate(sample_indices):
            plt.subplot(2, 3, i+1)
            g, p = (np.degrees(gt[idx]), np.degrees(pred[idx])) if is_deg else (gt[idx], pred[idx])
            plt.plot(steps, g, 'g-o', label='GT', alpha=0.6)
            plt.plot(steps, p, 'r--x', label='Pred')
            plt.title(f"Sample {idx} - {title}")
            plt.xlabel("Prediction Step (0 is Initial State)")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            if i == 0: plt.legend()
        plt.tight_layout()
        plt.savefig(f"test_analysis/{filename}")

    plot_metric(gt_u, pred_u, "Surge Velocity (u)", "Velocity [m/s]", "u_velocity_comparison.png")
    plot_metric(gt_v, pred_v, "Sway Velocity (v)", "Velocity [m/s]", "v_velocity_comparison.png")
    
    gt_r = full_targets[:, :, 2] * state_std_3d[2] + state_mean_3d[2]
    pred_r = full_preds[:, :, 2] * state_std_3d[2] + state_mean_3d[2]
    plot_metric(gt_r, pred_r, "Yaw Rate (r)", "Yaw Rate [deg/s]", "r_yawrate_comparison.png", True)

    plt.figure(figsize=(10, 6))
    # 注意切片，因为 step 0 误差必定是 0，我们从 step 1 开始画误差累积图即可
    plt.plot(steps[1:], mean_vel_error_per_step[1:], marker='s', color='blue', linewidth=2, label='Mean Vel L2 Error')
    plt.fill_between(steps[1:], 0, mean_vel_error_per_step[1:], color='blue', alpha=0.1)
    plt.title("Velocity Error Accumulation in Blind Test (Multi-step)")
    plt.xlabel("Prediction Step")
    plt.ylabel("Average Speed Error [m/s]")
    plt.xticks(steps[1:])
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig("test_analysis/velocity_error_curve.png", dpi=200)

if __name__ == "__main__":
    evaluate_all_and_plot()