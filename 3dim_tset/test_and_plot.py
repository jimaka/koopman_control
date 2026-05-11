import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from koopman import HorizontalKoopmanModel

class ExplicitKoopmanDataset(Dataset):
    def __init__(self, npz_path, pred_len=20, stats=None, is_train=False):
        super().__init__()
        self.segments = np.load(npz_path, allow_pickle=True)['datas']
        self.pred_len = pred_len
        self.indices = []
        
        for local_seg_idx, seg in enumerate(self.segments):
            if seg['len'] > pred_len:
                for t in range(0, seg['len'] - pred_len):
                    self.indices.append((local_seg_idx, t))
                    
        self.stats = stats 
        mode_str = "TRAIN" if is_train else "TEST"
        print(f"Dataset [{mode_str}] 加载完毕: {npz_path} | 包含 {len(self.segments)} 个动态段 | 共 {len(self.indices)} 个推演样本.")

    def _get_raw_state(self, seg, t):
        u = seg['Vel'][0, t]
        v = seg['Vel'][1, t]
        r = seg['pqr'][0, t]
        
        u_abs_u = u * abs(u)
        v_abs_v = v * abs(v)
        r_abs_r = r * abs(r)
        u_r = u * r
        v_r = v * r
        
        return np.array([u, v, r, u_abs_u, v_abs_v, r_abs_r, u_r, v_r], dtype=np.float32)

    def __getitem__(self, index):
        seg = self.segments[self.indices[index][0]]
        t = self.indices[index][1]
        
        x_t = self._get_raw_state(seg, t)
        x_seq = np.array([self._get_raw_state(seg, t + i) for i in range(1, self.pred_len + 1)])
        
        x_t_norm = (x_t - self.stats["state_mean"]) / self.stats["state_std"]
        x_seq_norm = (x_seq - self.stats["state_mean"]) / self.stats["state_std"]
        u_seq_norm = np.array([(seg['Thrusters_CMD'][:, t+i] - self.stats["ctrl_mean"])/self.stats["ctrl_std"] for i in range(self.pred_len)])
        
        return torch.FloatTensor(x_t_norm), torch.FloatTensor(x_seq_norm), torch.FloatTensor(u_seq_norm)

    def __len__(self): 
        return len(self.indices)


def evaluate_all_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "../checkpoints/koopman_best.pth"
    npz_path = "../koopman_test.npz"
    pred_len = 20
    batch_size = 2048 
    
    if not os.path.exists(ckpt_path) or not os.path.exists(npz_path):
        print(f"错误: 找不到模型或数据文件")
        return
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 【必须同步修改】：latent_dim=32
    model = HorizontalKoopmanModel(
        input_dim=8, 
        state_dim=3, 
        control_dim=4, 
        latent_dim=32, 
        enc_hidden=[128, 128], 
        dec_hidden=[128, 128],
        use_skip=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() 
    
    stats = checkpoint['stats']
    state_mean_3d = stats['state_mean'][:3]
    state_std_3d = stats['state_std'][:3]
    
    test_ds = ExplicitKoopmanDataset(npz_path, pred_len=pred_len, stats=stats, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    all_targets_list = []
    all_preds_list = []
    
    with torch.no_grad():
        for batch_idx, (x_t_8d, x_target_seq_8d, u_seq_norm) in enumerate(test_loader):
            x_t_8d = x_t_8d.to(device)
            x_target_seq_8d = x_target_seq_8d.to(device)
            u_seq_norm = u_seq_norm.to(device)
            
            z_current = model.encode(x_t_8d)
            pred_states_3d = []
            for step in range(pred_len):
                z_next = model.latent_step(z_current, u_seq_norm[:, step, :])
                x_hat_step = model.reconstruct_state(z_next) 
                pred_states_3d.append(x_hat_step)
                z_current = z_next
                
            x_pred_seq_3d = torch.stack(pred_states_3d, dim=1) 
            all_targets_list.append(x_target_seq_8d[:, :, :3].cpu().numpy())
            all_preds_list.append(x_pred_seq_3d.cpu().numpy())

    full_targets = np.concatenate(all_targets_list, axis=0) 
    full_preds = np.concatenate(all_preds_list, axis=0)     
    total_samples = full_targets.shape[0]

    print("分析盲考测试集上的速度物理误差...")
    
    gt_u = full_targets[:, :, 0] * state_std_3d[0] + state_mean_3d[0]
    gt_v = full_targets[:, :, 1] * state_std_3d[1] + state_mean_3d[1]
    
    pred_u = full_preds[:, :, 0] * state_std_3d[0] + state_mean_3d[0]
    pred_v = full_preds[:, :, 1] * state_std_3d[1] + state_mean_3d[1]

    vel_error = np.sqrt((gt_u - pred_u)**2 + (gt_v - pred_v)**2)
    mean_vel_error_per_step = np.mean(vel_error, axis=0)

    # =============== 绘图 ===============
    plot_dir = "test_analysis"
    os.makedirs(plot_dir, exist_ok=True)
    sample_indices = np.linspace(0, total_samples - 1, 6, dtype=int)
    
    plt.figure(figsize=(18, 12))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i+1)
        g_u = full_targets[idx, :, 0] * state_std_3d[0] + state_mean_3d[0]
        p_u = full_preds[idx, :, 0] * state_std_3d[0] + state_mean_3d[0]
        plt.plot(g_u, 'g-o', label='GT Surge (u)', alpha=0.6)
        plt.plot(p_u, 'r--x', label='Koopman Pred')
        plt.title(f"Sample {idx} - Surge Velocity (u)")
        plt.xlabel("Prediction Step")
        plt.ylabel("Velocity [m/s]")
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "u_velocity_comparison.png"))

    plt.figure(figsize=(18, 12))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i+1)
        g_v = full_targets[idx, :, 1] * state_std_3d[1] + state_mean_3d[1]
        p_v = full_preds[idx, :, 1] * state_std_3d[1] + state_mean_3d[1]
        plt.plot(g_v, 'g-o', label='GT Sway (v)', alpha=0.6)
        plt.plot(p_v, 'r--x', label='Koopman Pred')
        plt.title(f"Sample {idx} - Sway Velocity (v)")
        plt.xlabel("Prediction Step")
        plt.ylabel("Velocity [m/s]")
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "v_velocity_comparison.png"))

    plt.figure(figsize=(18, 12))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i+1)
        g_r = full_targets[idx, :, 2] * state_std_3d[2] + state_mean_3d[2]
        p_r = full_preds[idx, :, 2] * state_std_3d[2] + state_mean_3d[2]
        plt.plot(np.degrees(g_r), 'g-o', label='GT Yaw Rate (r)', alpha=0.6)
        plt.plot(np.degrees(p_r), 'r--x', label='Koopman Pred')
        plt.title(f"Sample {idx} - Yaw Rate (r)")
        plt.xlabel("Prediction Step")
        plt.ylabel("Yaw Rate [deg/s]")
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "r_yawrate_comparison.png"))

    plt.figure(figsize=(10, 6))
    steps = np.arange(1, pred_len + 1)
    plt.plot(steps, mean_vel_error_per_step, marker='s', color='blue', linewidth=2, label='Mean Velocity L2 Error (u, v)')
    plt.fill_between(steps, 0, mean_vel_error_per_step, color='blue', alpha=0.1)
    plt.title("Velocity Error Accumulation in Blind Test (Multi-step)", fontsize=14)
    plt.xlabel("Prediction Step", fontsize=12)
    plt.ylabel("Average Speed Error [m/s]", fontsize=12)
    plt.xticks(steps)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "velocity_error_curve.png"), dpi=200)
    
    print(f"✅ 可视化图表已保存至 {plot_dir} 目录。包含纵向速度(u)、横向速度(v)、角速度(r)对比及误差累积。")
    print(f"📊 盲考测试集平均合速度(u,v)误差 (全程): {np.mean(vel_error):.4f} m/s")

if __name__ == "__main__":
    evaluate_all_and_plot()