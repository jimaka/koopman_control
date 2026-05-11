import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from koopman import HorizontalKoopmanModel

# ==========================================
# 1. 显式读取的 Dataset (与新版训练代码保持一致)
# ==========================================
class ExplicitKoopmanDataset(Dataset):
    def __init__(self, npz_path, pred_len=20, stats=None, is_train=False):
        super().__init__()
        self.segments = np.load(npz_path, allow_pickle=True)['datas']
        self.pred_len = pred_len
        self.indices = []
        
        # 构建展平的时间索引
        for local_seg_idx, seg in enumerate(self.segments):
            if seg['len'] > pred_len:
                for t in range(0, seg['len'] - pred_len):
                    self.indices.append((local_seg_idx, t))
                    
        # 验证/测试集必须严格复用训练集的 stats
        self.stats = stats 
        if self.stats is None and is_train:
            self.stats = self._compute_local_statistics()
            
        mode_str = "TRAIN" if is_train else "TEST"
        print(f"Dataset [{mode_str}] 加载完毕: {npz_path} | 包含 {len(self.segments)} 个动态段 | 共 {len(self.indices)} 个推演样本.")

    def _get_raw_state(self, seg, t):
        return np.array([seg['Pos'][0, t], seg['Pos'][1, t], seg['Euler'][2, t],
                         seg['Vel'][0, t], seg['Vel'][1, t], seg['pqr'][0, t]], dtype=np.float32)

    def _transform_to_local(self, x_t_raw, x_seq_raw):
        yaw_t = x_t_raw[2]
        cos_yaw, sin_yaw = np.cos(yaw_t), np.sin(yaw_t)
        x_seq_local = np.zeros_like(x_seq_raw)
        
        for i in range(len(x_seq_raw)):
            dx, dy = x_seq_raw[i, 0] - x_t_raw[0], x_seq_raw[i, 1] - x_t_raw[1]
            x_seq_local[i] = [dx*cos_yaw + dy*sin_yaw, -dx*sin_yaw + dy*cos_yaw, 
                              (x_seq_raw[i, 2] - yaw_t + np.pi) % (2 * np.pi) - np.pi, 
                              x_seq_raw[i, 3], x_seq_raw[i, 4], x_seq_raw[i, 5]]
            
        x_t_local = np.array([0., 0., 0., x_t_raw[3], x_t_raw[4], x_t_raw[5]], dtype=np.float32)
        return x_t_local, x_seq_local

    def __getitem__(self, index):
        seg = self.segments[self.indices[index][0]]
        t = self.indices[index][1]
        
        x_t_raw = self._get_raw_state(seg, t)
        x_seq_raw = np.array([self._get_raw_state(seg, t + i) for i in range(1, self.pred_len + 1)])
        x_t_local, x_seq_local = self._transform_to_local(x_t_raw, x_seq_raw)
        
        x_t_norm = (x_t_local - self.stats["state_mean"]) / self.stats["state_std"]
        x_seq_norm = (x_seq_local - self.stats["state_mean"]) / self.stats["state_std"]
        u_seq_norm = np.array([(seg['Thrusters_CMD'][:, t+i] - self.stats["ctrl_mean"])/self.stats["ctrl_std"] for i in range(self.pred_len)])
        
        return torch.FloatTensor(x_t_norm), torch.FloatTensor(x_seq_norm), torch.FloatTensor(u_seq_norm)

    def __len__(self): 
        return len(self.indices)


# ==========================================
# 2. 评估与绘图主逻辑
# ==========================================
def evaluate_all_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/koopman_best.pth"
    npz_path = "koopman_test.npz" # 指向新的物理隔离测试集（盲考数据集）
    pred_len = 20
    batch_size = 2048 
    
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到模型文件 {ckpt_path}")
        return
    if not os.path.exists(npz_path):
        print(f"错误: 找不到测试集文件 {npz_path}，请确认分包脚本已运行。")
        return
        
    print(f"✅ 设备: {device}")
    
    # 注意这里设置了 weights_only=False 以兼容保存的归一化字典
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 1. 模型加载 (必须与新版训练代码中的 128 维模型保持绝对一致)
    model = HorizontalKoopmanModel(
        state_dim=6, 
        control_dim=4, 
        latent_dim=32, 
        enc_hidden=[64, 64], 
        dec_hidden=[64, 64],
        use_skip=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() 
    
    # 2. 获取训练集的全局归一化参数
    stats = checkpoint['stats']
    state_mean = stats['state_mean']
    state_std = stats['state_std']
    
    print("\n正在加载盲考测试集 (Random Sail)...")
    test_ds = ExplicitKoopmanDataset(npz_path, pred_len=pred_len, stats=stats, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    all_targets_list = []
    all_preds_list = []
    
    # 3. 多步预测推理
    print(f"🚀 开始在 {len(test_ds)} 个随机航行样本上进行全局多步验证...")
    with torch.no_grad():
        for batch_idx, (x_t_norm, x_target_seq_norm, u_seq_norm) in enumerate(test_loader):
            x_t_norm = x_t_norm.to(device)
            x_target_seq_norm = x_target_seq_norm.to(device)
            u_seq_norm = u_seq_norm.to(device)
            
            z_current = model.encode(x_t_norm)
            pred_states_norm = []
            for step in range(pred_len):
                u_t_step = u_seq_norm[:, step, :]
                
                # 调用 latent_step
                z_next = model.latent_step(z_current, u_t_step)
                x_hat_step = model.reconstruct_state(z_next)
                pred_states_norm.append(x_hat_step)
                z_current = z_next
                
            x_pred_seq_norm = torch.stack(pred_states_norm, dim=1) 
            
            all_targets_list.append(x_target_seq_norm.cpu().numpy())
            all_preds_list.append(x_pred_seq_norm.cpu().numpy())

    # 4. 数据拼接
    full_targets = np.concatenate(all_targets_list, axis=0) # [Samples, Steps, 6]
    full_preds = np.concatenate(all_preds_list, axis=0)     # [Samples, Steps, 6]
    total_samples = full_targets.shape[0]

    # ==========================================
    # 5. 速度误差计算 (纵向速度 u + 横向速度 v)
    # ==========================================
    print("正在分析盲考测试集上的速度物理误差...")
    
    gt_u = full_targets[:, :, 3] * state_std[3] + state_mean[3]
    gt_v = full_targets[:, :, 4] * state_std[4] + state_mean[4]
    
    pred_u = full_preds[:, :, 3] * state_std[3] + state_mean[3]
    pred_v = full_preds[:, :, 4] * state_std[4] + state_mean[4]

    vel_error = np.sqrt((gt_u - pred_u)**2 + (gt_v - pred_v)**2)
    mean_vel_error_per_step = np.mean(vel_error, axis=0)

    # ==========================================
    # 6. 作图 (新增了航向角、横纵速度随时间步的对比图)
    # ==========================================
    plot_dir = "test_analysis"
    os.makedirs(plot_dir, exist_ok=True)
    
    sample_indices = np.linspace(0, total_samples - 1, 6, dtype=int)
    
    # --- 子图 1: 轨迹对比 ---
    plt.figure(figsize=(18, 12))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i+1)
        gx = full_targets[idx, :, 0] * state_std[0] + state_mean[0]
        gy = full_targets[idx, :, 1] * state_std[1] + state_mean[1]
        px = full_preds[idx, :, 0] * state_std[0] + state_mean[0]
        py = full_preds[idx, :, 1] * state_std[1] + state_mean[1]
        
        plt.plot(gx, gy, 'g-o', label='GT Path', alpha=0.6)
        plt.plot(px, py, 'r--x', label='Koopman Pred')
        plt.title(f"Sample {idx} - Trajectory")
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "trajectory_comparison.png"))

    # --- 子图 2: 航向角 (Yaw) 对比 (状态索引 2) ---
    plt.figure(figsize=(18, 12))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i+1)
        g_yaw = full_targets[idx, :, 2] * state_std[2] + state_mean[2]
        p_yaw = full_preds[idx, :, 2] * state_std[2] + state_mean[2]
        
        # 转换为角度展示更直观
        plt.plot(np.degrees(g_yaw), 'g-o', label='GT Yaw', alpha=0.6)
        plt.plot(np.degrees(p_yaw), 'r--x', label='Koopman Pred')
        plt.title(f"Sample {idx} - Yaw Angle")
        plt.xlabel("Prediction Step")
        plt.ylabel("Local Yaw Angle [deg]")
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "yaw_comparison.png"))

    # --- 子图 3: 纵向速度 (u) 对比 (状态索引 3) ---
    plt.figure(figsize=(18, 12))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i+1)
        g_u = full_targets[idx, :, 3] * state_std[3] + state_mean[3]
        p_u = full_preds[idx, :, 3] * state_std[3] + state_mean[3]
        
        plt.plot(g_u, 'g-o', label='GT Surge (u)', alpha=0.6)
        plt.plot(p_u, 'r--x', label='Koopman Pred')
        plt.title(f"Sample {idx} - Surge Velocity (u)")
        plt.xlabel("Prediction Step")
        plt.ylabel("Velocity [m/s]")
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "u_velocity_comparison.png"))

    # --- 子图 4: 横向速度 (v) 对比 (状态索引 4) ---
    plt.figure(figsize=(18, 12))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i+1)
        g_v = full_targets[idx, :, 4] * state_std[4] + state_mean[4]
        p_v = full_preds[idx, :, 4] * state_std[4] + state_mean[4]
        
        plt.plot(g_v, 'g-o', label='GT Sway (v)', alpha=0.6)
        plt.plot(p_v, 'r--x', label='Koopman Pred')
        plt.title(f"Sample {idx} - Sway Velocity (v)")
        plt.xlabel("Prediction Step")
        plt.ylabel("Velocity [m/s]")
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "v_velocity_comparison.png"))

    # --- 子图 5: 速度总误差随步数变化 ---
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
    
    print(f"✅ 可视化图表已保存至 {plot_dir} 目录。包含轨迹、偏航角、横/纵速度对比。")
    print(f"📊 盲考测试集平均速度误差 (全程): {np.mean(vel_error):.4f} m/s")

if __name__ == "__main__":
    evaluate_all_and_plot()