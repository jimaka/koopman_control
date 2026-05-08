import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

from koopman import HorizontalKoopmanModel
from train_multistep_intra import IntraSegmentKoopmanDataset

def evaluate_all_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/koopman_best.pth"
    npz_path = "test_ds/koopman_test_dataset.npz"
    pred_len = 20
    batch_size = 2560 
    
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到模型文件 {ckpt_path}")
        return
    
    print(f"✅ 设备: {device}")
    # 注意这里设置了 weights_only=False 以兼容保存的归一化字典
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 1. 模型加载 (结构需与训练时保持一致)
    model = HorizontalKoopmanModel(state_dim=6, control_dim=4, latent_dim=32)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() 
    
    # 2. 获取归一化参数
    stats = checkpoint['stats']
    state_mean = stats['state_mean']
    state_std = stats['state_std']
    
    print("正在加载测试集...")
    test_ds = IntraSegmentKoopmanDataset(npz_path, split_mode='test', pred_len=pred_len, ratios=(0.0, 0.0, 1.0), stats=stats)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    all_targets_list = []
    all_preds_list = []
    
    # 3. 多步预测推理
    print(f"🚀 开始在 {len(test_ds)} 个样本上进行全局多步验证...")
    with torch.no_grad():
        for batch_idx, (x_t_norm, x_target_seq_norm, u_seq_norm) in enumerate(test_loader):
            x_t_norm = x_t_norm.to(device)
            x_target_seq_norm = x_target_seq_norm.to(device)
            u_seq_norm = u_seq_norm.to(device)
            
            z_current = model.encode(x_t_norm)
            pred_states_norm = []
            for step in range(pred_len):
                u_t_step = u_seq_norm[:, step, :]
                
                # 这里调用的是 koopman.py 里的 latent_step，内部已经实现了 z + delta_z 的残差逻辑
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
    # 5. 速度误差计算 (核心逻辑 - 已修复物理索引)
    # 标准船舶运动学索引: 0:x, 1:y, 2:yaw, 3:u(纵向速度), 4:v(横向速度), 5:r(角速度)
    # ==========================================
    print("正在分析速度误差...")
    
    # 反归一化真实的纵向速度(u)和横向速度(v)
    gt_u = full_targets[:, :, 3] * state_std[3] + state_mean[3]
    gt_v = full_targets[:, :, 4] * state_std[4] + state_mean[4]
    
    # 反归一化预测的纵向速度(u)和横向速度(v)
    pred_u = full_preds[:, :, 3] * state_std[3] + state_mean[3]
    pred_v = full_preds[:, :, 4] * state_std[4] + state_mean[4]

    # 计算每个步长的速度矢量模长误差: sqrt( (u_gt - u_pred)^2 + (v_gt - v_pred)^2 )
    vel_error = np.sqrt((gt_u - pred_u)**2 + (gt_v - pred_v)**2)
    
    # 计算所有样本在每一步的平均误差
    mean_vel_error_per_step = np.mean(vel_error, axis=0)

    # ==========================================
    # 6. 作图
    # ==========================================
    plot_dir = "test_analysis"
    os.makedirs(plot_dir, exist_ok=True)
    
    # --- 子图 1: 轨迹对比 ---
    plt.figure(figsize=(18, 12))
    sample_indices = np.linspace(0, total_samples - 1, 6, dtype=int)
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i+1)
        # 反归一化位置 (索引 0 为 x，索引 1 为 y)
        gx = full_targets[idx, :, 0] * state_std[0] + state_mean[0]
        gy = full_targets[idx, :, 1] * state_std[1] + state_mean[1]
        px = full_preds[idx, :, 0] * state_std[0] + state_mean[0]
        py = full_preds[idx, :, 1] * state_std[1] + state_mean[1]
        
        plt.plot(gx, gy, 'g-o', label='GT Path', alpha=0.6)
        plt.plot(px, py, 'r--x', label='Koopman Pred')
        plt.title(f"Sample {idx}")
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "trajectory_comparison.png"))

    # --- 子图 2: 速度误差随步数变化 ---
    plt.figure(figsize=(10, 6))
    steps = np.arange(1, pred_len + 1)
    plt.plot(steps, mean_vel_error_per_step, marker='s', color='blue', linewidth=2, label='Mean Velocity L2 Error (u, v)')
    plt.fill_between(steps, 0, mean_vel_error_per_step, color='blue', alpha=0.1)
    
    plt.title("Velocity Error Accumulation (Multi-step Prediction)", fontsize=14)
    plt.xlabel("Prediction Step", fontsize=12)
    plt.ylabel("Average Speed Error [m/s]", fontsize=12)
    plt.xticks(steps)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(os.path.join(plot_dir, "velocity_error_curve.png"), dpi=200)
    
    print(f"✅ 可视化图表已保存至 {plot_dir} 目录。")
    print(f"📊 平均速度误差 (全程): {np.mean(vel_error):.4f} m/s")

if __name__ == "__main__":
    evaluate_all_and_plot()