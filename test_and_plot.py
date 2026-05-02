import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 导入你的模型和 Dataset
from koopman import HorizontalKoopmanModel
# 注意：这里需要导入你上一个脚本里写的 Dataset 和 collate_fn
from train_multistep_intra import IntraSegmentKoopmanDataset, collate_fn_local_frame

def evaluate_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/koopman_best.pth"
    npz_path = "koopman_dataset_v1.npz"
    pred_len = 10
    
    print(f"正在加载最优模型: {ckpt_path}")
    # ================= 关键修改点 =================
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    # ==============================================
    
    # 1. 恢复模型
    model = HorizontalKoopmanModel(state_dim=6, control_dim=4, latent_dim=16)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 2. 恢复归一化参数
    stats = checkpoint['stats']
    
    # 后面的代码保持不变...
    
    # 3. 加载【测试集】(Test Split: 最后的 15% 数据)
    print("正在加载测试集数据...")
    test_ds = IntraSegmentKoopmanDataset(npz_path, split_mode='test', pred_len=pred_len, stats=stats)
    # batch_size 不要太大，方便我们抽取画图
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=True, collate_fn=collate_fn_local_frame)
    
    # 4. 抽取一个 Batch 进行多步推演
    x_t, x_target_seq, u_seq = next(iter(test_loader))
    x_t = x_t.to(device)
    u_seq = u_seq.to(device)
    
    with torch.no_grad():
        z_current = model.encode(x_t)
        pred_states = []
        for step in range(pred_len):
            z_next = model.latent_step(z_current, u_seq[:, step, :])
            pred_states.append(model.reconstruct_state(z_next))
            z_current = z_next
        x_pred_seq = torch.stack(pred_states, dim=1) # (B, pred_len, 6)
        
    # 转回 CPU numpy 以便绘图
    x_target_np = x_target_seq.cpu().numpy()
    x_pred_np = x_pred_seq.cpu().numpy()
    
    # 获取反归一化的标准差 (用于把坐标转换回真实的 '米')
    std_x = stats['state_std'][0]
    std_y = stats['state_std'][1]
    
    # 5. 可视化对比 (画出 6 个样本)
    plot_dir = "test_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    num_plots = min(6, x_target_np.shape[0])
    
    for i in range(num_plots):
        plt.subplot(2, 3, i+1)
        
        # 反归一化，恢复真实物理尺度 (米)
        # 注意：因为是局部坐标系(起点为0,0)，所以只乘标准差即可，不需要加均值
        gt_x = x_target_np[i, :, 0] * std_x
        gt_y = x_target_np[i, :, 1] * std_y
        
        pred_x = x_pred_np[i, :, 0] * std_x
        pred_y = x_pred_np[i, :, 1] * std_y
        
        # 绘制 Ground Truth (绿色)
        plt.plot(gt_x, gt_y, 'g-o', linewidth=2, markersize=4, label='Ground Truth')
        # 绘制 Prediction (红色虚线)
        plt.plot(pred_x, pred_y, 'r--x', linewidth=2, markersize=5, label='Koopman Predict')
        
        # 标记起点 (0,0)
        plt.scatter(0, 0, c='black', marker='*', s=150, zorder=5, label='Start (0,0)')
        
        plt.title(f"Test Sample {i+1} (Local Frame)")
        plt.xlabel("X [meters]")
        plt.ylabel("Y [meters]")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal') # 保证 X Y 比例真实
        if i == 0:
            plt.legend()
            
    plt.tight_layout()
    save_path = os.path.join(plot_dir, "test_trajectory_comparison.png")
    plt.savefig(save_path, dpi=200)
    print(f"\n✅ 测试集推演轨迹对比图已保存至: {save_path}")
    print("请打开图片，观察红线(预测)是否与绿线(真实)紧密贴合！")

if __name__ == "__main__":
    evaluate_and_plot()