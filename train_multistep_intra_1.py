import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 从你现有的 koopman.py 中导入模型
from koopman import HorizontalKoopmanModel

# ==========================================
# 1. 支持段内切分的自定义 Dataset
# ==========================================
class IntraSegmentKoopmanDataset(Dataset):
    def __init__(self, npz_path, split_mode='train', pred_len=10, ratios=(0.7, 0.15, 0.15), stats=None):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)['datas']
        self.pred_len = pred_len
        self.mode = split_mode
        self.segments_sliced = []
        self.indices = []
        
        train_r, val_r, test_r = ratios
        
        # 1. 动态按比例在每个物理段内部切分时间索引
        for seg_idx, seg in enumerate(data):
            L = seg['len']
            n_train = int(L * train_r)
            n_val = int(L * val_r)
            
            if split_mode == 'train':
                s_idx, e_idx = 0, n_train
            elif split_mode == 'val':
                s_idx, e_idx = n_train, n_train + n_val
            else: # test
                s_idx, e_idx = n_train + n_val, L
                
            if e_idx - s_idx > pred_len:
                self.segments_sliced.append((seg, s_idx, e_idx))
        
        # 2. 构建扁平化的索引列表
        for local_seg_idx, (seg, s_idx, e_idx) in enumerate(self.segments_sliced):
            for t in range(s_idx, e_idx - pred_len):
                self.indices.append((local_seg_idx, t))
                
        # 3. 处理归一化统计量 (现在是基于"局部坐标系"的统计量)
        if stats is not None:
            self.stats = stats
        else:
            assert split_mode == 'train', "只有训练集可以计算统计量"
            self.stats = self._compute_local_statistics()
            
        print(f"Dataset [{split_mode.upper()}]: 共提取了 {len(self.indices)} 个有效推演样本.")
        
    def save_to_npz(self, save_path):
        """
        将当前数据集模式下切分好的物理段数据，独立保存为一个全新的 npz 文件
        """
        saved_datas = []
        for seg, s_idx, e_idx in self.segments_sliced:
            new_seg = {'len': e_idx - s_idx}
            original_len = seg['len']
            
            for k, v in seg.items():
                if k == 'len':
                    continue
                # 假设多维数组中，时间维度在最后一维 (如 shape: [dim, time] 或 [time])
                if isinstance(v, np.ndarray) and v.shape[-1] == original_len:
                    new_seg[k] = v[..., s_idx:e_idx]
                else:
                    new_seg[k] = v # 对于非时间序列的数据（如果有），直接复制
                    
            saved_datas.append(new_seg)
            
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, datas=saved_datas)
        print(f">>> [切分输出] {self.mode.upper()} 数据集已独立保存至 {save_path} (共 {len(saved_datas)} 个物理段)")

    def _get_raw_state(self, seg, t):
        """仅提取原始物理量，不在这里做标准化"""
        return np.array([
            seg['Pos'][0, t], seg['Pos'][1, t], seg['Euler'][2, t],
            seg['Vel'][0, t], seg['Vel'][1, t], seg['pqr'][0, t]
        ], dtype=np.float32)

    def _transform_to_local(self, x_t_raw, x_seq_raw):
        """在真实的物理尺度下进行局部坐标系映射"""
        yaw_t = x_t_raw[2]
        cos_yaw = np.cos(yaw_t)
        sin_yaw = np.sin(yaw_t)
        
        x_seq_local = np.zeros_like(x_seq_raw)
        for i in range(len(x_seq_raw)):
            dx = x_seq_raw[i, 0] - x_t_raw[0]
            dy = x_seq_raw[i, 1] - x_t_raw[1]
            
            # 旋转
            local_x = dx * cos_yaw + dy * sin_yaw
            local_y = -dx * sin_yaw + dy * cos_yaw
            
            # 相对偏航角，并归一化到 [-pi, pi]
            local_yaw = x_seq_raw[i, 2] - yaw_t
            local_yaw = (local_yaw + np.pi) % (2 * np.pi) - np.pi
            
            x_seq_local[i] = [local_x, local_y, local_yaw, x_seq_raw[i, 3], x_seq_raw[i, 4], x_seq_raw[i, 5]]
            
        # 当前时刻在局部坐标系下，位置和角度必然是0
        x_t_local = np.array([0.0, 0.0, 0.0, x_t_raw[3], x_t_raw[4], x_t_raw[5]], dtype=np.float32)
        return x_t_local, x_seq_local

    def _compute_local_statistics(self):
        """将所有数据转为局部坐标系后，再求 Mean, Std, Max, Min"""
        print("正在计算局部坐标系下的 Mean, Std, Min, Max (约需几秒钟)...")
        local_states = []
        controls = []
        for local_seg_idx, t in self.indices:
            seg = self.segments_sliced[local_seg_idx][0]
            
            x_t_raw = self._get_raw_state(seg, t)
            x_seq_raw = np.array([self._get_raw_state(seg, t + i) for i in range(1, self.pred_len + 1)])
            
            x_t_local, x_seq_local = self._transform_to_local(x_t_raw, x_seq_raw)
            local_states.append(x_t_local)
            local_states.extend(x_seq_local)
            
            for i in range(self.pred_len):
                controls.append(seg['Thrusters_CMD'][:, t + i])
                
        local_states = np.array(local_states, dtype=np.float32)
        controls = np.array(controls, dtype=np.float32)
        
        return {
            "state_mean": np.mean(local_states, axis=0),
            "state_std": np.std(local_states, axis=0) + 1e-6,
            "state_min": np.min(local_states, axis=0),
            "state_max": np.max(local_states, axis=0),
            "ctrl_mean": np.mean(controls, axis=0),
            "ctrl_std": np.std(controls, axis=0) + 1e-6,
            "ctrl_min": np.min(controls, axis=0),
            "ctrl_max": np.max(controls, axis=0),
        }

    def __getitem__(self, index):
        local_seg_idx, t = self.indices[index]
        seg = self.segments_sliced[local_seg_idx][0]

        # 1. 提取真实物理数据
        x_t_raw = self._get_raw_state(seg, t)
        x_seq_raw = np.array([self._get_raw_state(seg, t + i) for i in range(1, self.pred_len + 1)])
        
        # 2. 进行物理意义上的局部映射
        x_t_local, x_seq_local = self._transform_to_local(x_t_raw, x_seq_raw)
        
        # 3. 进行标准化运算
        x_t_norm = (x_t_local - self.stats["state_mean"]) / self.stats["state_std"]
        x_seq_norm = (x_seq_local - self.stats["state_mean"]) / self.stats["state_std"]
        
        u_seq = []
        for i in range(self.pred_len):
            u = seg['Thrusters_CMD'][:, t + i].astype(np.float32)
            u = (u - self.stats["ctrl_mean"]) / self.stats["ctrl_std"]
            u_seq.append(u)
        u_seq_norm = np.array(u_seq)

        return torch.FloatTensor(x_t_norm), torch.FloatTensor(x_seq_norm), torch.FloatTensor(u_seq_norm)

    def __len__(self):
        return len(self.indices)

# ==========================================
# 2. 局部坐标系转换与 YAML 导出函数
# ==========================================
def batched_sequence_transform_to_local_frame(x_t: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
    dx = x_seq[:, :, 0] - x_t[:, 0].unsqueeze(1)
    dy = x_seq[:, :, 1] - x_t[:, 1].unsqueeze(1)
    yaw_t = x_t[:, 2].unsqueeze(1)
    cos_yaw = torch.cos(yaw_t)
    sin_yaw = torch.sin(yaw_t)
    
    # 局部坐标系旋转映射
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    
    local_yaw = x_seq[:, :, 2] - yaw_t
    local_yaw = (local_yaw + torch.pi) % (2 * torch.pi) - torch.pi
    
    return torch.stack([local_x, local_y, local_yaw, x_seq[:, :, 3], x_seq[:, :, 4], x_seq[:, :, 5]], dim=2)

def export_params_to_yaml(model, stats, save_path="checkpoints/koopman_deploy_params.yaml"):
    """
    将归一化参数、边界范围以及 Koopman A、B 矩阵导出为 YAML 格式
    """
    # 1. 鲁棒地提取 A 和 B 矩阵
    def extract_matrix(matrix_attr):
        if matrix_attr is None:
            return []
        # 如果是 nn.Linear 等网络层，提取其 .weight
        if isinstance(matrix_attr, nn.Module) and hasattr(matrix_attr, 'weight'):
            return matrix_attr.weight.detach().cpu().numpy().tolist()
        # 如果直接是 Parameter 或 Tensor
        elif hasattr(matrix_attr, 'detach'):
            return matrix_attr.detach().cpu().numpy().tolist()
        return []

    try:
        A_matrix = extract_matrix(getattr(model, 'A', None))
        B_matrix = extract_matrix(getattr(model, 'B', None))
        
        if not A_matrix:
            print("  [警告] 提取的 A 矩阵为空。请检查 koopman.py 中的变量命名。")
    except Exception as e:
        print(f"  [错误] 提取 A/B 矩阵时失败: {e}")
        A_matrix, B_matrix = [], []

    # 2. 计算方差 (Variance = Std^2)
    state_var = (stats["state_std"] ** 2).tolist()
    ctrl_var = (stats["ctrl_std"] ** 2).tolist()

    # 3. 构建字典
    yaml_data = {
        "normalization": {
            "state_mean": stats["state_mean"].tolist(),
            "state_variance": state_var,      
            "state_std": stats["state_std"].tolist(),
            "ctrl_mean": stats["ctrl_mean"].tolist(),
            "ctrl_variance": ctrl_var,        
            "ctrl_std": stats["ctrl_std"].tolist()
        },
        "bounds": {
            "state_min": stats["state_min"].tolist(),
            "state_max": stats["state_max"].tolist(),
            "ctrl_min": stats["ctrl_min"].tolist(),
            "ctrl_max": stats["ctrl_max"].tolist()
        },
        "system_matrices": {
            "A": A_matrix,
            "B": B_matrix
        }
    }

    # 4. 写入 YAML 文件
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False, indent=4)
    print(f">>> [参数导出] 均值、方差、极值及 A/B 矩阵已成功导出至: {save_path}")

# ==========================================
# 3. 多步训练主循环与模型保存
# ==========================================
def train():
    npz_path = "koopman_dataset_v1.npz"
    ckpt_dir = "checkpoints" # 模型保存路径
    os.makedirs(ckpt_dir, exist_ok=True)
    
    pred_len = 20
    batch_size = 2048
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n>>> [第一阶段] 正在加载并进行动态段内切分...")
    train_ds = IntraSegmentKoopmanDataset(npz_path, split_mode='train', pred_len=pred_len)
    val_ds = IntraSegmentKoopmanDataset(npz_path, split_mode='val', pred_len=pred_len, stats=train_ds.stats)
    
    # 提取并单独保存测试集为 npz 文件
    test_ds = IntraSegmentKoopmanDataset(npz_path, split_mode='test', pred_len=pred_len, stats=train_ds.stats)
    test_ds.save_to_npz("test_ds/koopman_test_dataset.npz")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = HorizontalKoopmanModel(state_dim=6, control_dim=4, latent_dim=32).to(device)
    
    # 使用初始较低学习率，配合 StepLR 解决振荡
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 【新增】：为 6 个状态量分配不同的权重。
    # 索引对应 -> 0:x, 1:y, 2:yaw, 3:u, 4:v, 5:r
    # 给 yaw(2倍), v(5倍), r(5倍) 施加更强的惩罚，逼迫模型学习弯道耦合动力学
    state_weights = torch.tensor([1.0, 1.0, 2.0, 1.0, 5.0, 5.0], device=device)
    
    # 隐空间的纯线性约束 Loss 仍用基础的 MSE
    mse_loss = nn.MSELoss()
    
    print(f"\n>>> [第二阶段] 开始多步训练 (使用设备: {device})")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss_train = 0.0
        
        # 记录具体分项Loss用于打印
        epoch_pred = 0.0
        epoch_recon = 0.0
        epoch_linear = 0.0
        
        for batch_idx, (x_t, x_target_seq, u_seq) in enumerate(train_loader):
            x_t = x_t.to(device)
            x_target_seq = x_target_seq.to(device)
            u_seq = u_seq.to(device)
            
            optimizer.zero_grad()
            
            # 1. 编码
            z_t = model.encode(x_t)
            x_t_recon = model.reconstruct_state(z_t)
            
            # 2. 多步隐空间推演
            z_current = z_t
            pred_states = []
            pred_latents = []
            
            for step in range(pred_len):
                u_t_step = u_seq[:, step, :]
                z_next = model.latent_step(z_current, u_t_step)
                x_hat_step = model.reconstruct_state(z_next)
                
                pred_latents.append(z_next)
                pred_states.append(x_hat_step)
                z_current = z_next
                
            x_pred_seq = torch.stack(pred_states, dim=1)
            
            # 3. 代价函数计算 (引入 state_weights 加权)
            # 预测 Loss 加权：均方误差乘以对应的特征权重
            loss_pred = torch.mean(state_weights * (x_pred_seq - x_target_seq)**2)
            
            # 重构 Loss 同样加权，保证特征重要性在编码/解码阶段一致
            loss_recon = torch.mean(state_weights * (x_t_recon - x_t)**2)
            
            # 隐空间是低维抽象特征，无法对应物理意义，保持原样
            target_z_final = model.encode(x_target_seq[:, -1, :])
            loss_linear = mse_loss(pred_latents[-1], target_z_final)
            
            rho = model.spectral_radius()
            loss_stab = torch.relu(rho - 1.0) ** 2
            
            loss = 10 * loss_pred + 1.0 * loss_recon + 10.0 * loss_linear + 0.1 * loss_stab
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防止爆炸
            optimizer.step()
            
            total_loss_train += loss.item()
            epoch_pred += loss_pred.item()
            epoch_recon += loss_recon.item()
            epoch_linear += loss_linear.item()
            
        num_batches = len(train_loader)
        avg_train_loss = total_loss_train / num_batches
        
        # --- 验证阶段 ---
        model.eval()
        total_loss_val = 0.0
        with torch.no_grad():
            for x_t, x_target_seq, u_seq in val_loader:
                x_t = x_t.to(device)
                x_target_seq = x_target_seq.to(device)
                u_seq = u_seq.to(device)
                
                z_t = model.encode(x_t)
                z_current = z_t
                pred_states = []
                for step in range(pred_len):
                    z_next = model.latent_step(z_current, u_seq[:, step, :])
                    pred_states.append(model.reconstruct_state(z_next))
                    z_current = z_next
                    
                x_pred_seq = torch.stack(pred_states, dim=1)
                
                # 验证集的 Loss 也换成加权计算，这样我们选出的 "最优模型" 才是弯道表现更好的那个
                val_loss_weighted = torch.mean(state_weights * (x_pred_seq - x_target_seq)**2)
                total_loss_val += val_loss_weighted.item()
                
        avg_val_loss = total_loss_val / len(val_loader)
        
        # StepLR 步进更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"LR: {current_lr:.6f} | "
              f"Train Total: {avg_train_loss:.4f} (Pred:{epoch_pred/num_batches:.4f}, Recon:{epoch_recon/num_batches:.4f}, Linear:{epoch_linear/num_batches:.4f}) | "
              f"Val Pred (Weighted): {avg_val_loss:.4f}")

        # --- 模型保存阶段 ---
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'stats': train_ds.stats 
        }

        latest_path = os.path.join(ckpt_dir, "koopman_latest.pth")
        torch.save(checkpoint, latest_path)

        # 验证集误差创新低时，保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(ckpt_dir, "koopman_best.pth")
            torch.save(checkpoint, best_path)
            print(f"  --> [*] 验证集 Loss 创新低 ({best_val_loss:.4f})，已保存最优模型至 {best_path}")

    print(f"\n>>> 训练全部完成！")
    print(f">>> 最优验证集 Loss 为: {best_val_loss:.4f}")
    print(f">>> 模型文件保存在目录: ./{ckpt_dir}/")

    # ===============================================
    # 训练结束后：加载最优模型并导出 YAML 参数
    # ===============================================
    print("\n>>> 开始导出部署所需的 YAML 参数文件...")
    # 加载表现最好的模型权重提取 A, B 矩阵
    best_path = os.path.join(ckpt_dir, "koopman_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    yaml_path = os.path.join(ckpt_dir, "koopman_deploy_params.yaml")
    export_params_to_yaml(model, train_ds.stats, save_path=yaml_path)

if __name__ == "__main__":
    train()