# import os
# import yaml
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# # 为了支持实船的 OSQP MPC，必须使用纯线性的 HorizontalKoopmanModel
# # 依靠 128 维的隐空间来拟合高密度数据包中的非线性特征
# from koopman import HorizontalKoopmanModel

# # ==========================================
# # 1. 显式读取的高效 Dataset (完美对接分包脚本)
# # ==========================================
# class ExplicitKoopmanDataset(Dataset):
#     def __init__(self, npz_path, pred_len=20, stats=None, is_train=False):
#         super().__init__()
#         # 直接读取分包脚本生成的高密度数据集
#         self.segments = np.load(npz_path, allow_pickle=True)['datas']
#         self.pred_len = pred_len
#         self.indices = []
        
#         # 构建展平的时间索引
#         for local_seg_idx, seg in enumerate(self.segments):
#             if seg['len'] > pred_len:
#                 for t in range(0, seg['len'] - pred_len):
#                     self.indices.append((local_seg_idx, t))
                    
#         # 统计量处理：验证/测试集必须严格复用训练集的 stats，严禁重新计算！
#         self.stats = stats if stats is not None else self._compute_local_statistics()
            
#         mode_str = "TRAIN" if is_train else "EVAL"
#         print(f"Dataset [{mode_str}] 加载完毕: {npz_path} | 包含 {len(self.segments)} 个动态段 | 共 {len(self.indices)} 个推演样本.")

#     def _get_raw_state(self, seg, t):
#         # [x, y, yaw, u, v, r]
#         return np.array([seg['Pos'][0, t], seg['Pos'][1, t], seg['Euler'][2, t],
#                          seg['Vel'][0, t], seg['Vel'][1, t], seg['pqr'][0, t]], dtype=np.float32)

#     def _transform_to_local(self, x_t_raw, x_seq_raw):
#         """将未来轨迹映射到 t 时刻的局部坐标系下"""
#         yaw_t = x_t_raw[2]
#         cos_yaw, sin_yaw = np.cos(yaw_t), np.sin(yaw_t)
#         x_seq_local = np.zeros_like(x_seq_raw)
        
#         for i in range(len(x_seq_raw)):
#             dx, dy = x_seq_raw[i, 0] - x_t_raw[0], x_seq_raw[i, 1] - x_t_raw[1]
#             x_seq_local[i] = [dx*cos_yaw + dy*sin_yaw, -dx*sin_yaw + dy*cos_yaw, 
#                               (x_seq_raw[i, 2] - yaw_t + np.pi) % (2 * np.pi) - np.pi, 
#                               x_seq_raw[i, 3], x_seq_raw[i, 4], x_seq_raw[i, 5]]
            
#         x_t_local = np.array([0., 0., 0., x_t_raw[3], x_t_raw[4], x_t_raw[5]], dtype=np.float32)
#         return x_t_local, x_seq_local

#     def _compute_local_statistics(self):
#         print(">>> 正在基于高密度训练集计算全局 Mean & Std...")
#         local_states, controls = [], []
#         for local_seg_idx, t in self.indices:
#             seg = self.segments[local_seg_idx]
#             x_t_raw = self._get_raw_state(seg, t)
#             x_seq_raw = np.array([self._get_raw_state(seg, t + i) for i in range(1, self.pred_len + 1)])
#             x_t_local, x_seq_local = self._transform_to_local(x_t_raw, x_seq_raw)
            
#             local_states.append(x_t_local)
#             local_states.extend(x_seq_local)
#             for i in range(self.pred_len): 
#                 controls.append(seg['Thrusters_CMD'][:, t + i])
                
#         local_states, controls = np.array(local_states, dtype=np.float32), np.array(controls, dtype=np.float32)
#         return {
#             "state_mean": np.mean(local_states, axis=0), "state_std": np.std(local_states, axis=0) + 1e-6,
#             "state_min": np.min(local_states, axis=0), "state_max": np.max(local_states, axis=0),
#             "ctrl_mean": np.mean(controls, axis=0), "ctrl_std": np.std(controls, axis=0) + 1e-6,
#             "ctrl_min": np.min(controls, axis=0), "ctrl_max": np.max(controls, axis=0)
#         }

#     def __getitem__(self, index):
#         seg = self.segments[self.indices[index][0]]
#         t = self.indices[index][1]
        
#         x_t_raw = self._get_raw_state(seg, t)
#         x_seq_raw = np.array([self._get_raw_state(seg, t + i) for i in range(1, self.pred_len + 1)])
#         x_t_local, x_seq_local = self._transform_to_local(x_t_raw, x_seq_raw)
        
#         x_t_norm = (x_t_local - self.stats["state_mean"]) / self.stats["state_std"]
#         x_seq_norm = (x_seq_local - self.stats["state_mean"]) / self.stats["state_std"]
#         u_seq_norm = np.array([(seg['Thrusters_CMD'][:, t+i] - self.stats["ctrl_mean"])/self.stats["ctrl_std"] for i in range(self.pred_len)])
        
#         return torch.FloatTensor(x_t_norm), torch.FloatTensor(x_seq_norm), torch.FloatTensor(u_seq_norm)

#     def __len__(self): 
#         return len(self.indices)


# # ==========================================
# # 2. 导出 OSQP 级 YAML 参数
# # ==========================================
# def export_params_to_yaml(model, stats, save_path="checkpoints/koopman_deploy_params.yaml"):
#     """
#     将归一化参数、边界范围以及 Koopman A、B 矩阵导出为 YAML 格式
#     """
#     # 1. 鲁棒地提取 A 和 B 矩阵
#     def extract_matrix(matrix_attr, is_A_matrix=False):
#         if matrix_attr is None:
#             return []
        
#         mat_tensor = None
#         # 如果是 nn.Linear 等网络层，提取其 .weight
#         if isinstance(matrix_attr, nn.Module) and hasattr(matrix_attr, 'weight'):
#             mat_tensor = matrix_attr.weight.detach().cpu()
#         # 如果直接是 Parameter 或 Tensor
#         elif hasattr(matrix_attr, 'detach'):
#             mat_tensor = matrix_attr.detach().cpu()
            
#         if mat_tensor is not None:
#             # 【核心修改点】：适配 OSQP MPC！
#             # 因为网络使用的是残差学习，真实的离散转移矩阵必须加上单位阵 I
#             if is_A_matrix:
#                 I = torch.eye(mat_tensor.size(0))
#                 mat_tensor = mat_tensor + I
#             return mat_tensor.numpy().tolist()
#         return []

#     try:
#         # 注意这里传入了 is_A_matrix=True
#         A_matrix = extract_matrix(getattr(model, 'A', None), is_A_matrix=True)
#         # B 矩阵不需要加单位阵
#         B_matrix = extract_matrix(getattr(model, 'B', None), is_A_matrix=False)
        
#         if not A_matrix:
#             print("  [警告] 提取的 A 矩阵为空。请检查 koopman.py 中的变量命名。")
#     except Exception as e:
#         print(f"  [错误] 提取 A/B 矩阵时失败: {e}")
#         A_matrix, B_matrix = [], []

#     # 2. 计算方差 (Variance = Std^2)
#     state_var = (stats["state_std"] ** 2).tolist()
#     ctrl_var = (stats["ctrl_std"] ** 2).tolist()

#     # 3. 构建字典
#     yaml_data = {
#         "normalization": {
#             "state_mean": stats["state_mean"].tolist(),
#             "state_variance": state_var,      
#             "state_std": stats["state_std"].tolist(),
#             "ctrl_mean": stats["ctrl_mean"].tolist(),
#             "ctrl_variance": ctrl_var,        
#             "ctrl_std": stats["ctrl_std"].tolist()
#         },
#         "bounds": {
#             "state_min": stats["state_min"].tolist(),
#             "state_max": stats["state_max"].tolist(),
#             "ctrl_min": stats["ctrl_min"].tolist(),
#             "ctrl_max": stats["ctrl_max"].tolist()
#         },
#         "system_matrices": {
#             "A": A_matrix,
#             "B": B_matrix
#         }
#     }

#     # 4. 写入 YAML 文件
#     with open(save_path, 'w', encoding='utf-8') as f:
#         yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False, indent=4)
#     print(f">>> [完美导出] 高维 OSQP 矩阵参数 (已补偿单位阵) 已导出至: {save_path}")


# # ==========================================
# # 3. 高密度数据训练主循环
# # ==========================================
# def train():
#     os.makedirs("checkpoints", exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     pred_len = 20
#     batch_size = 2048
#     # 由于数据密度变高，需要更多 Epoch 保证模型能拟合完整的全包络面
#     epochs = 100 
    
#     print("\n>>> [第一阶段] 正在加载显式切分的高密度数据集...")
#     train_ds = ExplicitKoopmanDataset("koopman_train.npz", pred_len=pred_len, is_train=True)
#     val_ds = ExplicitKoopmanDataset("koopman_val.npz", pred_len=pred_len, stats=train_ds.stats, is_train=False)
    
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
#     print(f"\n>>> [第二阶段] 构建高维线性算子网络 (使用设备: {device})")
#     # 纯线性架构：必须使用 128 维庞大隐空间，以此吸收 63 种 Z 型组合带来的强烈非线性
#     model = HorizontalKoopmanModel(
#         state_dim=6, 
#         control_dim=4, 
#         latent_dim=128, 
#         enc_hidden=[256, 256], 
#         dec_hidden=[256, 256],
#         use_skip=True
#     ).to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
#     # 放缓衰减节奏，让模型充分吸收复杂流场
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
#     # 弯道惩罚加权: 偏航角、横漂v、偏航角速度r 依然保持极高权重，抗衡直道数据的诱惑
#     state_weights = torch.tensor([2.0, 2.0, 2.0, 3.0, 5.0, 5.0], device=device)
#     mse_loss = nn.MSELoss()
    
#     best_val_loss = float('inf')
    
#     for epoch in range(epochs):
#         model.train()
#         total_loss_train = 0.0
        
#         for x_t, x_target_seq, u_seq in train_loader:
#             x_t, x_target_seq, u_seq = x_t.to(device), x_target_seq.to(device), u_seq.to(device)
#             optimizer.zero_grad()
            
#             z_current = model.encode(x_t)
#             x_t_recon = model.reconstruct_state(z_current)
#             pred_states, pred_latents = [], []
            
#             # 多步自回归推演
#             for step in range(pred_len):
#                 z_next = model.latent_step(z_current, u_seq[:, step, :])
#                 pred_latents.append(z_next)
#                 pred_states.append(model.reconstruct_state(z_next))
#                 z_current = z_next
                
#             x_pred_seq = torch.stack(pred_states, dim=1)
            
#             # 代价函数计算
#             loss_pred = torch.mean(state_weights * (x_pred_seq - x_target_seq)**2)
#             loss_recon = torch.mean(state_weights * (x_t_recon - x_t)**2)
#             loss_linear = mse_loss(pred_latents[-1], model.encode(x_target_seq[:, -1, :]))
#             loss_stab = torch.relu(model.spectral_radius() - 1.0) ** 2
            
#             loss = 10 * loss_pred + 1.0 * loss_recon + 10.0 * loss_linear + 0.1 * loss_stab
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             total_loss_train += loss.item()
            
#         # 验证阶段
#         model.eval()
#         total_loss_val = 0.0
#         with torch.no_grad():
#             for x_t, x_target_seq, u_seq in val_loader:
#                 x_t, x_target_seq, u_seq = x_t.to(device), x_target_seq.to(device), u_seq.to(device)
#                 z_current = model.encode(x_t)
#                 pred_states = []
#                 for step in range(pred_len):
#                     z_current = model.latent_step(z_current, u_seq[:, step, :])
#                     pred_states.append(model.reconstruct_state(z_current))
                    
#                 val_loss_weighted = torch.mean(state_weights * (torch.stack(pred_states, dim=1) - x_target_seq)**2)
#                 total_loss_val += val_loss_weighted.item()
                
#         scheduler.step()
#         avg_train_loss = total_loss_train / len(train_loader)
#         avg_val_loss = total_loss_val / len(val_loader)
        
#         print(f"Epoch [{epoch+1:03d}/{epochs}] | LR: {optimizer.param_groups[0]['lr']:.6f} | "
#               f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

#         checkpoint = {
#             'epoch': epoch + 1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'stats': train_ds.stats 
#         }
#         torch.save(checkpoint, os.path.join("checkpoints", "koopman_latest.pth"))

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(checkpoint, os.path.join("checkpoints", "koopman_best.pth"))
#             print(f"  --> [*] 验证集创新低 ({best_val_loss:.4f})，最优模型已保存。")

#     print("\n>>> 训练完成！开始提取部署参数...")
#     checkpoint = torch.load("checkpoints/koopman_best.pth", map_location=device, weights_only=False)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     export_params_to_yaml(model, train_ds.stats)

# if __name__ == "__main__":
#     train()


import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 为了支持实船的 OSQP MPC，必须使用纯线性的 HorizontalKoopmanModel
# 依靠 128 维的隐空间来拟合高密度数据包中的非线性特征
from koopman import HorizontalKoopmanModel

# ==========================================
# 1. 显式读取的高效 Dataset (完美对接分包脚本)
# ==========================================
class ExplicitKoopmanDataset(Dataset):
    def __init__(self, npz_path, pred_len=20, stats=None, is_train=False):
        super().__init__()
        # 直接读取分包脚本生成的高密度数据集
        self.segments = np.load(npz_path, allow_pickle=True)['datas']
        self.pred_len = pred_len
        self.indices = []
        
        # 构建展平的时间索引
        for local_seg_idx, seg in enumerate(self.segments):
            if seg['len'] > pred_len:
                for t in range(0, seg['len'] - pred_len):
                    self.indices.append((local_seg_idx, t))
                    
        # 统计量处理：验证/测试集必须严格复用训练集的 stats，严禁重新计算！
        self.stats = stats if stats is not None else self._compute_local_statistics()
            
        mode_str = "TRAIN" if is_train else "EVAL"
        print(f"Dataset [{mode_str}] 加载完毕: {npz_path} | 包含 {len(self.segments)} 个动态段 | 共 {len(self.indices)} 个推演样本.")

    def _get_raw_state(self, seg, t):
        # [x, y, yaw, u, v, r]
        return np.array([seg['Pos'][0, t], seg['Pos'][1, t], seg['Euler'][2, t],
                         seg['Vel'][0, t], seg['Vel'][1, t], seg['pqr'][0, t]], dtype=np.float32)

    def _transform_to_local(self, x_t_raw, x_seq_raw):
        """将未来轨迹映射到 t 时刻的局部坐标系下"""
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

    def _compute_local_statistics(self):
        print(">>> 正在基于高密度训练集计算全局 Mean & Std...")
        local_states, controls = [], []
        for local_seg_idx, t in self.indices:
            seg = self.segments[local_seg_idx]
            x_t_raw = self._get_raw_state(seg, t)
            x_seq_raw = np.array([self._get_raw_state(seg, t + i) for i in range(1, self.pred_len + 1)])
            x_t_local, x_seq_local = self._transform_to_local(x_t_raw, x_seq_raw)
            
            local_states.append(x_t_local)
            local_states.extend(x_seq_local)
            for i in range(self.pred_len): 
                controls.append(seg['Thrusters_CMD'][:, t + i])
                
        local_states, controls = np.array(local_states, dtype=np.float32), np.array(controls, dtype=np.float32)
        return {
            "state_mean": np.mean(local_states, axis=0), "state_std": np.std(local_states, axis=0) + 1e-6,
            "state_min": np.min(local_states, axis=0), "state_max": np.max(local_states, axis=0),
            "ctrl_mean": np.mean(controls, axis=0), "ctrl_std": np.std(controls, axis=0) + 1e-6,
            "ctrl_min": np.min(controls, axis=0), "ctrl_max": np.max(controls, axis=0)
        }

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
# 2. 导出 OSQP 级 YAML 参数
# ==========================================
def export_params_to_yaml(model, stats, save_path="checkpoints/koopman_deploy_params.yaml"):
    """
    将归一化参数、边界范围以及 Koopman A、B 矩阵导出为 YAML 格式
    """
    def extract_matrix(matrix_attr, is_A_matrix=False):
        if matrix_attr is None:
            return []
        
        mat_tensor = None
        if isinstance(matrix_attr, nn.Module) and hasattr(matrix_attr, 'weight'):
            mat_tensor = matrix_attr.weight.detach().cpu()
        elif hasattr(matrix_attr, 'detach'):
            mat_tensor = matrix_attr.detach().cpu()
            
        if mat_tensor is not None:
            if is_A_matrix:
                I = torch.eye(mat_tensor.size(0))
                mat_tensor = mat_tensor + I
            return mat_tensor.numpy().tolist()
        return []

    try:
        A_matrix = extract_matrix(getattr(model, 'A', None), is_A_matrix=True)
        B_matrix = extract_matrix(getattr(model, 'B', None), is_A_matrix=False)
        
        if not A_matrix:
            print("  [警告] 提取的 A 矩阵为空。请检查 koopman.py 中的变量命名。")
    except Exception as e:
        print(f"  [错误] 提取 A/B 矩阵时失败: {e}")
        A_matrix, B_matrix = [], []

    state_var = (stats["state_std"] ** 2).tolist()
    ctrl_var = (stats["ctrl_std"] ** 2).tolist()

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

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False, indent=4)
    print(f">>> [完美导出] 高维 OSQP 矩阵参数 (已补偿单位阵) 已导出至: {save_path}")


# ==========================================
# 3. 高密度数据训练主循环
# ==========================================
def train():
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pred_len = 20
    batch_size = 2048
    epochs = 100 
    
    print("\n>>> [第一阶段] 正在加载显式切分的高密度数据集...")
    train_ds = ExplicitKoopmanDataset("koopman_train.npz", pred_len=pred_len, is_train=True)
    val_ds = ExplicitKoopmanDataset("koopman_val.npz", pred_len=pred_len, stats=train_ds.stats, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"\n>>> [第二阶段] 构建高维线性算子网络 (使用设备: {device})")
    model = HorizontalKoopmanModel(
        state_dim=6, 
        control_dim=4, 
        latent_dim=128, 
        enc_hidden=[256, 256], 
        dec_hidden=[256, 256],
        use_skip=True
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    # ---------------------------------------------------------------------------------
    # [NEW/MODIFIED 1] 重平衡状态权重：大幅提升纵向速度u和位置(x,y)的惩罚，防止为了拟合偏航而牺牲速度
    # 状态索引: [x, y, yaw, u, v, r]
    # 原先: torch.tensor([1.0, 1.0, 2.0, 1.0, 5.0, 5.0])
    # ---------------------------------------------------------------------------------
    state_weights = torch.tensor([2.0, 2.0, 2.0, 3.0, 5.0, 5.0], device=device)
    mse_loss = nn.MSELoss()
    
    # ---------------------------------------------------------------------------------
    # [NEW/MODIFIED 2] 引入时间递增折扣：越靠后的预测步数，惩罚权重越高，逼迫模型学习长程稳定性
    # 例如：第 1 步权重 1.0，第 20 步权重逐渐上升至 1.95
    # ---------------------------------------------------------------------------------
    temporal_weights_list = [1.0 + 0.05 * step for step in range(pred_len)]
    temporal_weights = torch.tensor(temporal_weights_list, device=device).view(1, -1, 1) # 形状 [1, 20, 1] 方便广播
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss_train = 0.0
        
        for x_t, x_target_seq, u_seq in train_loader:
            x_t, x_target_seq, u_seq = x_t.to(device), x_target_seq.to(device), u_seq.to(device)
            optimizer.zero_grad()
            
            z_current = model.encode(x_t)
            x_t_recon = model.reconstruct_state(z_current)
            pred_states, pred_latents = [], []
            
            # 多步自回归推演
            for step in range(pred_len):
                z_next = model.latent_step(z_current, u_seq[:, step, :])
                pred_latents.append(z_next)
                pred_states.append(model.reconstruct_state(z_next))
                z_current = z_next
                
            x_pred_seq = torch.stack(pred_states, dim=1)
            pred_latents_stack = torch.stack(pred_latents, dim=1) # [B, 20, 128]
            
# ---------------------------------------------------------------------------------
            # [NEW/MODIFIED 3] 全序列隐空间一致性约束：防止自回归过程中间态崩溃
            # 将所有目标序列都投影到隐空间作为 Label (需先展平以适配 Conv1d 的维度要求)
            # ---------------------------------------------------------------------------------
            B, seq_len, dim = x_target_seq.shape
            x_target_seq_flat = x_target_seq.view(B * seq_len, dim)         # 展平为 [B*20, 6]
            target_latents_flat = model.encode(x_target_seq_flat)           # 编码为 [B*20, 128]
            target_latents = target_latents_flat.view(B, seq_len, -1)       # 还原为 [B, 20, 128]
            # 1. 基础预测代价值计算
            loss_pred = torch.mean(temporal_weights * state_weights * (x_pred_seq - x_target_seq)**2)
            loss_recon = torch.mean(state_weights * (x_t_recon - x_t)**2)
            loss_linear = mse_loss(pred_latents_stack, target_latents) 
            loss_stab = torch.relu(model.spectral_radius() - 1.01) ** 2
            
            # ---------------------------------------------------------------------------------
            # [新增] 计算加速度（速度的一阶差分）惩罚，约束运动学平滑性
            # 状态索引: [x, y, yaw, u, v, r]，提取 u, v, r (即索引 3, 4, 5)
            # ---------------------------------------------------------------------------------
            pred_vel = x_pred_seq[:, :, 3:6]
            target_vel = x_target_seq[:, :, 3:6]
            
            # 计算相邻时间步的速度差 (B, 19, 3)
            pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
            target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
            
            # 对速度变化趋势的偏差进行 MSE 惩罚
            loss_acc = torch.mean((pred_acc - target_acc)**2)
            # ---------------------------------------------------------------------------------

            # ---------------------------------------------------------------------------------
            # [修改] 总损失融合：将 5.0 * loss_acc 加进去
            # ---------------------------------------------------------------------------------
            loss = 10.0 * loss_pred + 5.0 * loss_acc + 1.0 * loss_recon + 1.0 * loss_linear + 0.1 * loss_stab
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss_train += loss.item()
            
        # 验证阶段
        model.eval()
        total_loss_val = 0.0
        with torch.no_grad():
            for x_t, x_target_seq, u_seq in val_loader:
                x_t, x_target_seq, u_seq = x_t.to(device), x_target_seq.to(device), u_seq.to(device)
                z_current = model.encode(x_t)
                pred_states = []
                for step in range(pred_len):
                    z_current = model.latent_step(z_current, u_seq[:, step, :])
                    pred_states.append(model.reconstruct_state(z_current))
                    
                # 验证集同样使用时间惩罚来衡量客观指标
                val_loss_weighted = torch.mean(
                    temporal_weights * state_weights * (torch.stack(pred_states, dim=1) - x_target_seq)**2
                )
                total_loss_val += val_loss_weighted.item()
                
        scheduler.step()
        avg_train_loss = total_loss_train / len(train_loader)
        avg_val_loss = total_loss_val / len(val_loader)
        
        print(f"Epoch [{epoch+1:03d}/{epochs}] | LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': train_ds.stats 
        }
        torch.save(checkpoint, os.path.join("checkpoints", "koopman_latest.pth"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join("checkpoints", "koopman_best.pth"))
            print(f"  --> [*] 验证集创新低 ({best_val_loss:.4f})，最优模型已保存。")

    print("\n>>> 训练完成！开始提取部署参数...")
    checkpoint = torch.load("checkpoints/koopman_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    export_params_to_yaml(model, train_ds.stats)

if __name__ == "__main__":
    train()