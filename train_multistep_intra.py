import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 从你现有的 koopman.py 中导入模型
from koopman import DeepKoopmanModel

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
        """将所有数据转为局部坐标系后，再求 Mean 和 Std"""
        print("正在计算局部坐标系下的 Mean 和 Std (约需几秒钟)...")
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
            "ctrl_mean": np.mean(controls, axis=0),
            "ctrl_std": np.std(controls, axis=0) + 1e-6,
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
# 2. 局部坐标系转换函数
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

def collate_fn_local_frame(batch):
    x_t, x_target_seq, u_seq = zip(*batch)
    x_t = torch.stack(x_t, dim=0)
    x_target_seq = torch.stack(x_target_seq, dim=0)
    u_seq = torch.stack(u_seq, dim=0)
    
    x_target_seq_local = batched_sequence_transform_to_local_frame(x_t, x_target_seq)
    
    # 把当前时刻的位置状态归零 (因为是局部坐标系原点)
    x_t_local = x_t.clone()
    x_t_local[:, 0:3] = 0.0
    
    return x_t_local, x_target_seq_local, u_seq


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Horizon-weighted MSE for tensors shaped (B, T, D)."""
    return ((pred - target) ** 2 * weights).mean()


def compute_deep_koopman_rollout_losses(
    model: DeepKoopmanModel,
    x_t: torch.Tensor,
    x_target_seq: torch.Tensor,
    u_seq: torch.Tensor,
    mse_loss: nn.Module,
):
    """Compute standard Deep Koopman losses on the existing rollout batch.

    The batch is already in the current local normalized state definition.  We
    only change how the model is constrained: one encode at t=0, then fully
    open-loop linear latent evolution across the provided control sequence.
    """
    batch_size, pred_len, state_dim = x_target_seq.shape

    z0 = model.encode(x_t)
    x_t_recon = model.reconstruct_state(z0)

    z_current = z0
    pred_states = []
    pred_latents = []
    for step in range(pred_len):
        z_current = model.latent_step(z_current, u_seq[:, step, :])
        pred_latents.append(z_current)
        pred_states.append(model.reconstruct_state(z_current))

    x_pred_seq = torch.stack(pred_states, dim=1)
    z_pred_seq = torch.stack(pred_latents, dim=1)

    flat_targets = x_target_seq.reshape(batch_size * pred_len, state_dim)
    z_target_seq = model.encode(flat_targets).reshape(batch_size, pred_len, -1)
    x_target_recon = model.reconstruct_state(
        z_target_seq.reshape(batch_size * pred_len, -1)
    ).reshape(batch_size, pred_len, state_dim)

    horizon_weights = torch.linspace(
        1.0, 2.0, pred_len, device=x_t.device, dtype=x_t.dtype
    ).view(1, pred_len, 1)

    loss_pred = _weighted_mse(x_pred_seq, x_target_seq, horizon_weights)
    loss_recon = mse_loss(x_t_recon, x_t) + 0.25 * mse_loss(x_target_recon, x_target_seq)
    loss_linear = _weighted_mse(z_pred_seq, z_target_seq, horizon_weights)

    # Re-encoding decoded rollout states catches latent drift that is invisible
    # to state-only prediction loss.
    z_reencoded = model.encode(x_pred_seq.reshape(batch_size * pred_len, state_dim))
    z_reencoded = z_reencoded.reshape_as(z_pred_seq)
    loss_consistency = _weighted_mse(z_reencoded, z_pred_seq.detach(), horizon_weights)

    latent_energy = torch.mean(z_pred_seq ** 2)
    loss_drift = torch.relu(latent_energy - 4.0) ** 2
    loss_stab = model.stability_regularization(rho_limit=0.995)

    losses = {
        "pred": loss_pred,
        "recon": loss_recon,
        "linear": loss_linear,
        "consistency": loss_consistency,
        "drift": loss_drift,
        "stab": loss_stab,
    }
    return losses, x_pred_seq, z_pred_seq

# ==========================================
# 3. 多步训练主循环与模型保存
# ==========================================
def train():
    npz_path = "koopman_dataset_v1.npz"
    ckpt_dir = "checkpoints" # 模型保存路径
    os.makedirs(ckpt_dir, exist_ok=True)
    
    pred_len = 10
    batch_size = 64
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n>>> [第一阶段] 正在加载并进行动态段内切分...")
    train_ds = IntraSegmentKoopmanDataset(npz_path, split_mode='train', pred_len=pred_len)
    val_ds = IntraSegmentKoopmanDataset(npz_path, split_mode='val', pred_len=pred_len, stats=train_ds.stats)
    
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_local_frame)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_local_frame)

    # 把 collate_fn=collate_fn_local_frame 删掉
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = DeepKoopmanModel(state_dim=6, control_dim=4, latent_dim=16).to(device)
    
    # 使用初始较低学习率，配合 StepLR 解决振荡
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
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
        epoch_consistency = 0.0
        epoch_drift = 0.0
        epoch_stab = 0.0
        
        for batch_idx, (x_t, x_target_seq, u_seq) in enumerate(train_loader):
            x_t = x_t.to(device)
            x_target_seq = x_target_seq.to(device)
            u_seq = u_seq.to(device)
            
            optimizer.zero_grad()
            
            losses, _, _ = compute_deep_koopman_rollout_losses(
                model, x_t, x_target_seq, u_seq, mse_loss
            )
            loss_pred = losses["pred"]
            loss_recon = losses["recon"]
            loss_linear = losses["linear"]
            loss_consistency = losses["consistency"]
            loss_drift = losses["drift"]
            loss_stab = losses["stab"]

            loss = (
                1.0 * loss_pred
                + 1.0 * loss_recon
                + 10.0 * loss_linear
                + 1.0 * loss_consistency
                + 0.05 * loss_drift
                + 0.5 * loss_stab
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防止爆炸
            optimizer.step()
            model.project_stable_dynamics_(rho_limit=0.995)
            
            total_loss_train += loss.item()
            epoch_pred += loss_pred.item()
            epoch_recon += loss_recon.item()
            epoch_linear += loss_linear.item()
            epoch_consistency += loss_consistency.item()
            epoch_drift += loss_drift.item()
            epoch_stab += loss_stab.item()
            
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
                
                losses, x_pred_seq, _ = compute_deep_koopman_rollout_losses(
                    model, x_t, x_target_seq, u_seq, mse_loss
                )
                val_loss = (
                    losses["pred"]
                    + losses["recon"]
                    + 10.0 * losses["linear"]
                    + losses["consistency"]
                    + 0.05 * losses["drift"]
                    + 0.5 * losses["stab"]
                )
                total_loss_val += val_loss.item()
                
        avg_val_loss = total_loss_val / len(val_loader)
        
        # StepLR 步进更新学习率 (千万不能忘！)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印这一轮的各项代价函数指标
        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"LR: {current_lr:.6f} | "
              f"Train Total: {avg_train_loss:.4f} "
              f"(Pred:{epoch_pred/num_batches:.4f}, Recon:{epoch_recon/num_batches:.4f}, "
              f"Linear:{epoch_linear/num_batches:.4f}, Cons:{epoch_consistency/num_batches:.4f}, "
              f"Drift:{epoch_drift/num_batches:.4f}, Stab:{epoch_stab/num_batches:.4f}) | "
              f"Val Total: {avg_val_loss:.4f}")

        # --- 模型保存阶段 ---
        # 组装要保存的字典（非常重要：把数据集归一化参数也存进去，实际部署推理时需要用）
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'config': {
                'model_type': 'deep',
                'state_dim': 6,
                'control_dim': 4,
                'latent_dim': 16,
                'enc_hidden': [64, 64],
                'dec_hidden': [64, 64],
                'use_skip': True,
                'pred_len': pred_len,
            },
            'stats': train_ds.stats # 包含 state_mean, state_std, ctrl_mean, ctrl_std
        }

        # 保存每一轮的最新模型
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

if __name__ == "__main__":
    train()