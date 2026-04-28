import os
import time
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from pelican_torch_dataset import PelicanHorizontalTransitionDataset, PelicanSequenceDataset, make_sequence_dataloader
from koopman import HorizontalKoopmanModel, DeepKoopmanModel, create_koopman_model
import matplotlib.pyplot as plt


def batched_transform_to_local_frame(x_t: torch.Tensor, x_tp1: torch.Tensor) -> torch.Tensor:
    """
    [新增] 将下一时刻的全局状态批量转换到当前时刻的局部坐标系下
    输入 shape: (batch_size, 6) -> 对应 [x, y, yaw, u, v, r]
    """
    # 计算全局平移量
    dx = x_tp1[:, 0] - x_t[:, 0]
    dy = x_tp1[:, 1] - x_t[:, 1]
    
    # 提取当前航向角
    yaw_t = x_t[:, 2]
    cos_yaw = torch.cos(yaw_t)
    sin_yaw = torch.sin(yaw_t)
    
    # 旋转到局部坐标系 (左正右负标准)
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    
    # 计算相对航向角并归一化到 [-pi, pi]
    local_yaw = x_tp1[:, 2] - yaw_t
    local_yaw = (local_yaw + torch.pi) % (2 * torch.pi) - torch.pi
    
    # 速度与角速度保持不变 (已经在本体坐标系下)
    # 拼接返回新的 x_tp1_local
    x_tp1_local = torch.stack([
        local_x, local_y, local_yaw, 
        x_tp1[:, 3], x_tp1[:, 4], x_tp1[:, 5]
    ], dim=1)
    
    return x_tp1_local


def batched_sequence_transform_to_local_frame(x_t: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
    """
    [序列新增] 将未来多步的全局状态序列批量转换到当前时刻 x_t 的局部坐标系下
    x_t shape: (batch_size, 6)
    x_seq shape: (batch_size, seq_length, 6)
    返回 shape: (batch_size, seq_length, 6)
    """
    # 扩展 x_t 以匹配 seq_length 维度
    dx = x_seq[:, :, 0] - x_t[:, 0].unsqueeze(1)
    dy = x_seq[:, :, 1] - x_t[:, 1].unsqueeze(1)
    
    yaw_t = x_t[:, 2].unsqueeze(1)
    cos_yaw = torch.cos(yaw_t)
    sin_yaw = torch.sin(yaw_t)
    
    # 旋转到局部坐标系 (左正右负标准)
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    
    # 计算相对航向角并归一化到 [-pi, pi]
    local_yaw = x_seq[:, :, 2] - yaw_t
    local_yaw = (local_yaw + torch.pi) % (2 * torch.pi) - torch.pi
    
    # 拼接返回新的 x_seq_local
    x_seq_local = torch.stack([
        local_x, local_y, local_yaw, 
        x_seq[:, :, 3], x_seq[:, :, 4], x_seq[:, :, 5]
    ], dim=2)
    
    return x_seq_local


# TensorBoard logging via tensorboardX
try:
    from tensorboardX import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None


@dataclass
class Config:
    """Configuration for training horizontal plane Koopman model."""
    npz_path: str = "pelican_dataset_horizontal.npz"
    device: str = "cpu"
    batch_size: int = 64
    latent_dim: int = 16
    enc_hidden: Tuple[int, int] = (64, 64)
    dec_hidden: Tuple[int, int] = (64, 64)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    log_every: int = 100  # steps
    save_every: int = 1000  # steps
    seed: int = 42
    ckpt_dir: str = "checkpoints_horizontal"
    
    # Model type: 'horizontal' or 'deep'
    model_type: str = "horizontal"
    
    # Loss weights
    recon_loss_weight: float = 1.0
    linear_loss_weight: float = 50.0
    pred_loss_weight: float = 1.0
    stab_loss_weight: float = 1.0
    
    # Data normalization
    normalize_data: bool = True
    
    # Sequence training (if sequence_length > 1)
    use_sequence: bool = False
    seq_length: int = 10
    pred_length: int = 1
    seq_stride: int = 5
    
    # Skip connection in horizontal model
    use_skip: bool = True


class StandardScaler:
    """Standard scaler for normalization (zero mean, unit variance)."""
    
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.u_mean = None
        self.u_std = None
        
    def fit(self, x_all: np.ndarray, u_all: np.ndarray):
        """Fit scaler to data."""
        self.x_mean = x_all.mean(axis=0, keepdims=True)
        self.x_std = x_all.std(axis=0, keepdims=True)
        self.x_std[self.x_std < 1e-8] = 1.0
        
        self.u_mean = u_all.mean(axis=0, keepdims=True)
        self.u_std = u_all.std(axis=0, keepdims=True)
        self.u_std[self.u_std < 1e-8] = 1.0
        
    def transform_x(self, x: torch.Tensor) -> torch.Tensor:
        """Transform state to normalized space."""
        if self.x_mean is None or self.x_std is None:
            return x
        x_mean = torch.tensor(self.x_mean, device=x.device, dtype=x.dtype)
        x_std = torch.tensor(self.x_std, device=x.device, dtype=x.dtype)
        return (x - x_mean) / x_std
    
    def transform_u(self, u: torch.Tensor) -> torch.Tensor:
        """Transform control to normalized space."""
        if self.u_mean is None or self.u_std is None:
            return u
        u_mean = torch.tensor(self.u_mean, device=u.device, dtype=u.dtype)
        u_std = torch.tensor(self.u_std, device=u.device, dtype=u.dtype)
        return (u - u_mean) / u_std
    
    def inverse_transform_x(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Inverse transform state from normalized space."""
        if self.x_mean is None or self.x_std is None:
            return x_norm
        x_mean = torch.tensor(self.x_mean, device=x_norm.device, dtype=x_norm.dtype)
        x_std = torch.tensor(self.x_std, device=x_norm.device, dtype=x_norm.dtype)
        return x_norm * x_std + x_mean
    
    def inverse_transform_u(self, u_norm: torch.Tensor) -> torch.Tensor:
        """Inverse transform control from normalized space."""
        if self.u_mean is None or self.u_std is None:
            return u_norm
        u_mean = torch.tensor(self.u_mean, device=u_norm.device, dtype=u_norm.dtype)
        u_std = torch.tensor(self.u_std, device=u_norm.device, dtype=u_norm.dtype)
        return u_norm * u_std + u_mean


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_tb(logdir: str, run_name: str = None):
    """Initialize TensorBoard SummaryWriter."""
    class _DummyWriter:
        def add_scalar(self, *_args, **_kwargs):
            pass
        def add_text(self, *_args, **_kwargs):
            pass
        def flush(self):
            pass
        def close(self):
            pass
    
    if SummaryWriter is None:
        print("tensorboardX not available; install with 'pip install tensorboardX' to enable TB logs.")
        return _DummyWriter()
    
    name = run_name or f"koopman-horizontal-{int(time.time())}"
    run_dir = os.path.join(logdir, name)
    os.makedirs(run_dir, exist_ok=True)
    try:
        return SummaryWriter(logdir=run_dir)
    except Exception as e:
        print(f"Failed to initialize TensorBoard writer at '{run_dir}': {e}")
        return _DummyWriter()


def collect_data_for_scaling(dataset, max_samples=10000):
    """Collect data for fitting scaler (支持 Subset 和 坐标转换)."""
    x_t_list = []
    x_tp1_list = []
    u_list = []
    
    n_samples = min(len(dataset), max_samples)
    print(f"Collecting {n_samples} samples for scaling...")
    
    for i in range(n_samples):
        try:
            # 直接通过索引获取数据，兼容 Subset 和原生 Dataset
            item = dataset[i]
            
            # 判断返回的是不是 3 个元素
            if len(item) == 3:
                x, y, u = item
                
                # 检查是单步 Transition 还是 Sequence 序列
                if (isinstance(x, torch.Tensor) and x.ndim > 1) or (isinstance(x, np.ndarray) and x.ndim > 1):
                    # Sequence dataset
                    x_t_list.append(x[-1])
                    x_tp1_list.append(y[0])
                    u_list.append(u[-1])
                else:
                    # Transition dataset (单步)
                    x_t_list.append(x)
                    x_tp1_list.append(y)
                    u_list.append(u)
                    
        except Exception as e:
            print(f"Error collecting sample {i}: {e}")
            continue
            
    if x_t_list:
        # 判断数据类型并统一转换为 PyTorch Tensor
        if isinstance(x_t_list[0], torch.Tensor):
            x_t_tensor = torch.stack(x_t_list, dim=0).float()
            x_tp1_tensor = torch.stack(x_tp1_list, dim=0).float()
            u_all = torch.stack(u_list, dim=0).numpy()
        else:
            x_t_tensor = torch.tensor(np.stack(x_t_list, axis=0), dtype=torch.float32)
            x_tp1_tensor = torch.tensor(np.stack(x_tp1_list, axis=0), dtype=torch.float32)
            u_all = np.stack(u_list, axis=0)
            
        # ========== 应用坐标转换逻辑 ==========
        x_tp1_local = batched_transform_to_local_frame(x_t_tensor, x_tp1_tensor)
        x_t_local = x_t_tensor.clone()
        x_t_local[:, 0:3] = 0.0
        
        x_all = torch.cat([x_t_local, x_tp1_local], dim=0).numpy()
        
        print(f"Successfully collected and transformed {len(x_t_list)} samples.")
        return x_all, u_all
    else:
        raise ValueError("No data collected for scaling. 检查 dataset 的 __getitem__ 输出格式！")


def train(args=None):
    """Main training function for horizontal Koopman model."""
    cfg = Config()
    
    # Apply CLI overrides
    if args is not None:
        for key, value in vars(args).items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
    
    set_seed(cfg.seed)
    
    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but not available. Use --device cpu")
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")
    
    print(f"Training on device: {device}")
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    
    if cfg.use_sequence:
        print(f"Using sequence training with seq_len={cfg.seq_length}, pred_len={cfg.pred_length}")
        train_loader, val_loader, test_loader, norm_stats = make_sequence_dataloader(
            npz_path=cfg.npz_path,
            seq_len=cfg.seq_length,
            pred_len=cfg.pred_length,
            stride=cfg.seq_stride,
            batch_size=cfg.batch_size,
            use_normalized=cfg.normalize_data,
            split_ratios=(1 - cfg.val_fraction - cfg.test_fraction, 
                         cfg.val_fraction, cfg.test_fraction)
        )
        
        scaler = StandardScaler()
        if cfg.normalize_data and norm_stats is not None:
            scaler.x_mean = norm_stats['state_mean']
            scaler.x_std = norm_stats['state_std']
            scaler.u_mean = norm_stats['control_mean']
            scaler.u_std = norm_stats['control_std']
    else:
        print("Using transition dataset (single-step predictions)")
        full_ds = PelicanHorizontalTransitionDataset(
            cfg.npz_path, 
            return_flight_index=False,
            use_normalized=False
        )
        n_total = len(full_ds)
        n_test = int(n_total * cfg.test_fraction)
        n_val = int(n_total * cfg.val_fraction)
        n_train = n_total - n_val - n_test
        
        train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test])
        
        scaler = StandardScaler()
        if cfg.normalize_data:
            x_all, u_all = collect_data_for_scaling(train_ds)
            scaler.fit(x_all, u_all)
        
        def create_loader(dataset, shuffle):
            def collate_fn(batch):
                x_t, x_tp1, u_t = zip(*batch)
                x_t = torch.stack(x_t, dim=0)
                x_tp1 = torch.stack(x_tp1, dim=0)
                u_t = torch.stack(u_t, dim=0)
                
                # ========== [新增转换逻辑] ==========
                x_tp1 = batched_transform_to_local_frame(x_t, x_tp1)
                x_t_local = x_t.clone()
                x_t_local[:, 0:3] = 0.0
                x_t = x_t_local
                
                if cfg.normalize_data:
                    x_t = scaler.transform_x(x_t)
                    x_tp1 = scaler.transform_x(x_tp1)
                    u_t = scaler.transform_u(u_t)
                return x_t, x_tp1, u_t
            
            return DataLoader(
                dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn
            )
        
        train_loader = create_loader(train_ds, shuffle=True)
        val_loader = create_loader(val_ds, shuffle=False)
        test_loader = create_loader(test_ds, shuffle=False)
    
    if cfg.model_type == "horizontal":
        model = HorizontalKoopmanModel(
            state_dim=6, control_dim=4, latent_dim=cfg.latent_dim,
            enc_hidden=list(cfg.enc_hidden), dec_hidden=list(cfg.dec_hidden), use_skip=cfg.use_skip
        ).to(device)
    elif cfg.model_type == "deep":
        model = DeepKoopmanModel(
            state_dim=6, control_dim=4, latent_dim=cfg.latent_dim,
            enc_hidden=list(cfg.enc_hidden), dec_hidden=list(cfg.dec_hidden)
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    mse_loss = nn.MSELoss()
    
    writer = init_tb("runs_horizontal", run_name=args.run_name if args else None)
    global_step = 0
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {cfg.epochs} epochs...")
    
    for epoch in range(cfg.epochs):
        # ---------------- TRAINING PHASE ----------------
        model.train()
        train_losses = {'total': 0.0, 'recon': 0.0, 'linear': 0.0, 'pred': 0.0, 'stab': 0.0}
        n_train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if cfg.use_sequence:
                input_states, input_controls, target_states = batch
                input_states, input_controls, target_states = input_states.to(device), input_controls.to(device), target_states.to(device)
                
                # 取出起点 x_t
                x_t = input_states[:, -1, :]
                seq_length = target_states.size(1)
                
                # 1. 坐标转换：将未来所有目标点转换到 x_t 的坐标系下
                x_target_seq_local = batched_sequence_transform_to_local_frame(x_t, target_states)
                
                # 2. x_t 自身位置归零 (处于局部系原点)
                x_t_local = x_t.clone()
                x_t_local[:, 0:3] = 0.0
                
                # 编码当前状态
                z_t = model.encode(x_t_local)
                x_t_recon = model.reconstruct_state(z_t)
                loss_recon = mse_loss(x_t_local, x_t_recon)
                
                loss_linear, loss_pred = 0.0, 0.0
                z_curr = z_t
                
                # 自回归循环推演未来 N 步
                for i in range(seq_length):
                    # 注意：请确保你的 input_controls 数据包含了对应 target_states 长度的动作序列
                    # 如果 input_controls 对应历史输入，你需要从 dataset 侧修改以输出未来 controls
                    u_curr = input_controls[:, i, :] if input_controls.size(1) >= seq_length else input_controls[:, -1, :] 
                    
                    z_next_hat = model.latent_step(z_curr, u_curr)
                    x_next_hat = model.reconstruct_state(z_next_hat)
                    
                    x_target_step = x_target_seq_local[:, i, :]
                    z_next_true = model.encode(x_target_step)
                    
                    loss_linear += mse_loss(z_next_true, z_next_hat)
                    loss_pred += mse_loss(x_target_step, x_next_hat)
                    
                    z_curr = z_next_hat
                
                loss_linear /= seq_length
                loss_pred /= seq_length
                rho = model.spectral_radius()
                loss_stab = torch.relu(rho - 1.0) ** 2
                
                loss = (cfg.recon_loss_weight * loss_recon +
                       cfg.linear_loss_weight * loss_linear +
                       cfg.pred_loss_weight * loss_pred +
                       cfg.stab_loss_weight * loss_stab)
            else:
                x_t, x_tp1, u_t = batch
                x_t, x_tp1, u_t = x_t.to(device), x_tp1.to(device), u_t.to(device)
                
                z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat = model(x_t, u_t, x_tp1)
                
                loss_recon = mse_loss(x_t, x_t_recon)
                loss_linear = mse_loss(z_tp1, z_tp1_hat)
                loss_pred = mse_loss(x_tp1, x_tp1_hat)
                rho = model.spectral_radius()
                loss_stab = torch.relu(rho - 1.0) ** 2
                
                loss = (cfg.recon_loss_weight * loss_recon +
                       cfg.linear_loss_weight * loss_linear +
                       cfg.pred_loss_weight * loss_pred +
                       cfg.stab_loss_weight * loss_stab)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses['total'] += loss.item()
            train_losses['recon'] += loss_recon.item()
            train_losses['linear'] += loss_linear.item()
            train_losses['pred'] += loss_pred.item()
            train_losses['stab'] += loss_stab.item()
            n_train_batches += 1
            
            if global_step % cfg.log_every == 0:
                print(f"Epoch {epoch+1}/{cfg.epochs}, Step {global_step}: "
                      f"Loss={loss.item():.6f}, Recon={loss_recon.item():.6f}, "
                      f"Linear={loss_linear.item():.6f}, Pred={loss_pred.item():.6f}")
            global_step += 1
        
        for key in train_losses: train_losses[key] /= n_train_batches
        
        # ---------------- VALIDATION PHASE ----------------
        model.eval()
        val_losses = {'total': 0.0, 'recon': 0.0, 'linear': 0.0, 'pred': 0.0, 'stab': 0.0}
        n_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if cfg.use_sequence:
                    input_states, input_controls, target_states = batch
                    input_states, input_controls, target_states = input_states.to(device), input_controls.to(device), target_states.to(device)
                    x_t = input_states[:, -1, :]
                    seq_length = target_states.size(1)
                    x_target_seq_local = batched_sequence_transform_to_local_frame(x_t, target_states)
                    x_t_local = x_t.clone()
                    x_t_local[:, 0:3] = 0.0
                    
                    z_t = model.encode(x_t_local)
                    x_t_recon = model.reconstruct_state(z_t)
                    loss_recon = mse_loss(x_t_local, x_t_recon)
                    loss_linear, loss_pred = 0.0, 0.0
                    z_curr = z_t
                    
                    for i in range(seq_length):
                        u_curr = input_controls[:, i, :] if input_controls.size(1) >= seq_length else input_controls[:, -1, :] 
                        z_next_hat = model.latent_step(z_curr, u_curr)
                        x_next_hat = model.reconstruct_state(z_next_hat)
                        x_target_step = x_target_seq_local[:, i, :]
                        z_next_true = model.encode(x_target_step)
                        loss_linear += mse_loss(z_next_true, z_next_hat)
                        loss_pred += mse_loss(x_target_step, x_next_hat)
                        z_curr = z_next_hat
                        
                    loss_linear /= seq_length
                    loss_pred /= seq_length
                    rho = model.spectral_radius()
                    loss_stab = torch.relu(rho - 1.0) ** 2
                else:
                    x_t, x_tp1, u_t = batch
                    x_t, x_tp1, u_t = x_t.to(device), x_tp1.to(device), u_t.to(device)
                    z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat = model(x_t, u_t, x_tp1)
                    loss_recon = mse_loss(x_t, x_t_recon)
                    loss_linear = mse_loss(z_tp1, z_tp1_hat)
                    loss_pred = mse_loss(x_tp1, x_tp1_hat)
                    rho = model.spectral_radius()
                    loss_stab = torch.relu(rho - 1.0) ** 2
                
                loss = (cfg.recon_loss_weight * loss_recon +
                       cfg.linear_loss_weight * loss_linear +
                       cfg.pred_loss_weight * loss_pred +
                       cfg.stab_loss_weight * loss_stab)
                
                val_losses['total'] += loss.item()
                val_losses['recon'] += loss_recon.item()
                val_losses['linear'] += loss_linear.item()
                val_losses['pred'] += loss_pred.item()
                val_losses['stab'] += loss_stab.item()
                n_val_batches += 1
                
        for key in val_losses: val_losses[key] /= n_val_batches
        scheduler.step(val_losses['total'])
        
        print(f"\nEpoch {epoch+1}/{cfg.epochs} Summary:")
        print(f"  Train Loss: {train_losses['total']:.6f} | Val Loss: {val_losses['total']:.6f}")
        
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_model_path = os.path.join(cfg.ckpt_dir, "best_model.pt")
            torch.save({'model_state_dict': model.state_dict()}, best_model_path)
            print(f"  Saved best model to {best_model_path}")
            
    # ---------------- TESTING PHASE ----------------
    print("\n" + "="*50)
    print("Final evaluation on test set:")
    print("="*50)

    all_gt, all_pred = [], []
    model.eval()
    test_losses = {'total': 0.0, 'recon': 0.0, 'linear': 0.0, 'pred': 0.0, 'stab': 0.0}
    n_test_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if cfg.use_sequence:
                input_states, input_controls, target_states = batch
                input_states, input_controls, target_states = input_states.to(device), input_controls.to(device), target_states.to(device)
                x_t = input_states[:, -1, :]
                seq_length = target_states.size(1)
                
                x_target_seq_local = batched_sequence_transform_to_local_frame(x_t, target_states)
                x_t_local = x_t.clone()
                x_t_local[:, 0:3] = 0.0
                
                z_t = model.encode(x_t_local)
                x_t_recon = model.reconstruct_state(z_t)
                loss_recon = mse_loss(x_t_local, x_t_recon)
                loss_linear, loss_pred = 0.0, 0.0
                z_curr = z_t
                
                x_pred_seq = []
                for i in range(seq_length):
                    u_curr = input_controls[:, i, :] if input_controls.size(1) >= seq_length else input_controls[:, -1, :] 
                    z_next_hat = model.latent_step(z_curr, u_curr)
                    x_next_hat = model.reconstruct_state(z_next_hat)
                    x_target_step = x_target_seq_local[:, i, :]
                    z_next_true = model.encode(x_target_step)
                    loss_linear += mse_loss(z_next_true, z_next_hat)
                    loss_pred += mse_loss(x_target_step, x_next_hat)
                    
                    z_curr = z_next_hat
                    x_pred_seq.append(x_next_hat)
                
                # 为了复用原始的绘图代码，我们将 sequence 展平为 (N, 6) 的形状
                all_gt.append(x_target_seq_local.view(-1, 6).cpu())
                all_pred.append(torch.stack(x_pred_seq, dim=1).view(-1, 6).cpu())
                
                loss_linear /= seq_length
                loss_pred /= seq_length
                rho = model.spectral_radius()
                loss_stab = torch.relu(rho - 1.0) ** 2
            else:
                x_t, x_tp1, u_t = batch
                x_t, x_tp1, u_t = x_t.to(device), x_tp1.to(device), u_t.to(device)
                z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat = model(x_t, u_t, x_tp1)
                all_gt.append(x_tp1.cpu())
                all_pred.append(x_tp1_hat.cpu())
                
                loss_recon = mse_loss(x_t, x_t_recon)
                loss_linear = mse_loss(z_tp1, z_tp1_hat)
                loss_pred = mse_loss(x_tp1, x_tp1_hat)
                rho = model.spectral_radius()
                loss_stab = torch.relu(rho - 1.0) ** 2
            
            loss = (cfg.recon_loss_weight * loss_recon +
                   cfg.linear_loss_weight * loss_linear +
                   cfg.pred_loss_weight * loss_pred +
                   cfg.stab_loss_weight * loss_stab)
            
            test_losses['total'] += loss.item()
            test_losses['recon'] += loss_recon.item()
            test_losses['linear'] += loss_linear.item()
            test_losses['pred'] += loss_pred.item()
            test_losses['stab'] += loss_stab.item()
            n_test_batches += 1

    all_gt = torch.cat(all_gt, dim=0).numpy()
    all_pred = torch.cat(all_pred, dim=0).numpy()

    state_names = ["x", "y", "yaw", "u", "v", "r"]
    save_dir = os.path.join(cfg.ckpt_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    n_plot = min(1000, len(all_gt))

    for i in range(all_gt.shape[1]):
        plt.figure(figsize=(10,4))
        plt.plot(all_gt[:n_plot, i], label="Ground Truth")
        plt.plot(all_pred[:n_plot, i], label="Prediction")
        plt.title(f"Test Comparison - {state_names[i]}")
        plt.xlabel("Sample")
        plt.ylabel(state_names[i])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"test_compare_{state_names[i]}.png"))
        plt.close()

    error = np.abs(all_gt - all_pred)

    for i in range(all_gt.shape[1]):
        plt.figure(figsize=(10,4))
        plt.plot(error[:n_plot, i])
        plt.title(f"Prediction Error - {state_names[i]}")
        plt.xlabel("Sample")
        plt.ylabel("Absolute Error")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"test_error_{state_names[i]}.png"))
        plt.close()

    for key in test_losses: test_losses[key] /= n_test_batches
    print(f"\nTest Results:")
    print(f"  Total Loss: {test_losses['total']:.6f} | Pred Loss: {test_losses['pred']:.6f}")
    
    writer.close()


def export_local_data_to_npz(dataset, filename="local_dataset.npz"):
    # (保持原有功能，未修改)
    pass

def plot_local_transitions(x_tp1_np):
    # (保持原有功能，未修改)
    pass

def save_and_plot_local_data(dataset, filename="processed_data_local.npz", plot_path="data_visualization.png"):
    # (保持原有功能，未修改)
    pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Koopman model for horizontal ship motion")
    
    parser.add_argument("--npz-path", type=str, default="sim.npz", help="Path to NPZ dataset file")
    parser.add_argument("--model-type", type=str, choices=["horizontal", "deep"], default="horizontal")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--enc-hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--dec-hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--use-skip", action="store_true", default=True)
    parser.add_argument("--recon-loss-weight", type=float, default=1.0)
    parser.add_argument("--linear-loss-weight", type=float, default=10.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)
    parser.add_argument("--stab-loss-weight", type=float, default=0.1)
    
    parser.add_argument("--use-sequence", action="store_true")
    parser.add_argument("--seq-length", type=int, default=10)
    parser.add_argument("--pred-length", type=int, default=1)
    parser.add_argument("--seq-stride", type=int, default=5)
    
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--normalize-data", action="store_true", default=True)
    
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_horizontal")
    parser.add_argument("--run-name", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.enc_hidden: args.enc_hidden = tuple(args.enc_hidden)
    if args.dec_hidden: args.dec_hidden = tuple(args.dec_hidden)
    
    train(args)


if __name__ == "__main__":
    main()