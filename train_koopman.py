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


def batched_sequence_transform_to_local_frame(x_t: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
    """
    将未来多步的全局状态序列批量转换到当前时刻 x_t 的局部坐标系下
    x_t shape: (batch_size, 6)
    x_seq shape: (batch_size, seq_length, 6)
    返回 shape: (batch_size, seq_length, 6)
    """
    # 扩展 x_t 以匹配 seq_length 维度
    # shape 变为 (batch_size, 1) 以便利用 broadcasting
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
            
            # 判断返回的是不是 3 个元素 (x_t, x_tp1, u_t) 或 (x_seq, u_seq, target_seq)
            if len(item) == 3:
                x, y, u = item
                
                # 检查是单步 Transition 还是 Sequence 序列
                if (isinstance(x, torch.Tensor) and x.ndim > 1) or (isinstance(x, np.ndarray) and x.ndim > 1):
                    # Sequence dataset: 取输入序列的最后一步作为 t，目标序列的第一步作为 t+1
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
        # 1. 将 t+1 时刻转换到以 t 时刻为原点的局部坐标系下
        x_tp1_local = batched_transform_to_local_frame(x_t_tensor, x_tp1_tensor)
        
        # 2. 当前状态的绝对位置 (x, y) 和偏航角 (yaw) 归零
        x_t_local = x_t_tensor.clone()
        x_t_local[:, 0:3] = 0.0
        
        # 3. 将 t 和 t+1 时刻的局部状态合并起来，一起拟合 Scaler 的统计分布
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
    
    # Set random seed
    set_seed(cfg.seed)
    
    # Device setup
    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but not available. Use --device cpu")
        device = torch.device(cfg.device)
        # Test CUDA
        try:
            a = torch.randn(2, 2, device=device)
            b = torch.randn(2, 2, device=device)
            _ = a @ b
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError(f"CUDA initialization failed: {e}. Use --device cpu")
    else:
        device = torch.device("cpu")
    
    print(f"Training on device: {device}")
    
    # Create checkpoint directory
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    
    # Create dataset
    if cfg.use_sequence:
        print(f"Using sequence training with seq_len={cfg.seq_length}, pred_len={cfg.pred_length}")
        
        # Create sequence dataloaders
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
        
        # Initialize scaler with normalization stats
        scaler = StandardScaler()
        if cfg.normalize_data and norm_stats is not None:
            scaler.x_mean = norm_stats['state_mean']
            scaler.x_std = norm_stats['state_std']
            scaler.u_mean = norm_stats['control_mean']
            scaler.u_std = norm_stats['control_std']
            
            print(f"Normalization stats loaded:")
            print(f"  State mean: {scaler.x_mean.flatten()}")
            print(f"  State std: {scaler.x_std.flatten()}")
            print(f"  Control mean: {scaler.u_mean.flatten()}")
            print(f"  Control std: {scaler.u_std.flatten()}")
    else:
        # Use transition dataset
        print("Using transition dataset (single-step predictions)")
        
        # Load full dataset
        full_ds = PelicanHorizontalTransitionDataset(
            cfg.npz_path, 
            return_flight_index=False,
            use_normalized=False
        )
        
        # Split dataset
        n_total = len(full_ds)
        n_test = int(n_total * cfg.test_fraction)
        n_val = int(n_total * cfg.val_fraction)
        n_train = n_total - n_val - n_test
        
        train_ds, val_ds, test_ds = random_split(
            full_ds, [n_train, n_val, n_test]
        )

        # # 1. 导出测试集数据到本地
        # export_local_data_to_npz(test_ds, "processed_test_data.npz")

        # # 2. 读取并作图演示
        # data = np.load("processed_test_data.npz")
        # plot_local_transitions(data['x_tp1'])
        # save_and_plot_local_data(test_ds, filename="test_local_coords.npz", plot_path="test_trajectory_plot.png")

        
        # Fit scaler on training data
        scaler = StandardScaler()
        if cfg.normalize_data:
            x_all, u_all = collect_data_for_scaling(train_ds)
            scaler.fit(x_all, u_all)
            print(f"Fitted scaler on {len(x_all)} samples")
        
        # Create dataloaders
        def create_loader(dataset, shuffle):
            def collate_fn(batch):
                x_t, x_tp1, u_t = zip(*batch)
                x_t = torch.stack(x_t, dim=0)
                x_tp1 = torch.stack(x_tp1, dim=0)
                u_t = torch.stack(u_t, dim=0)
                
                # ========== [新增转换逻辑] ==========
                # 1. 将 x_tp1 转为以 x_t 为原点的局部坐标
                x_tp1 = batched_transform_to_local_frame(x_t, x_tp1)
                
                # 2. 将 x_t 的位置和偏航角清零 (0, 0, 0)
                # 因为在局部坐标系下，车辆始终从 (0,0,0) 出发。
                # 这样 Koopman 模型就只会基于当前的 u, v, r 和 控制量 u_t 进行预测
                x_t_local = x_t.clone()
                x_t_local[:, 0:3] = 0.0
                x_t = x_t_local
                # ====================================
                
                if cfg.normalize_data:
                    x_t = scaler.transform_x(x_t)
                    x_tp1 = scaler.transform_x(x_tp1)
                    u_t = scaler.transform_u(u_t)
                
                return x_t, x_tp1, u_t
            
            return DataLoader(
                dataset, 
                batch_size=cfg.batch_size, 
                shuffle=shuffle,
                num_workers=0,
                collate_fn=collate_fn
            )
        
        train_loader = create_loader(train_ds, shuffle=True)
        val_loader = create_loader(val_ds, shuffle=False)
        test_loader = create_loader(test_ds, shuffle=False)
    
    # Create model
    if cfg.model_type == "horizontal":
        model = HorizontalKoopmanModel(
            state_dim=6,
            control_dim=4,
            latent_dim=cfg.latent_dim,
            enc_hidden=list(cfg.enc_hidden),
            dec_hidden=list(cfg.dec_hidden),
            use_skip=cfg.use_skip
        ).to(device)
    elif cfg.model_type == "deep":
        model = DeepKoopmanModel(
            state_dim=6,
            control_dim=4,
            latent_dim=cfg.latent_dim,
            enc_hidden=list(cfg.enc_hidden),
            dec_hidden=list(cfg.dec_hidden)
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss function
    mse_loss = nn.MSELoss()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {cfg.model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"State dimension: {model.state_dim}")
    print(f"Control dimension: {model.control_dim}")
    print(f"Latent dimension: {cfg.latent_dim}")
    
    # Print dataset info
    print(f"\nDataset info:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {cfg.batch_size}")
    
    # TensorBoard writer
    writer = init_tb("runs_horizontal", run_name=args.run_name if args else None)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {cfg.epochs} epochs...")
    
    for epoch in range(cfg.epochs):
        # Training phase
        model.train()
        train_losses = {
            'total': 0.0,
            'recon': 0.0,
            'linear': 0.0,
            'pred': 0.0,
            'stab': 0.0
        }
        n_train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if cfg.use_sequence:
                # Sequence data
                input_states, input_controls, target_states = batch
                input_states = input_states.to(device)
                input_controls = input_controls.to(device)
                target_states = target_states.to(device)
                
                # For sequence training, we predict multiple steps
                # Here we use the first step prediction for simplicity
                # TODO: Implement multi-step sequence loss
                x_t = input_states[:, -1, :]  # Last state in sequence
                x_tp1 = target_states[:, 0, :]  # First target state
                u_t = input_controls[:, -1, :]  # Last control in sequence
            else:
                # Transition data
                x_t, x_tp1, u_t = batch
                x_t = x_t.to(device)
                x_tp1 = x_tp1.to(device)
                u_t = u_t.to(device)
                # print("single step batch shapes:", x_t.shape, x_tp1.shape, u_t.shape)
            
            # Forward pass
            z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat = model(x_t, u_t, x_tp1)
            
            # Compute losses
            loss_recon = mse_loss(x_t, x_t_recon)
            loss_linear = mse_loss(z_tp1, z_tp1_hat)
            loss_pred = mse_loss(x_tp1, x_tp1_hat)
            
            # Spectral radius penalty for stability
            rho = model.spectral_radius()
            loss_stab = torch.relu(rho - 1.0) ** 2
            
            # Total loss
            loss = (cfg.recon_loss_weight * loss_recon +
                   cfg.linear_loss_weight * loss_linear +
                   cfg.pred_loss_weight * loss_pred +
                   cfg.stab_loss_weight * loss_stab)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            train_losses['total'] += loss.item()
            train_losses['recon'] += loss_recon.item()
            train_losses['linear'] += loss_linear.item()
            train_losses['pred'] += loss_pred.item()
            train_losses['stab'] += loss_stab.item()
            n_train_batches += 1
            
            # Logging
            if global_step % cfg.log_every == 0:
                # TensorBoard logging
                writer.add_scalar('train/loss_total', loss.item(), global_step)
                writer.add_scalar('train/loss_recon', loss_recon.item(), global_step)
                writer.add_scalar('train/loss_linear', loss_linear.item(), global_step)
                writer.add_scalar('train/loss_pred', loss_pred.item(), global_step)
                writer.add_scalar('train/loss_stab', loss_stab.item(), global_step)
                writer.add_scalar('train/spectral_radius', rho.item(), global_step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                
                print(f"Epoch {epoch+1}/{cfg.epochs}, Step {global_step}: "
                      f"Loss={loss.item():.6f}, Recon={loss_recon.item():.6f}, "
                      f"Linear={loss_linear.item():.6f}, Pred={loss_pred.item():.6f}, "
                      f"Rho={rho.item():.4f}")
            
            # Save checkpoint
            if global_step % cfg.save_every == 0 and global_step > 0:
                checkpoint_path = os.path.join(cfg.ckpt_dir, f"checkpoint_step_{global_step}.pt")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': loss.item(),
                    'config': cfg.__dict__,
                    'scaler': {
                        'x_mean': scaler.x_mean.tolist() if scaler.x_mean is not None else None,
                        'x_std': scaler.x_std.tolist() if scaler.x_std is not None else None,
                        'u_mean': scaler.u_mean.tolist() if scaler.u_mean is not None else None,
                        'u_std': scaler.u_std.tolist() if scaler.u_std is not None else None,
                    } if cfg.normalize_data else None
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            global_step += 1
        
        # Average training losses for the epoch
        if n_train_batches > 0:
            for key in train_losses:
                train_losses[key] /= n_train_batches
        
        # Validation phase
        model.eval()
        val_losses = {
            'total': 0.0,
            'recon': 0.0,
            'linear': 0.0,
            'pred': 0.0,
            'stab': 0.0
        }
        n_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if cfg.use_sequence:
                    input_states, input_controls, target_states = batch
                    input_states = input_states.to(device)
                    input_controls = input_controls.to(device)
                    target_states = target_states.to(device)
                    
                    x_t = input_states[:, -1, :]
                    x_tp1 = target_states[:, 0, :]
                    u_t = input_controls[:, -1, :]
                else:
                    x_t, x_tp1, u_t = batch
                    x_t = x_t.to(device)
                    x_tp1 = x_tp1.to(device)
                    u_t = u_t.to(device)
                
                # Forward pass
                z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat = model(x_t, u_t, x_tp1)
                
                # Compute losses
                loss_recon = mse_loss(x_t, x_t_recon)
                loss_linear = mse_loss(z_tp1, z_tp1_hat)
                loss_pred = mse_loss(x_tp1, x_tp1_hat)
                rho = model.spectral_radius()
                loss_stab = torch.relu(rho - 1.0) ** 2
                
                loss = (cfg.recon_loss_weight * loss_recon +
                       cfg.linear_loss_weight * loss_linear +
                       cfg.pred_loss_weight * loss_pred +
                       cfg.stab_loss_weight * loss_stab)
                
                # Accumulate losses
                val_losses['total'] += loss.item()
                val_losses['recon'] += loss_recon.item()
                val_losses['linear'] += loss_linear.item()
                val_losses['pred'] += loss_pred.item()
                val_losses['stab'] += loss_stab.item()
                n_val_batches += 1
        
        # Average validation losses
        if n_val_batches > 0:
            for key in val_losses:
                val_losses[key] /= n_val_batches
        
        # Update learning rate
        scheduler.step(val_losses['total'])
        
        # TensorBoard logging for epoch
        writer.add_scalar('epoch/train_loss_total', train_losses['total'], epoch)
        writer.add_scalar('epoch/val_loss_total', val_losses['total'], epoch)
        writer.add_scalar('epoch/val_loss_recon', val_losses['recon'], epoch)
        writer.add_scalar('epoch/val_loss_linear', val_losses['linear'], epoch)
        writer.add_scalar('epoch/val_loss_pred', val_losses['pred'], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{cfg.epochs} Summary:")
        print(f"  Train Loss: {train_losses['total']:.6f}")
        print(f"  Val Loss: {val_losses['total']:.6f}")
        print(f"  Val Recon: {val_losses['recon']:.6f}")
        print(f"  Val Linear: {val_losses['linear']:.6f}")
        print(f"  Val Pred: {val_losses['pred']:.6f}")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_model_path = os.path.join(cfg.ckpt_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'config': cfg.__dict__,
                'scaler': {
                    'x_mean': scaler.x_mean.tolist() if scaler.x_mean is not None else None,
                    'x_std': scaler.x_std.tolist() if scaler.x_std is not None else None,
                    'u_mean': scaler.u_mean.tolist() if scaler.u_mean is not None else None,
                    'u_std': scaler.u_std.tolist() if scaler.u_std is not None else None,
                } if cfg.normalize_data else None
            }, best_model_path)
            print(f"  Saved best model to {best_model_path} (val loss: {best_val_loss:.6f})")
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final evaluation on test set:")
    print("="*50)



    all_gt = []
    all_pred = []
    
    model.eval()
    test_losses = {
        'total': 0.0,
        'recon': 0.0,
        'linear': 0.0,
        'pred': 0.0,
        'stab': 0.0
    }
    n_test_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if cfg.use_sequence:
                input_states, input_controls, target_states = batch
                input_states = input_states.to(device)
                input_controls = input_controls.to(device)
                target_states = target_states.to(device)
                
                x_t = input_states[:, -1, :]
                x_tp1 = target_states[:, 0, :]
                u_t = input_controls[:, -1, :]
            else:
                x_t, x_tp1, u_t = batch
                x_t = x_t.to(device)
                x_tp1 = x_tp1.to(device)
                u_t = u_t.to(device)
            
            # Forward pass
            z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat = model(x_t, u_t, x_tp1)

            all_gt.append(x_tp1.cpu())
            all_pred.append(x_tp1_hat.cpu())
            
            # Compute losses
            loss_recon = mse_loss(x_t, x_t_recon)
            loss_linear = mse_loss(z_tp1, z_tp1_hat)
            loss_pred = mse_loss(x_tp1, x_tp1_hat)
            rho = model.spectral_radius()
            loss_stab = torch.relu(rho - 1.0) ** 2
            
            loss = (cfg.recon_loss_weight * loss_recon +
                   cfg.linear_loss_weight * loss_linear +
                   cfg.pred_loss_weight * loss_pred +
                   cfg.stab_loss_weight * loss_stab)
            
            # Accumulate losses
            test_losses['total'] += loss.item()
            test_losses['recon'] += loss_recon.item()
            test_losses['linear'] += loss_linear.item()
            test_losses['pred'] += loss_pred.item()
            test_losses['stab'] += loss_stab.item()
            n_test_batches += 1


    all_gt = torch.cat(all_gt, dim=0).numpy()
    all_pred = torch.cat(all_pred, dim=0).numpy()
    # if cfg.normalize_data:
    #     all_gt = scaler.inverse_transform_x(torch.tensor(all_gt)).numpy()
    #     all_pred = scaler.inverse_transform_x(torch.tensor(all_pred)).numpy()

    state_names = ["x", "y", "yaw", "u", "v", "r"]
    save_dir = os.path.join(cfg.ckpt_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    n_plot = min(1000, len(all_gt))  # 只画前1000个点

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
        print(f"Saved plot for {state_names[i]} comparison.")
        print(all_gt[:n_plot, i])
        print(all_pred[:n_plot, i])

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



    # Average test losses
    if n_test_batches > 0:
        for key in test_losses:
            test_losses[key] /= n_test_batches
    
    print(f"\nTest Results:")
    print(f"  Total Loss: {test_losses['total']:.6f}")
    print(f"  Recon Loss: {test_losses['recon']:.6f}")
    print(f"  Linear Loss: {test_losses['linear']:.6f}")
    print(f"  Pred Loss: {test_losses['pred']:.6f}")
    print(f"  Stab Loss: {test_losses['stab']:.6f}")
    
    # Save final model
    final_model_path = os.path.join(cfg.ckpt_dir, "final_model.pt")
    torch.save({
        'epoch': cfg.epochs,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'test_loss': test_losses['total'],
        'config': cfg.__dict__,
        'scaler': {
            'x_mean': scaler.x_mean.tolist() if scaler.x_mean is not None else None,
            'x_std': scaler.x_std.tolist() if scaler.x_std is not None else None,
            'u_mean': scaler.u_mean.tolist() if scaler.u_mean is not None else None,
            'u_std': scaler.u_std.tolist() if scaler.u_std is not None else None,
        } if cfg.normalize_data else None
    }, final_model_path)
    
    print(f"\nSaved final model to {final_model_path}")
    print(f"Training completed!")
    
    # Close TensorBoard writer
    writer.close()


def export_local_data_to_npz(dataset, filename="local_dataset.npz"):
    """
    遍历数据集，应用坐标转换逻辑，并将结果保存为 npz 文件
    """
    x_t_list = []
    x_tp1_list = []
    u_list = []

    print(f"正在转换并导出数据到 {filename}...")
    
    # 模拟 DataLoader 的批量处理以加快速度
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    with torch.no_grad():
        for x_t_batch, x_tp1_batch, u_batch in loader:
            # 1. 应用坐标转换函数 (假设已定义 batched_transform_to_local_frame)
            x_tp1_local = batched_transform_to_local_frame(x_t_batch, x_tp1_batch)
            
            # 2. 当前状态归零 (局部系原点)
            x_t_local = x_t_batch.clone()
            x_t_local[:, 0:3] = 0.0
            
            x_t_list.append(x_t_local.numpy())
            x_tp1_list.append(x_tp1_local.numpy())
            u_list.append(u_batch.numpy())

    # 合并数据
    x_t_final = np.concatenate(x_t_list, axis=0)
    x_tp1_final = np.concatenate(x_tp1_list, axis=0)
    u_final = np.concatenate(u_list, axis=0)

    # 保存
    np.savez(filename, x_t=x_t_final, x_tp1=x_tp1_final, u=u_final)
    print(f"保存成功！总样本数: {len(x_t_final)}")



def plot_local_transitions(x_tp1_np):
    """
    绘制局部坐标系下的下一时刻位置分布
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x_tp1_np[:, 0], x_tp1_np[:, 1], alpha=0.5, s=2, label='Next State (Local)')
    
    # 绘制车辆当前位置 (原点)
    plt.scatter(0, 0, color='red', marker='X', s=100, label='Current Pos (0,0)')
    
    # 绘制车头指向线条
    plt.arrow(0, 0, 1.0, 0, head_width=0.2, color='black', label='Heading Direction')
    
    plt.xlabel('Local X (Forward) [m]')
    plt.ylabel('Local Y (Left) [m]')
    plt.title('Vehicle State Transitions in Local Frame')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()


def save_and_plot_local_data(dataset, filename="processed_data_local.npz", plot_path="data_visualization.png"):
    """
    1. 遍历数据集并进行坐标转换
    2. 另存为 .npz 文件
    3. 生成轨迹转移图并保存为图片
    """
    x_t_list = []
    x_tp1_list = []
    u_list = []
    
    # 使用 DataLoader 批量处理以提高效率
    # 注意：这里的 dataset 可以是 train_ds, val_ds 或 test_ds
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    print(f"正在处理数据并转换至局部坐标系...")
    with torch.no_grad():
        for x_t_batch, x_tp1_batch, u_batch in loader:
            # 应用之前定义的批量转换函数
            x_tp1_local = batched_transform_to_local_frame(x_t_batch, x_tp1_batch)
            
            # 当前时刻位置归零 (x, y, yaw = 0)
            x_t_local = x_t_batch.clone()
            x_t_local[:, 0:3] = 0.0
            
            x_t_list.append(x_t_local.numpy())
            x_tp1_list.append(x_tp1_local.numpy())
            u_list.append(u_batch.numpy())

    # 合并为大数组
    x_t_final = np.concatenate(x_t_list, axis=0)
    x_tp1_final = np.concatenate(x_tp1_list, axis=0)
    u_final = np.concatenate(u_list, axis=0)

    # --- 1. 保存 .npz 文件 ---
    np.savez(filename, x_t=x_t_final, x_tp1=x_tp1_final, u=u_final)
    print(f"数据已另存为: {filename}")

    # --- 2. 绘图并保存图片 ---
    plt.figure(figsize=(10, 8))
    
    # 绘制下一时刻相对于当前时刻(0,0)的位置分布
    # x_tp1_final[:, 0] 是纵向位移 (Forward)
    # x_tp1_final[:, 1] 是侧向位移 (Left)
    plt.scatter(x_tp1_final[:, 0], x_tp1_final[:, 1], alpha=0.4, s=5, c='blue', label='下一时刻相对位置')
    
    # 标记当前时刻原点
    plt.scatter(0, 0, color='red', marker='X', s=100, label='当前位置 (0,0)')
    
    # 画出车头指向箭头 (x轴正方向)
    plt.arrow(0, 0, 0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k', label='车头指向')

    plt.xlabel('局部 X (前向 / m)')
    plt.ylabel('局部 Y (左向 / m)')
    plt.title('局部坐标系下的状态转移分布 (左正右负)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # 保证横纵比例一致，防止坐标轴畸变
    
    # 保存图片
    plt.savefig(plot_path, dpi=300)
    print(f"可视化图片已保存至: {plot_path}")
    plt.close()



def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Koopman model for horizontal ship motion")
    
    # Data and model
    parser.add_argument("--npz-path", type=str, default="sim.npz", 
                       help="Path to NPZ dataset file")
    parser.add_argument("--model-type", type=str, choices=["horizontal", "deep"], default="horizontal",
                       help="Type of Koopman model to use")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    
    # Model architecture
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--enc-hidden", type=int, nargs="+", default=[64, 64], 
                       help="Encoder hidden layer sizes")
    parser.add_argument("--dec-hidden", type=int, nargs="+", default=[64, 64],
                       help="Decoder hidden layer sizes")
    parser.add_argument("--use-skip", action="store_true", default=True,
                       help="Use skip connection in horizontal model")
    
    # Loss weights
    parser.add_argument("--recon-loss-weight", type=float, default=1.0,
                       help="Weight for reconstruction loss")
    parser.add_argument("--linear-loss-weight", type=float, default=10.0,
                       help="Weight for linear dynamics loss")
    parser.add_argument("--pred-loss-weight", type=float, default=1.0,
                       help="Weight for prediction loss")
    parser.add_argument("--stab-loss-weight", type=float, default=0.1,
                       help="Weight for stability loss")
    
    # Sequence training
    parser.add_argument("--use-sequence", action="store_true",
                       help="Use sequence training instead of single-step")
    parser.add_argument("--seq-length", type=int, default=10,
                       help="Sequence length for training")
    parser.add_argument("--pred-length", type=int, default=1,
                       help="Prediction length for sequence training")
    parser.add_argument("--seq-stride", type=int, default=5,
                       help="Stride for sequence sampling")
    
    # Data handling
    parser.add_argument("--val-fraction", type=float, default=0.15,
                       help="Fraction of data for validation")
    parser.add_argument("--test-fraction", type=float, default=0.15,
                       help="Fraction of data for testing")
    parser.add_argument("--normalize-data", action="store_true", default=True,
                       help="Normalize data to zero mean and unit variance")
    
    # Training control
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use: cpu or cuda")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log-every", type=int, default=100,
                       help="Log training metrics every N steps")
    parser.add_argument("--save-every", type=int, default=2000,
                       help="Save checkpoint every N steps")
    
    # Checkpoint directory
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_horizontal",
                       help="Directory to save checkpoints")
    
    # TensorBoard
    parser.add_argument("--run-name", type=str, default=None,
                       help="Run name for TensorBoard")
    
    args = parser.parse_args()
    
    # Convert list arguments to tuples
    if args.enc_hidden:
        args.enc_hidden = tuple(args.enc_hidden)
    if args.dec_hidden:
        args.dec_hidden = tuple(args.dec_hidden)
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()