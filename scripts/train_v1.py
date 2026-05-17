import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from koopman.paths import setup_repo

setup_repo()

import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import subprocess

from koopman import HorizontalKoopmanModel

# ==========================================
# 0. 辅助工具
# ==========================================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("KoopmanTrainer")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, f"train_{timestamp}.log"), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter); ch.setFormatter(formatter)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger, timestamp

def get_gpu_power_bar(bar_length=10):
    if not torch.cuda.is_available(): return "[N/A]"
    try:
        cmd = "nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader,nounits"
        draw, limit = map(float, subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split('\n')[0].split(','))
        percent = (draw / limit) * 100 if limit > 0 else 0
        return f"[{'|' * int((percent / 100) * bar_length):<10}] {percent:4.1f}%"
    except: return "[Pwr Error]"

# ==========================================
# 1. 高速数据集加载器
# ==========================================
class ExplicitKoopmanDataset(Dataset):
    def __init__(self, npz_path, pred_len=20, stats=None, is_train=False, logger=None):
        super().__init__()
        self.segments = np.load(npz_path, allow_pickle=True)['datas']
        self.pred_len = pred_len
        self.indices = [(i, t) for i, seg in enumerate(self.segments) if seg['len'] > pred_len for t in range(0, seg['len'] - pred_len)]
        self.stats = stats if stats is not None else self._compute_local_statistics()
        if logger: logger.info(f"Dataset [{'TRAIN' if is_train else 'EVAL'}] 加载完毕 | {len(self.indices)} 样本.")

    def _get_raw_state(self, seg, t):
        return np.array([seg['Pos'][0, t], seg['Pos'][1, t], seg['Euler'][2, t],
                         seg['Vel'][0, t], seg['Vel'][1, t], seg['pqr'][0, t]], dtype=np.float32)

    def _compute_local_statistics(self):
        local_states, controls = [], []
        for local_seg_idx, t in self.indices:
            seg = self.segments[local_seg_idx]
            local_states.extend([self._get_raw_state(seg, t + i) for i in range(self.pred_len + 1)])
            for i in range(self.pred_len): controls.append(seg['Thrusters_CMD'][:, t + i])
            
        local_states, controls = np.array(local_states, dtype=np.float32), np.array(controls, dtype=np.float32)
        return {
            "state_mean": np.mean(local_states, axis=0), "state_std": np.std(local_states, axis=0) + 1e-6,
            "ctrl_mean": np.mean(controls, axis=0), "ctrl_std": np.std(controls, axis=0) + 1e-6
        }

    def __getitem__(self, index):
        seg = self.segments[self.indices[index][0]]
        t = self.indices[index][1]
        
        x_raw_seq = np.array([self._get_raw_state(seg, t + i) for i in range(self.pred_len + 1)])
        x_norm_seq = (x_raw_seq - self.stats["state_mean"]) / self.stats["state_std"]
        
        x_t_norm = x_norm_seq[0]
        x_seq_norm = x_norm_seq[1:]
        
        cmd_seq = seg['Thrusters_CMD'][:, t : t + self.pred_len].T
        u_seq_norm = (cmd_seq - self.stats["ctrl_mean"]) / self.stats["ctrl_std"]
        
        return torch.FloatTensor(x_t_norm), torch.FloatTensor(x_seq_norm), torch.FloatTensor(u_seq_norm)

    def __len__(self): return len(self.indices)

def export_params_to_yaml(model, stats, logger, save_path):
    def extract_matrix(matrix_attr, is_A_matrix=False):
        if matrix_attr is None: return []
        mat_tensor = getattr(matrix_attr, 'weight', getattr(matrix_attr, 'detach', lambda: None)())
        if mat_tensor is not None:
            mat_tensor = mat_tensor.detach().cpu()
            if is_A_matrix: mat_tensor = mat_tensor + torch.eye(mat_tensor.size(0))
            return mat_tensor.numpy().tolist()
        return []
        
    A_bias = model.A.bias.detach().cpu().numpy().tolist() if getattr(model.A, 'bias', None) is not None else []
    
    yaml_data = {
        "normalization": {
            "dyn_mean": stats["state_mean"][3:6].tolist(), "dyn_std": stats["state_std"][3:6].tolist(),
            "ctrl_mean": stats["ctrl_mean"].tolist(), "ctrl_std": stats["ctrl_std"].tolist()
        },
        "system_matrices": {"A_weight": extract_matrix(model.A, True), "A_bias": A_bias, "B": extract_matrix(model.B, False)},
        "info": "Latent z structure: [u, v, r, u|u|, v|v|, r|r|, vr, ur, h_1 ... h_24]"
    }
    with open(save_path, 'w', encoding='utf-8') as f: yaml.dump(yaml_data, f, indent=4)

# ==========================================
# 2. 训练核心
# ==========================================
def train(args):
    logger, timestamp = setup_logger(args.log_dir)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f'tensorboard_{timestamp}'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds = ExplicitKoopmanDataset(args.train_data, pred_len=args.pred_len, is_train=True, logger=logger)
    val_ds = ExplicitKoopmanDataset(args.val_data, pred_len=args.pred_len, stats=train_ds.stats, is_train=False, logger=logger)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=args.prefetch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 【加载带物理字典的新架构】
    model = HorizontalKoopmanModel(state_dim=3, hidden_dim=24).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    mse_loss = nn.MSELoss()
    
    stat_mean_dyn = torch.tensor(train_ds.stats['state_mean'][3:6], device=device)
    stat_std_dyn = torch.tensor(train_ds.stats['state_std'][3:6], device=device)
    internal_dt = 0.1  
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {'total': 0.0, 'phy_acc': 0.0, 'linear': 0.0, 'stab': 0.0}
        
        for x_t_full, x_target_seq_full, u_seq in train_loader:
            x_t_full, x_target_seq_full, u_seq = x_t_full.to(device), x_target_seq_full.to(device), u_seq.to(device)
            optimizer.zero_grad()
            
            dyn_t = x_t_full[:, 3:6]
            dyn_target_seq = x_target_seq_full[:, :, 3:6]
            
            z_current = model.encode(dyn_t)
            pred_dyn_states, pred_latents = [], []
            
            for step in range(args.pred_len):
                z_current = model.latent_step(z_current, u_seq[:, step, :])
                pred_latents.append(z_current)
                pred_dyn_states.append(model.reconstruct_state(z_current))
                
            pred_seq = torch.stack(pred_dyn_states, dim=1)
            pred_latents_stack = torch.stack(pred_latents, dim=1)
            
            B, seq_len, _ = dyn_target_seq.shape
            target_latents = model.encode(dyn_target_seq.view(B * seq_len, 3)).view(B, seq_len, -1)
            
            pred_full_seq = torch.cat([dyn_t.unsqueeze(1), pred_seq], dim=1)
            target_full_seq = torch.cat([dyn_t.unsqueeze(1), dyn_target_seq], dim=1)
            
            # 【物理加速度误差核心】
            pred_phys = pred_full_seq * stat_std_dyn + stat_mean_dyn
            target_phys = target_full_seq * stat_std_dyn + stat_mean_dyn
            pred_acc_phys = (pred_phys[:, 1:, :] - pred_phys[:, :-1, :]) / internal_dt
            target_acc_phys = (target_phys[:, 1:, :] - target_phys[:, :-1, :]) / internal_dt
            loss_phy_acc = mse_loss(pred_acc_phys, target_acc_phys)
            
            # 【隐空间线性与稳定约束】
            loss_linear = mse_loss(pred_latents_stack, target_latents) 
            loss_stab = torch.relu(model.spectral_radius() - 1.01) ** 2

            loss = (args.w_acc * loss_phy_acc + 
                    args.w_linear * loss_linear + 
                    args.w_stab * loss_stab)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses['total'] += loss.item(); epoch_losses['phy_acc'] += loss_phy_acc.item()
            epoch_losses['linear'] += loss_linear.item(); epoch_losses['stab'] += loss_stab.item()
            
        for k in epoch_losses: 
            epoch_losses[k] /= len(train_loader)
            tb_writer.add_scalar(f'Train/{k.capitalize()}_Loss', epoch_losses[k], epoch)
            
        gpu_power_str, gpu_mem_str = get_gpu_power_bar(), "[N/A]"
        if torch.cuda.is_available():
            mem_percent = (torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(device).total_memory) * 100
            gpu_mem_str = f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.0f}MB ({mem_percent:.1f}%)"

        model.eval()
        total_loss_val = 0.0
        with torch.no_grad():
            for x_t_full, x_target_seq_full, u_seq in val_loader:
                dyn_t, dyn_target_seq, u_seq = x_t_full[:, 3:6].to(device), x_target_seq_full[:, :, 3:6].to(device), u_seq.to(device)
                z_current = model.encode(dyn_t)
                pred_states = []
                for step in range(args.pred_len):
                    z_current = model.latent_step(z_current, u_seq[:, step, :])
                    pred_states.append(model.reconstruct_state(z_current))
                
                pred_seq = torch.stack(pred_states, dim=1)
                pred_full_seq = torch.cat([dyn_t.unsqueeze(1), pred_seq], dim=1)
                target_full_seq = torch.cat([dyn_t.unsqueeze(1), dyn_target_seq], dim=1)
                
                val_pred_acc = ((pred_full_seq[:, 1:, :] * stat_std_dyn) - (pred_full_seq[:, :-1, :] * stat_std_dyn)) / internal_dt
                val_target_acc = ((target_full_seq[:, 1:, :] * stat_std_dyn) - (target_full_seq[:, :-1, :] * stat_std_dyn)) / internal_dt
                total_loss_val += mse_loss(val_pred_acc, val_target_acc).item()
                
        scheduler.step()
        avg_val_loss = total_loss_val / len(val_loader)
        
        log_msg = (
            f"Epoch [{epoch+1:03d}/{args.epochs}] | LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Loss: {epoch_losses['total']:.4f} (PhyAcc:{epoch_losses['phy_acc']:.4f}, Lin:{epoch_losses['linear']:.4f}) | "
            f"Val Acc Loss: {avg_val_loss:.4f} | Mem: {gpu_mem_str} | Pwr: {gpu_power_str}"
        )
        logger.info(log_msg)

        checkpoint = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'stats': train_ds.stats}
        torch.save(checkpoint, os.path.join(args.ckpt_dir, "koopman_latest.pth"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(args.ckpt_dir, "koopman_best.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from koopman import paths as P

    parser.add_argument('--train_data', type=str, default=str(P.TRAIN_MERGED))
    parser.add_argument('--val_data', type=str, default=str(P.VAL))
    parser.add_argument('--ckpt_dir', type=str, default=str(P.CKPT_DIR))
    parser.add_argument('--log_dir', type=str, default=str(P.LOG_DIR))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--pred_len', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--prefetch', type=int, default=40)
    
    parser.add_argument('--w_acc', type=float, default=100.0)
    parser.add_argument('--w_linear', type=float, default=15.0)
    parser.add_argument('--w_stab', type=float, default=0.1)
    args = parser.parse_args()
    train(args)