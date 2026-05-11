import os
import math
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

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("KoopmanTrainer")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, timestamp

def get_gpu_power_bar(bar_length=10):
    if not torch.cuda.is_available(): return "[N/A (CPU)]"
    try:
        cmd = "nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader,nounits"
        result = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split('\n')[0]
        if "N/A" in result or "Not Supported" in result: return "[Pwr: N/A]"
        draw, limit = map(float, result.split(','))
        percent = (draw / limit) * 100 if limit > 0 else 0
        filled = int((percent / 100) * bar_length)
        bar = '|' * filled + ' ' * (bar_length - filled)
        return f"[{bar}] {percent:4.1f}%"
    except:
        return "[Pwr: Error]"

class ExplicitKoopmanDataset(Dataset):
    def __init__(self, npz_path, pred_len=20, stats=None, is_train=False, logger=None):
        super().__init__()
        self.logger = logger
        self.segments = np.load(npz_path, allow_pickle=True)['datas']
        self.pred_len = pred_len
        self.indices = []
        
        for local_seg_idx, seg in enumerate(self.segments):
            if seg['len'] > pred_len:
                for t in range(0, seg['len'] - pred_len):
                    self.indices.append((local_seg_idx, t))
                    
        self.stats = stats if stats is not None else self._compute_local_statistics()
            
        mode_str = "TRAIN" if is_train else "EVAL"
        msg = f"Dataset [{mode_str}] 加载: {npz_path} | {len(self.segments)}段 | {len(self.indices)}样本"
        if self.logger: self.logger.info(msg) 
        else: print(msg)

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

    def _compute_local_statistics(self):
        if self.logger: self.logger.info("正在计算 8 维特征全局统计量...")
        local_states, controls = [], []
        for local_seg_idx, t in self.indices:
            seg = self.segments[local_seg_idx]
            x_t_raw = self._get_raw_state(seg, t)
            x_seq_raw = np.array([self._get_raw_state(seg, t + i) for i in range(1, self.pred_len + 1)])
            
            local_states.append(x_t_raw)
            local_states.extend(x_seq_raw)
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
        
        x_t = self._get_raw_state(seg, t)
        x_seq = np.array([self._get_raw_state(seg, t + i) for i in range(1, self.pred_len + 1)])
        
        x_t_norm = (x_t - self.stats["state_mean"]) / self.stats["state_std"]
        x_seq_norm = (x_seq - self.stats["state_mean"]) / self.stats["state_std"]
        u_seq_norm = np.array([(seg['Thrusters_CMD'][:, t+i] - self.stats["ctrl_mean"])/self.stats["ctrl_std"] for i in range(self.pred_len)])
        
        return torch.FloatTensor(x_t_norm), torch.FloatTensor(x_seq_norm), torch.FloatTensor(u_seq_norm)

    def __len__(self): 
        return len(self.indices)


def export_params_to_yaml(model, stats, logger, save_path):
    def extract_matrix(matrix_attr, is_A_matrix=False):
        if matrix_attr is None: return []
        mat_tensor = getattr(matrix_attr, 'weight', getattr(matrix_attr, 'detach', lambda: None)())
        if callable(mat_tensor): mat_tensor = mat_tensor()
        if mat_tensor is not None:
            mat_tensor = mat_tensor.detach().cpu()
            if is_A_matrix: mat_tensor = mat_tensor + torch.eye(mat_tensor.size(0))
            return mat_tensor.numpy().tolist()
        return []

    def extract_bias(matrix_attr):
        if matrix_attr is None or getattr(matrix_attr, 'bias', None) is None: return []
        return matrix_attr.bias.detach().cpu().numpy().tolist()

    yaml_data = {
        "normalization": {
            "state_mean": stats["state_mean"].tolist(), "state_variance": (stats["state_std"]**2).tolist(), "state_std": stats["state_std"].tolist(),
            "ctrl_mean": stats["ctrl_mean"].tolist(), "ctrl_variance": (stats["ctrl_std"]**2).tolist(), "ctrl_std": stats["ctrl_std"].tolist()
        },
        "bounds": {
            "state_min": stats["state_min"].tolist(), "state_max": stats["state_max"].tolist(),
            "ctrl_min": stats["ctrl_min"].tolist(), "ctrl_max": stats["ctrl_max"].tolist()
        },
        "system_matrices": {
            "A": extract_matrix(getattr(model, 'A', None), True), 
            "B": extract_matrix(getattr(model, 'B', None), False),
            "c": extract_bias(getattr(model, 'A', None)) # 导出常数偏置项
        }
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False, indent=4)
    logger.info(f">>> [完美导出] 高维 OSQP 矩阵参数（包含Bias）已导出至: {save_path}")


def train(args):
    logger, timestamp = setup_logger(args.log_dir)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f'tensorboard_{timestamp}'))
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_weights_list = [float(w) for w in args.state_weights.split(',')]
    assert len(state_weights_list) == 3, "目标状态维度必须为 3 [u, v, r]"
    state_weights = torch.tensor(state_weights_list, device=device)
    
    train_ds = ExplicitKoopmanDataset(args.train_data, pred_len=args.pred_len, is_train=True, logger=logger)
    val_ds = ExplicitKoopmanDataset(args.val_data, pred_len=args.pred_len, stats=train_ds.stats, is_train=False, logger=logger)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=args.prefetch, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 【优化】：使用降维模型 (latent_dim=32, enc_hidden=[128,128])
    model = HorizontalKoopmanModel(input_dim=8, state_dim=3, control_dim=4, latent_dim=32, enc_hidden=[128, 128], dec_hidden=[128, 128], use_skip=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    mse_loss = nn.MSELoss()
    
    # 强制指数递增的时间权重
    temporal_weights_list = [math.exp(0.15 * step) for step in range(args.pred_len)]
    temporal_weights = torch.tensor(temporal_weights_list, device=device).view(1, -1, 1)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {'total': 0.0, 'pred': 0.0, 'acc': 0.0, 'recon': 0.0, 'linear': 0.0, 'inertia': 0.0}
        
        for x_t_8d, x_target_seq_8d, u_seq in train_loader:
            x_t_8d, x_target_seq_8d, u_seq = x_t_8d.to(device), x_target_seq_8d.to(device), u_seq.to(device)
            optimizer.zero_grad()
            
            z_current = model.encode(x_t_8d)
            x_t_recon_3d = model.reconstruct_state(z_current)
            pred_states_3d, pred_latents = [], []
            
            for step in range(args.pred_len):
                z_next = model.latent_step(z_current, u_seq[:, step, :])
                pred_latents.append(z_next)
                pred_states_3d.append(model.reconstruct_state(z_next))
                z_current = z_next
                
            x_pred_seq_3d = torch.stack(pred_states_3d, dim=1)
            pred_latents_stack = torch.stack(pred_latents, dim=1)
            
            B, seq_len, dim = x_target_seq_8d.shape
            target_latents = model.encode(x_target_seq_8d.view(B * seq_len, dim)).view(B, seq_len, -1)
            
            x_target_seq_3d = x_target_seq_8d[:, :, :3]
            x_t_target_3d = x_t_8d[:, :3]
            
            # 【优化】：引入全序列自编码重构惩罚
            target_recon_3d = model.reconstruct_state(target_latents.view(B * seq_len, -1)).view(B, seq_len, 3)
            loss_recon_seq = torch.mean(state_weights * (target_recon_3d - x_target_seq_3d)**2)
            loss_recon_t0 = torch.mean(state_weights * (x_t_recon_3d - x_t_target_3d)**2)
            
            loss_recon = loss_recon_t0 + 0.5 * loss_recon_seq
            
            loss_pred = torch.mean(temporal_weights * state_weights * (x_pred_seq_3d - x_target_seq_3d)**2)
            loss_linear = mse_loss(pred_latents_stack, target_latents) 
            loss_acc = torch.mean(((x_pred_seq_3d[:, 1:, :] - x_pred_seq_3d[:, :-1, :]) - (x_target_seq_3d[:, 1:, :] - x_target_seq_3d[:, :-1, :]))**2)
            loss_inertia = torch.mean(model.A.weight ** 2)

            # 惯性权重微调为 0.1，允许矩阵适度学习阻尼
            loss = (args.w_pred * loss_pred + 
                    args.w_acc * loss_acc + 
                    args.w_recon * loss_recon + 
                    args.w_linear * loss_linear + 
                    0.1 * loss_inertia)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses['total'] += loss.item()
            epoch_losses['pred'] += loss_pred.item()
            epoch_losses['acc'] += loss_acc.item()
            epoch_losses['recon'] += loss_recon.item()
            epoch_losses['linear'] += loss_linear.item()
            epoch_losses['inertia'] += loss_inertia.item()
            
        num_batches = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            tb_writer.add_scalar(f'Train/{k.capitalize()}_Loss', epoch_losses[k], epoch)
            
        model.eval()
        total_loss_val = 0.0
        with torch.no_grad():
            for x_t_8d, x_target_seq_8d, u_seq in val_loader:
                x_t_8d, x_target_seq_8d, u_seq = x_t_8d.to(device), x_target_seq_8d.to(device), u_seq.to(device)
                z_current = model.encode(x_t_8d)
                pred_states_3d = []
                for step in range(args.pred_len):
                    z_current = model.latent_step(z_current, u_seq[:, step, :])
                    pred_states_3d.append(model.reconstruct_state(z_current))
                val_loss_weighted = torch.mean(temporal_weights * state_weights * (torch.stack(pred_states_3d, dim=1) - x_target_seq_8d[:, :, :3])**2)
                total_loss_val += val_loss_weighted.item()
                
        scheduler.step()
        avg_val_loss = total_loss_val / len(val_loader)
        tb_writer.add_scalar('Val/Total_Loss', avg_val_loss, epoch)
        
        log_msg = (
            f"Ep [{epoch+1:03d}/{args.epochs}] LR:{optimizer.param_groups[0]['lr']:.6f} | "
            f"Loss:{epoch_losses['total']:.4f} (Pr:{epoch_losses['pred']:.4f}, Ac:{epoch_losses['acc']:.4f}, "
            f"Rc:{epoch_losses['recon']:.4f}, Lr:{epoch_losses['linear']:.4f}, Iner:{epoch_losses['inertia']:.4f}) | "
            f"Val:{avg_val_loss:.4f} | Pwr: {get_gpu_power_bar()}"
        )
        logger.info(log_msg)

        checkpoint = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'stats': train_ds.stats}
        torch.save(checkpoint, os.path.join(args.ckpt_dir, "koopman_latest.pth"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(args.ckpt_dir, "koopman_best.pth"))

    logger.info("\n>>> 训练完成！")
    export_params_to_yaml(model, train_ds.stats, logger, save_path=os.path.join(args.ckpt_dir, "koopman_deploy_params.yaml"))
    tb_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Koopman 8D-EDMD Multi-step Training")
    parser.add_argument('--train_data', type=str, default='../koopman_train.npz')
    parser.add_argument('--val_data', type=str, default='../koopman_val.npz')
    parser.add_argument('--ckpt_dir', type=str, default='../checkpoints')
    parser.add_argument('--log_dir', type=str, default='../logs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=25)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--pred_len', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--prefetch', type=int, default=40)
    
    parser.add_argument('--w_pred', type=float, default=10.0)
    parser.add_argument('--w_acc', type=float, default=5.0)
    parser.add_argument('--w_recon', type=float, default=1.0)
    parser.add_argument('--w_linear', type=float, default=1.0)
    
    parser.add_argument('--state_weights', type=str, default='3.0,5.0,5.0', help='各状态维度权重[u, v, r]')

    args = parser.parse_args()
    train(args)