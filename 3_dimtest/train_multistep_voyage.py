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
    logger = logging.getLogger("KoopmanTrainer")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, f"train_{timestamp}.log"), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, timestamp

class ExplicitKoopmanDataset(Dataset):
    def __init__(self, npz_path, pred_len=20, stats=None, is_train=False):
        self.segments = np.load(npz_path, allow_pickle=True)['datas']
        self.pred_len = pred_len
        self.indices = [(i, t) for i, seg in enumerate(self.segments) if seg['len'] > pred_len for t in range(0, seg['len'] - pred_len)]
        self.stats = stats if stats is not None else self._compute_local_statistics()
        print(f"Dataset加载: {npz_path} | {len(self.indices)}样本")

    def _get_raw_state(self, seg, t):
        u, v, r = seg['Vel'][0, t], seg['Vel'][1, t], seg['pqr'][0, t]
        return np.array([u, v, r, u*abs(u), v*abs(v), r*abs(r), u*r, v*r], dtype=np.float32)

    def _compute_local_statistics(self):
        local_states, controls = [], []
        for i, t in self.indices:
            seg = self.segments[i]
            local_states.extend([self._get_raw_state(seg, t+j) for j in range(self.pred_len + 1)])
            controls.extend([seg['Thrusters_CMD'][:, t+j] for j in range(self.pred_len)])
        ls, ct = np.array(local_states, dtype=np.float32), np.array(controls, dtype=np.float32)
        return {
            "state_mean": np.mean(ls, axis=0), "state_std": np.std(ls, axis=0) + 1e-6,
            "ctrl_mean": np.mean(ct, axis=0), "ctrl_std": np.std(ct, axis=0) + 1e-6
        }

    def __getitem__(self, index):
        seg, t = self.segments[self.indices[index][0]], self.indices[index][1]
        x_t = (self._get_raw_state(seg, t) - self.stats["state_mean"]) / self.stats["state_std"]
        x_seq = (np.array([self._get_raw_state(seg, t + i + 1) for i in range(self.pred_len)]) - self.stats["state_mean"]) / self.stats["state_std"]
        u_seq = (np.array([seg['Thrusters_CMD'][:, t+i] for i in range(self.pred_len)]) - self.stats["ctrl_mean"]) / self.stats["ctrl_std"]
        return torch.FloatTensor(x_t), torch.FloatTensor(x_seq), torch.FloatTensor(u_seq)

    def __len__(self): return len(self.indices)

def export_params_to_yaml(model, stats, logger, save_path):
    get_mat = lambda m, add_I=False: (m.weight.detach().cpu() + (torch.eye(m.weight.size(0)) if add_I else 0)).numpy().tolist() if m else []
    get_bias = lambda m: m.bias.detach().cpu().numpy().tolist() if m and m.bias is not None else []
    
    yaml_data = {
        "normalization": {
            "state_mean": stats["state_mean"].tolist(), "state_std": stats["state_std"].tolist(),
            "ctrl_mean": stats["ctrl_mean"].tolist(), "ctrl_std": stats["ctrl_std"].tolist()
        },
        "system_matrices": {"A": get_mat(model.A, True), "B": get_mat(model.B, False), "c": get_bias(model.A)}
    }
    with open(save_path, 'w') as f: yaml.dump(yaml_data, f, sort_keys=False, indent=4)
    logger.info(f"参数导出至: {save_path}")

def train(args):
    logger, timestamp = setup_logger(args.log_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_weights = torch.tensor([float(w) for w in args.state_weights.split(',')], device=device)
    train_ds = ExplicitKoopmanDataset(args.train_data, pred_len=args.pred_len, is_train=True)
    val_ds = ExplicitKoopmanDataset(args.val_data, pred_len=args.pred_len, stats=train_ds.stats, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = HorizontalKoopmanModel(input_dim=8, state_dim=3, control_dim=4, latent_dim=32, enc_hidden=[128, 128]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # 稍微温和一点的权重递增
    temporal_weights = torch.tensor([1.0 + 0.1 * step for step in range(args.pred_len)], device=device).view(1, -1, 1)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_pred = 0, 0
        
        for x_t_8d, x_target_seq_8d, u_seq in train_loader:
            x_t_8d, x_target_seq_8d, u_seq = x_t_8d.to(device), x_target_seq_8d.to(device), u_seq.to(device)
            optimizer.zero_grad()
            
            z_current = model.encode(x_t_8d)
            pred_states_3d, pred_latents = [], []
            
            for step in range(args.pred_len):
                z_current = model.latent_step(z_current, u_seq[:, step, :])
                pred_latents.append(z_current)
                pred_states_3d.append(model.reconstruct_state(z_current))
                
            x_pred_seq_3d = torch.stack(pred_states_3d, dim=1)
            pred_latents_stack = torch.stack(pred_latents, dim=1)
            
            B, seq_len, dim = x_target_seq_8d.shape
            target_latents = model.encode(x_target_seq_8d.view(B * seq_len, dim)).view(B, seq_len, -1)
            
            # 【完美简化】：只有物理预测Loss和隐空间对齐Loss，移除破坏阻尼的惯性Loss
            loss_pred = torch.mean(temporal_weights * state_weights * (x_pred_seq_3d - x_target_seq_8d[:, :, :3])**2)
            loss_linear = nn.MSELoss()(pred_latents_stack, target_latents) 
            
            loss = args.w_pred * loss_pred + args.w_linear * loss_linear
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item(); total_pred += loss_pred.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_t_8d, x_target_seq_8d, u_seq in val_loader:
                x_t_8d, x_target_seq_8d, u_seq = x_t_8d.to(device), x_target_seq_8d.to(device), u_seq.to(device)
                z_current = model.encode(x_t_8d)
                preds = []
                for step in range(args.pred_len):
                    z_current = model.latent_step(z_current, u_seq[:, step, :])
                    preds.append(model.reconstruct_state(z_current))
                val_loss += torch.mean(temporal_weights * state_weights * (torch.stack(preds, dim=1) - x_target_seq_8d[:, :, :3])**2).item()
                
        scheduler.step()
        val_loss /= len(val_loader)
        
        logger.info(f"Ep {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} (Pred: {total_pred/len(train_loader):.4f}) | Val: {val_loss:.4f}")

        checkpoint = {'model_state_dict': model.state_dict(), 'stats': train_ds.stats}
        torch.save(checkpoint, os.path.join(args.ckpt_dir, "koopman_latest.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.ckpt_dir, "koopman_best.pth"))

    export_params_to_yaml(model, train_ds.stats, logger, os.path.join(args.ckpt_dir, "koopman_deploy_params.yaml"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--w_pred', type=float, default=10.0)
    parser.add_argument('--w_linear', type=float, default=2.0)
    parser.add_argument('--state_weights', type=str, default='3.0,5.0,5.0')
    train(parser.parse_args())