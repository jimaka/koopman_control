import os
import yaml
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataclasses import dataclass

from koopman import create_koopman_model

# ==========================================
# 1. Config
# ==========================================
@dataclass
class Config:
    npz_path: str = "koopman_dataset_v1.npz"
    device: str = "cuda"
    batch_size: int = 64
    epochs: int = 2
    lr: float = 1e-3
    latent_dim: int = 32

    log_every: int = 50
    save_every: int = 500

    ckpt_dir: str = "checkpoints"
    seed: int = 42


# ==========================================
# 2. Utils
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 3. Dataset（原样保留）
# ==========================================
class PelicanMultiStepDataset(Dataset):
    def __init__(self, npz_path, include_indices=None, pred_len=10, external_stats=None):
        with np.load(npz_path, allow_pickle=True) as data:
            all_segments = data["datas"]

        self._datas = [all_segments[i] for i in include_indices] if include_indices else all_segments
        self.pred_len = pred_len
        self.stats = external_stats if external_stats is not None else self._compute_global_stats()
        self.stats.update(self._compute_normalized_bounds())

        self._indices = []
        for seg_idx, seg in enumerate(self._datas):
            for t in range(seg['len'] - self.pred_len):
                self._indices.append((seg_idx, t))

    def _compute_global_stats(self):
        all_x, all_u = [], []
        for seg in self._datas:
            x = np.vstack([seg['Pos'], seg['Euler'][2:3, :], seg['Vel'], seg['pqr']]).T
            u = seg['Thrusters_CMD'].T
            all_x.append(x); all_u.append(u)
        all_x, all_u = np.vstack(all_x), np.vstack(all_u)
        return {
            "state_mean": np.mean(all_x, axis=0),
            "state_std": np.std(all_x, axis=0) + 1e-6,
            "state_min": np.min(all_x, axis=0),
            "state_max": np.max(all_x, axis=0),
            "ctrl_mean": np.mean(all_u, axis=0),
            "ctrl_std": np.std(all_u, axis=0) + 1e-6,
            "ctrl_min": np.min(all_u, axis=0),
            "ctrl_max": np.max(all_u, axis=0)
        }

    def _compute_normalized_bounds(self):
        return {
            "state_min_norm": (self.stats["state_min"] - self.stats["state_mean"]) / self.stats["state_std"],
            "state_max_norm": (self.stats["state_max"] - self.stats["state_mean"]) / self.stats["state_std"],
            "ctrl_min_norm": (self.stats["ctrl_min"] - self.stats["ctrl_mean"]) / self.stats["ctrl_std"],
            "ctrl_max_norm": (self.stats["ctrl_max"] - self.stats["ctrl_mean"]) / self.stats["ctrl_std"]
        }

    def _norm_s(self, s): return (s - self.stats["state_mean"]) / self.stats["state_std"]
    def _norm_u(self, u): return (u - self.stats["ctrl_mean"]) / self.stats["ctrl_std"]

    def __getitem__(self, index):
        seg_idx, t = self._indices[index]
        seg = self._datas[seg_idx]

        x_t = self._norm_s(np.array([
            seg['Pos'][0,t], seg['Pos'][1,t], seg['Euler'][2,t],
            seg['Vel'][0,t], seg['Vel'][1,t], seg['pqr'][0,t]
        ]))

        x_targets = np.array([
            self._norm_s(np.array([
                seg['Pos'][0,t+i], seg['Pos'][1,t+i], seg['Euler'][2,t+i],
                seg['Vel'][0,t+i], seg['Vel'][1,t+i], seg['pqr'][0,t+i]
            ])) for i in range(1, self.pred_len + 1)
        ])

        u_seq = np.array([
            self._norm_u(seg['Thrusters_CMD'][:, t+i]) for i in range(self.pred_len)
        ])

        return torch.FloatTensor(x_t), torch.FloatTensor(x_targets), torch.FloatTensor(u_seq)

    def __len__(self):
        return len(self._indices)


# ==========================================
# 4. YAML 导出（原样）
# ==========================================
def export_robust_yaml(model, dataset, filename="koopman_config.yaml"):
    model.eval()
    raw_model = model.module if hasattr(model, 'module') else model

    A = raw_model.A.weight.detach().cpu().numpy()
    B = raw_model.B.weight.detach().cpu().numpy()
    C = raw_model.C.weight.detach().cpu().numpy()

    config = {
        "dimensions": {
            "nx": C.shape[0],
            "nu": B.shape[1],
            "nk": A.shape[0]
        },
        "dynamics": {
            "A": A.flatten().tolist(),
            "B": B.flatten().tolist(),
            "C": C.flatten().tolist()
        },
        "normalization": dataset.stats
    }

    with open(filename, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"[YAML] saved: {filename}")


# ==========================================
# 5. 训练主函数（核心升级）
# ==========================================
def train_and_export(args=None):
    cfg = Config()

    if args:
        for k, v in vars(args).items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)

    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", datetime.now().strftime("%Y%m%d-%H%M%S")))

    dataset = PelicanMultiStepDataset(cfg.npz_path, pred_len=10)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = create_koopman_model(
        state_dim=6,
        control_dim=4,
        latent_dim=cfg.latent_dim,
        model_type="horizontal"
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    global_step = 0
    best_loss = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0

        for x_t, x_targets, u_seq in loader:
            x_t, x_targets, u_seq = x_t.to(device), x_targets.to(device), u_seq.to(device)

            optimizer.zero_grad()

            # ===== 编码 =====
            z0 = model.encode(x_t)

            # ===== reconstruction =====
            x_recon = model.decode(z0)
            loss_recon = criterion(x_recon, x_t)

            z_prev = z0
            loss_pred = 0
            loss_linear = 0

            for i in range(10):
                z_next = model.latent_step(z_prev, u_seq[:, i, :])

                # prediction
                x_pred = model.decode(z_next)
                loss_pred += criterion(x_pred, x_targets[:, i, :])

                # Koopman linear constraint
                z_true = model.encode(x_targets[:, i, :]).detach()
                loss_linear += criterion(z_next, z_true)

                # 防止爆炸
                if i > 0:
                    z_next = z_next.detach()

                z_prev = z_next

            loss_pred /= 10
            loss_linear /= 10

            # ===== stability =====
            rho = model.spectral_radius()
            loss_stab = torch.relu(rho - 1.0) ** 2

            # ===== total =====
            loss = (
                1.0 * loss_recon +
                50.0 * loss_linear +
                1.0 * loss_pred +
                1.0 * loss_stab
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # logging
            if global_step % cfg.log_every == 0:
                writer.add_scalar("train/loss_total", loss.item(), global_step)
                writer.add_scalar("train/loss_pred", loss_pred.item(), global_step)
                writer.add_scalar("train/loss_linear", loss_linear.item(), global_step)
                writer.add_scalar("train/rho", rho.item(), global_step)

                print(f"Epoch {epoch+1} Step {global_step} Loss {loss.item():.6f}")

            # checkpoint
            if global_step % cfg.save_every == 0 and global_step > 0:
                path = os.path.join(cfg.ckpt_dir, f"step_{global_step}.pth")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg.__dict__,
                    "stats": dataset.stats
                }, path)
                print(f"[CKPT] {path}")

            global_step += 1

        epoch_loss /= len(loader)
        writer.add_scalar("epoch/loss", epoch_loss, epoch)

        print(f"Epoch {epoch+1} | Loss {epoch_loss:.6f}")

        # best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, "best_model.pth"))

    # final
    torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, "final_model.pth"))
    export_robust_yaml(model, dataset)

    writer.close()
    print("Training finished.")


# ==========================================
# 6. CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--npz-path", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", type=str)

    args = parser.parse_args()
    train_and_export(args)


if __name__ == "__main__":
    main()