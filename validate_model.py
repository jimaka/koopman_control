import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pelican_torch_dataset import PelicanHorizontalTransitionDataset
from koopman import HorizontalKoopmanModel, DeepKoopmanModel


# ==============================
# 1️⃣ 路径（和脚本同目录）
# ==============================

MODEL_PATH = "checkpoints_horizontal/final_model.pt"        # 或 final_model.pt
NPZ_PATH = "sim_1HZ.npz"

DEVICE = "cuda"  # 如有GPU可改为 "cuda"


# ==============================
# 2️⃣ 加载 checkpoint
# ==============================

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
cfg_dict = checkpoint["config"]
scaler_dict = checkpoint.get("scaler", None)

print("Loaded model config:")
for k, v in cfg_dict.items():
    print(f"{k}: {v}")


# ==============================
# 3️⃣ 恢复 scaler
# ==============================

class StandardScaler:
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.u_mean = None
        self.u_std = None

    def transform_x(self, x):
        if self.x_mean is None:
            return x
        x_mean = torch.tensor(self.x_mean, device=x.device, dtype=x.dtype)
        x_std = torch.tensor(self.x_std, device=x.device, dtype=x.dtype)
        return (x - x_mean) / x_std

    def transform_u(self, u):
        if self.u_mean is None:
            return u
        u_mean = torch.tensor(self.u_mean, device=u.device, dtype=u.dtype)
        u_std = torch.tensor(self.u_std, device=u.device, dtype=u.dtype)
        return (u - u_mean) / u_std

    def inverse_transform_x(self, x):
        if self.x_mean is None:
            return x
        x_mean = torch.tensor(self.x_mean, device=x.device, dtype=x.dtype)
        x_std = torch.tensor(self.x_std, device=x.device, dtype=x.dtype)
        return x * x_std + x_mean


scaler = StandardScaler()

if scaler_dict is not None:
    scaler.x_mean = np.array(scaler_dict["x_mean"])
    scaler.x_std = np.array(scaler_dict["x_std"])
    scaler.u_mean = np.array(scaler_dict["u_mean"])
    scaler.u_std = np.array(scaler_dict["u_std"])

    print("Scaler restored.")
else:
    print("No scaler found.")


# ==============================
# 4️⃣ 构建模型
# ==============================

if cfg_dict["model_type"] == "horizontal":
    model = HorizontalKoopmanModel(
        state_dim=6,
        control_dim=4,
        latent_dim=cfg_dict["latent_dim"],
        enc_hidden=list(cfg_dict["enc_hidden"]),
        dec_hidden=list(cfg_dict["dec_hidden"]),
        use_skip=cfg_dict["use_skip"]
    )
else:
    model = DeepKoopmanModel(
        state_dim=6,
        control_dim=4,
        latent_dim=cfg_dict["latent_dim"],
        enc_hidden=list(cfg_dict["enc_hidden"]),
        dec_hidden=list(cfg_dict["dec_hidden"])
    )

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ==============================
# 5️⃣ 加载数据（完整数据作为测试）
# ==============================

dataset = PelicanHorizontalTransitionDataset(
    NPZ_PATH,
    return_flight_index=False,
    use_normalized=False
)

print(f"Dataset size: {len(dataset)}")


# ==============================
# 6️⃣ 推理
# ==============================

all_gt = []
all_pred = []

mse_loss = nn.MSELoss()

with torch.no_grad():
    for i in range(len(dataset)):

        x_t, x_tp1, u_t = dataset[i]

        x_t = x_t.unsqueeze(0).to(DEVICE)
        x_tp1 = x_tp1.unsqueeze(0).to(DEVICE)
        u_t = u_t.unsqueeze(0).to(DEVICE)

        if scaler_dict is not None:
            x_t = scaler.transform_x(x_t)
            x_tp1 = scaler.transform_x(x_tp1)
            u_t = scaler.transform_u(u_t)

        z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat = model(x_t, u_t, x_tp1)

        # if scaler_dict is not None:
        #     x_tp1_hat = scaler.inverse_transform_x(x_tp1_hat)
        #     x_tp1 = scaler.inverse_transform_x(x_tp1)

        all_gt.append(x_tp1.cpu())
        all_pred.append(x_tp1_hat.cpu())


all_gt = torch.cat(all_gt, dim=0).numpy()
all_pred = torch.cat(all_pred, dim=0).numpy()

all_err = all_pred - all_gt


# ==============================
# 7️⃣ 误差计算
# ==============================

mse = np.mean((all_gt - all_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(all_gt - all_pred))

rmse_per_dim = np.sqrt(np.mean((all_gt - all_pred) ** 2, axis=0))

print("\n==============================")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")
print("RMSE per dimension:")
print(rmse_per_dim)
print("==============================\n")


# ==============================
# 8️⃣ 画图
# ==============================

state_names = ["x", "y", "yaw", "u", "v", "r"]
save_dir = "validation_plots"
os.makedirs(save_dir, exist_ok=True)

n_plot = min(1000, len(all_gt))

for i in range(all_gt.shape[1]):

    plt.figure(figsize=(10,4))

    plt.plot(all_gt[:n_plot, i], label="Ground Truth")
    plt.plot(all_pred[:n_plot, i], label="Prediction")

    plt.title(f"Validation - {state_names[i]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{state_names[i]}.png"))

    
    plt.close()

for i in range(all_err.shape[1]):
    plt.figure(figsize=(10,4))
    plt.plot(all_err[:, i])
    plt.title(f"Prediction Error (Normalized) - {state_names[i]}")
    plt.xlabel("Time step")
    plt.ylabel("Pred - GT")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"error_{state_names[i]}.png"))
    plt.close()

print("Plots saved to validation_plots/")
