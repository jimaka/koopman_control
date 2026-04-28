import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from koopman import create_koopman_model
from train_koopman import StandardScaler, batched_transform_to_local_frame


# ===============================
# 配置
# ===============================
CKPT_PATH = "checkpoints_horizontal/best_model.pt"
NPZ_PATH = "sim_0.5HZ.npz"
DEVICE = "cpu"
PLOT_LEN = 3000


# ===============================
# 1. 加载模型
# ===============================
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
cfg = ckpt["config"]

model = create_koopman_model(
    model_type=cfg["model_type"],
    state_dim=6,
    control_dim=4,
    latent_dim=cfg["latent_dim"],
    enc_hidden=list(cfg["enc_hidden"]),
    dec_hidden=list(cfg["dec_hidden"]),
    use_skip=cfg.get("use_skip", True),
)

model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

print("✅ Model loaded")


# ===============================
# 2. 加载 scaler
# ===============================
scaler = StandardScaler()

if ckpt["scaler"] is not None:
    scaler.x_mean = np.array(ckpt["scaler"]["x_mean"])
    scaler.x_std  = np.array(ckpt["scaler"]["x_std"])
    scaler.u_mean = np.array(ckpt["scaler"]["u_mean"])
    scaler.u_std  = np.array(ckpt["scaler"]["u_std"])
    print("✅ Scaler loaded from checkpoint")
    print(f"  x_mean: {scaler.x_mean}, x_std: {scaler.x_std}")
    print(f"  u_mean: {scaler.u_mean}, u_std: {scaler.u_std}")

print("✅ Scaler loaded")


# ===============================
# 3. 解析 npz（适配你的数据）
# ===============================
data = np.load(NPZ_PATH, allow_pickle=True)

if "datas" not in data:
    raise ValueError("❌ 当前npz不是bag转换格式")

datas = data["datas"]
d = datas[0]

# (D,N) → (N,D)
pos = d["Pos"].T
vel = d["Vel"].T
yaw = d["Euler"][2].reshape(-1,1)
r   = d["pqr"].T
u   = d["Thrusters_CMD"].T

# 拼状态: [x, y, yaw, u, v, r]
x = np.concatenate([pos, yaw, vel, r], axis=1)

# transition
x_t   = x[:-1]
x_tp1 = x[1:]
u     = u[:-1]

print("✅ Data processed:", x_t.shape)

# 转 torch
x_t   = torch.tensor(x_t, dtype=torch.float32)
x_tp1 = torch.tensor(x_tp1, dtype=torch.float32)
u     = torch.tensor(u, dtype=torch.float32)


# ===============================
# 4. 坐标转换（必须和训练一致）
# ===============================
x_tp1_local = batched_transform_to_local_frame(x_t, x_tp1)

x_t_local = x_t.clone()
x_t_local[:, 0:3] = 0.0


# ===============================
# 5. 标准化
# ===============================
x_t_norm   = scaler.transform_x(x_t_local)
x_tp1_norm = scaler.transform_x(x_tp1_local)
u_norm     = scaler.transform_u(u)


# ===============================
# 6. 单步预测
# ===============================
with torch.no_grad():
    _, _, _, x_tp1_hat = model(x_t_norm.to(DEVICE), u_norm.to(DEVICE))

# x_pred = scaler.inverse_transform_x(x_tp1_hat.cpu())
x_pred = x_tp1_hat.cpu()
x_gt   = x_tp1_norm

x_pred = scaler.inverse_transform_x(x_tp1_hat.cpu())
x_gt   = scaler.inverse_transform_x(x_tp1_norm)


# ===============================
# 7. 误差计算
# ===============================
error = x_pred - x_gt
rmse = torch.sqrt((error ** 2).mean(dim=0))

state_names = ["x", "y", "yaw", "u", "v", "r"]

print("\n===== RMSE (one-step) =====")
for i in range(6):
    print(f"{state_names[i]}: {rmse[i].item():.6f}")


# ===============================
# 8. 画图
# ===============================
save_dir = "eval_plots"
os.makedirs(save_dir, exist_ok=True)

n_plot = min(PLOT_LEN, x_gt.shape[0])

for i in range(6):
    plt.figure(figsize=(10,4))

    plt.plot(x_gt[:n_plot, i], label="GT")
    plt.plot(x_pred[:n_plot, i], label="Pred")

    plt.title(f"{state_names[i]} (local frame)")
    plt.legend()
    plt.grid()

    plt.savefig(f"{save_dir}/{state_names[i]}_compare.png")
    plt.close()


# ===============================
# 9. 误差图
# ===============================
abs_error = torch.abs(error)

for i in range(6):
    plt.figure(figsize=(10,4))

    plt.plot(abs_error[:n_plot, i])
    plt.title(f"{state_names[i]} error")

    plt.grid()
    plt.savefig(f"{save_dir}/{state_names[i]}_error.png")
    plt.close()

print("\n✅ Done. Results saved to:", save_dir)