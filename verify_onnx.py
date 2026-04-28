import torch
import numpy as np
import onnxruntime as ort

from koopman import create_koopman_model


# ===============================
# 1️⃣ 加载 PyTorch 模型
# ===============================
def load_model(ckpt_path, device="cpu"):
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint["config"]

    model = create_koopman_model(
        model_type=cfg["model_type"],
        state_dim=6,
        control_dim=4,
        latent_dim=cfg["latent_dim"],
        enc_hidden=list(cfg["enc_hidden"]),
        dec_hidden=list(cfg["dec_hidden"]),
        use_skip=cfg.get("use_skip", True),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ===============================
# 2️⃣ ONNX 推理
# ===============================
def run_onnx(onnx_path, x, u):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    inputs = {
        "x": x.astype(np.float32),
        "u": u.astype(np.float32),
    }

    outputs = sess.run(None, inputs)
    return outputs[0]


# ===============================
# 3️⃣ 验证 full_step.onnx
# ===============================
def verify_full_model():
    device = "cpu"

    model = load_model("checkpoints_horizontal/best_model.pt", device)

    # 随机输入
    x = torch.randn(8, 6)
    u = torch.randn(8, 4)

    # PyTorch 输出
    with torch.no_grad():
        z = model.encode(x)
        z_next = model.latent_step(z, u)
        x_next_pt = model.reconstruct_state(z_next)

    x_np = x.numpy()
    u_np = u.numpy()

    # ONNX 输出
    x_next_onnx = run_onnx("onnx_export/full_step.onnx", x_np, u_np)

    # 误差
    diff = x_next_pt.numpy() - x_next_onnx

    print("===================================")
    print("Max error:", np.abs(diff).max())
    print("Mean error:", np.abs(diff).mean())
    print("===================================")


if __name__ == "__main__":
    verify_full_model()