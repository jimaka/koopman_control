import torch
import numpy as np
import onnxruntime as ort
import os
import matplotlib.pyplot as plt

# 关闭中文配置，适配英文显示
plt.rcParams['axes.unicode_minus'] = False  # Solve negative sign display problem

from koopman import create_koopman_model


# ===============================
# 1️⃣ Load PyTorch Model (with exception handling + log)
# ===============================
def load_model(ckpt_path, device="cpu"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model file not found: {ckpt_path}")
    
    print(f"🔄 Loading model: {ckpt_path} (device: {device})")
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint.get("config", checkpoint.get("model_config", {}))
    if not cfg:
        raise ValueError("No model config found in checkpoint (config/model_config)")

    model = create_koopman_model(
        model_type=cfg.get("model_type", "horizontal"),
        state_dim=6,
        control_dim=4,
        latent_dim=cfg["latent_dim"],
        enc_hidden=list(cfg.get("enc_hidden", [64, 64])),
        dec_hidden=list(cfg.get("dec_hidden", [64, 64])),
        use_skip=cfg.get("use_skip", True),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("✅ PyTorch model loaded successfully (eval mode)")
    return model


# ===============================
# 2️⃣ ONNX Inference (with dimension check + dynamic input names)
# ===============================
def run_onnx(onnx_path, x, u):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    print(f"\n🔄 Loading ONNX model: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    print(f"📥 ONNX input names: {input_names}")
    print(f"📤 ONNX output names: {[out.name for out in sess.get_outputs()]}")
    
    if x.shape[1] != 6:
        raise ValueError(f"Invalid state input dim (expected 6, got {x.shape[1]})")
    if u.shape[1] != 4:
        raise ValueError(f"Invalid control input dim (expected 4, got {u.shape[1]})")
    
    inputs = {}
    if len(input_names) >= 2:
        inputs[input_names[0]] = x.astype(np.float32)
        inputs[input_names[1]] = u.astype(np.float32)
    else:
        raise RuntimeError(f"Invalid ONNX input count (expected 2, got {len(input_names)})")

    outputs = sess.run(None, inputs)
    x_next_onnx = outputs[0]
    
    if x_next_onnx.shape[1] != 6:
        raise ValueError(f"Invalid ONNX output dim (expected 6, got {x_next_onnx.shape[1]})")
    
    print("✅ ONNX inference completed")
    return x_next_onnx


# ===============================
# 3️⃣ Plot Function: PyTorch vs ONNX Comparison Visualization
# ===============================
def plot_pytorch_vs_onnx(x_next_pt, x_next_onnx, diff, save_dir="onnx_verify_plots"):
    """
    Plot comparison of PyTorch and ONNX model outputs
    :param x_next_pt: PyTorch output (np.array, [batch,6])
    :param x_next_onnx: ONNX output (np.array, [batch,6])
    :param diff: Error (PyTorch - ONNX) (np.array, [batch,6])
    :param save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    state_names = ["x", "y", "yaw", "u", "v", "r"]  # 6-DOF state names
    batch_size = x_next_pt.shape[0]
    sample_idx = 0  # Select the 1st sample for dimension-wise comparison

    # ---------- Plot 1: Single Sample - 6D Output Comparison (PyTorch vs ONNX) ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        # Plot output values
        ax.plot([x_next_pt[sample_idx, i]], marker='o', color='blue', label='PyTorch', markersize=8)
        ax.plot([x_next_onnx[sample_idx, i]], marker='s', color='orange', label='ONNX', markersize=8)
        # Annotate error value
        ax.text(0.5, 0.9, f'Error: {diff[sample_idx, i]:.8f}', ha='center', va='top', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax.set_title(f'State: {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample', fontsize=10)
        ax.set_ylabel('Output Value', fontsize=10)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
    fig.suptitle(f'Single Sample (No.{sample_idx+1}) - PyTorch vs ONNX Output Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'single_sample_dim_compare.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---------- Plot 2: Batch Samples - Per-Dimension Error Distribution Histogram ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        ax.hist(diff[:, i], bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
        # Annotate error statistics
        mean_err = np.mean(diff[:, i])
        std_err = np.std(diff[:, i])
        ax.axvline(mean_err, color='red', linestyle='--', label=f'Mean: {mean_err:.8f}')
        ax.axvline(mean_err + std_err, color='orange', linestyle=':', label=f'+1σ: {mean_err+std_err:.8f}')
        ax.axvline(mean_err - std_err, color='orange', linestyle=':', label=f'-1σ: {mean_err-std_err:.8f}')
        ax.set_title(f'State: {name} - Error Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Error (PyTorch - ONNX)', fontsize=10)
        ax.set_ylabel('Sample Count', fontsize=10)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
    fig.suptitle(f'Batch Samples (Total {batch_size}) - Per-Dimension Error Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'batch_sample_error_hist.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---------- Plot 3: Batch Samples - Global Error Scatter Plot (All Dimensions) ----------
    plt.figure(figsize=(10, 6))
    # Flatten all error values for scatter plot
    all_diff = diff.flatten()
    plt.scatter(range(len(all_diff)), all_diff, color='darkred', alpha=0.6, s=5)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.8, label='Error = 0')
    # Annotate max/min error
    max_idx = np.argmax(all_diff)
    min_idx = np.argmin(all_diff)
    plt.scatter(max_idx, all_diff[max_idx], color='red', s=20, label=f'Max Error: {all_diff[max_idx]:.8f}')
    plt.scatter(min_idx, all_diff[min_idx], color='blue', s=20, label=f'Min Error: {all_diff[min_idx]:.8f}')
    plt.title(f'Batch Samples - Global Error Scatter Plot (Total {batch_size*6} Data Points)', fontsize=14, fontweight='bold')
    plt.xlabel('Data Point Index (All Dimensions)', fontsize=12)
    plt.ylabel('Error (PyTorch - ONNX)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'batch_all_error_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Comparison plots saved to directory: {save_dir}/")
    print(f"   Includes: Single sample comparison, batch error histogram, global error scatter plot")


# ===============================
# 4️⃣ Verify full_step.onnx (with error analysis + plotting)
# ===============================
def verify_full_model():
    # Configuration
    device = "cpu"
    ckpt_path = "checkpoints_horizontal/best_model.pt"
    onnx_path = "onnx_export/full_step.onnx"
    batch_size = 32  # Test batch size, can be modified to 32/64

    # 1. Load PyTorch model
    model = load_model(ckpt_path, device)

    # 2. Create test input (fixed random seed for reproducibility)
    torch.manual_seed(42)
    x = torch.randn(batch_size, 6).to(device)
    u = torch.randn(batch_size, 4).to(device)
    print(f"\n📊 Test input shape: x={x.shape}, u={u.shape}")

    # 3. PyTorch inference
    print("\n🔄 Running PyTorch inference...")
    with torch.no_grad():
        z = model.encode(x)
        z_next = model.latent_step(z, u)
        x_next_pt = model.reconstruct_state(z_next)
    x_np = x.cpu().numpy()
    u_np = u.cpu().numpy()
    x_next_pt_np = x_next_pt.cpu().numpy()

    # 4. ONNX inference
    x_next_onnx = run_onnx(onnx_path, x_np, u_np)

    # 5. Error calculation
    diff = x_next_pt_np - x_next_onnx
    abs_diff = np.abs(diff)
    state_names = ["x", "y", "yaw", "u", "v", "r"]

    # 6. Print error analysis
    print("\n" + "="*60)
    print("📈 PyTorch vs ONNX Error Analysis")
    print("="*60)
    print(f"Batch size: {batch_size} | State dimensions: 6")
    print(f"Max Error: {abs_diff.max():.8f}")
    print(f"Mean Error: {abs_diff.mean():.8f}")
    print(f"MSE (Mean Squared Error): {np.mean(diff**2):.8f}")
    print(f"RMSE (Root Mean Squared Error): {np.sqrt(np.mean(diff**2)):.8f}")
    print("-"*60)
    print("Per-dimension RMSE:")
    for i, name in enumerate(state_names):
        rmse = np.sqrt(np.mean(diff[:, i]**2))
        print(f"  {name:4s}: {rmse:.8f}")
    print("="*60)

    # 7. Plot comparison figures
    plot_pytorch_vs_onnx(x_next_pt_np, x_next_onnx, diff)

    # 8. Verification result judgment
    if abs_diff.max() < 1e-5:
        print("\n✅ ONNX model verification passed! PyTorch and ONNX outputs are highly consistent")
    else:
        print("\n⚠️ ONNX model verification warning! Error exceeds threshold (1e-5)")
        print("   Possible causes: Mismatched ONNX opset version / Model not in eval mode / Uncompatible custom ops")


if __name__ == "__main__":
    try:
        verify_full_model()
    except Exception as e:
        print(f"\n❌ Verification failed: {type(e).__name__}: {e}")
        # Uncomment below to print detailed error traceback
        # import traceback
        # traceback.print_exc()