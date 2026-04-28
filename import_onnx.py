import torch
import torch.nn as nn
import onnx
import os
import numpy as np

from koopman import create_koopman_model

def fix_ir_version(path):
    model = onnx.load(path)
    model.ir_version = 9   # ⭐ 强制降级
    onnx.save(model, path)


def load_model_from_checkpoint(ckpt_path, device="cpu"):
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

    print("✅ Model loaded successfully")
    return model


# -------------------------------
# ONNX export helpers
# -------------------------------

def export_encoder(model, save_path, device):
    dummy = torch.randn(1, model.state_dim, device=device)
    torch.onnx.export(
        model.encoder,
        dummy,
        save_path,
        input_names=["x"],
        output_names=["z"],
        opset_version=18,
        do_constant_folding=True,
        # use_external_data_format=False,
        dynamic_axes={"x": {0: "batch"}, "z": {0: "batch"}},
    )
    fix_ir_version(save_path)
    print("✅ encoder exported")


def export_decoder(model, save_path, device):
    dummy = torch.randn(1, model.latent_dim, device=device)
    torch.onnx.export(
        model.decoder,
        dummy,
        save_path,
        input_names=["z"],
        output_names=["x"],
        opset_version=18,
        do_constant_folding=True,
        # use_external_data_format=False,
        dynamic_axes={"z": {0: "batch"}, "x": {0: "batch"}},
    )
    fix_ir_version(save_path)
    print("✅ decoder exported")


class LatentDynamicsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.A = model.A
        self.B = model.B

    def forward(self, z, u):
        return self.A(z) + self.B(u)


def export_latent_dynamics(model, save_path, device):
    wrapper = LatentDynamicsWrapper(model).to(device)

    z_dummy = torch.randn(1, model.latent_dim, device=device)
    u_dummy = torch.randn(1, model.control_dim, device=device)

    torch.onnx.export(
        wrapper,
        (z_dummy, u_dummy),
        save_path,
        input_names=["z", "u"],
        output_names=["z_next"],
        opset_version=18,
        do_constant_folding=True,
        # use_external_data_format=False,
        dynamic_axes={
            "z": {0: "batch"},
            "u": {0: "batch"},
            "z_next": {0: "batch"},
        },
    )

    fix_ir_version(save_path)
    print("✅ latent dynamics exported")


class FullStepWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, u):
        z = self.model.encode(x)
        z_next = self.model.latent_step(z, u)
        x_next = self.model.reconstruct_state(z_next)
        return x_next


def export_full_model(model, save_path, device):
    wrapper = FullStepWrapper(model).to(device)

    x_dummy = torch.randn(1, model.state_dim, device=device)
    u_dummy = torch.randn(1, model.control_dim, device=device)

    torch.onnx.export(
        wrapper,
        (x_dummy, u_dummy),
        save_path,
        input_names=["x", "u"],
        output_names=["x_next"],
        opset_version=18,
        do_constant_folding=True,
        # use_external_data_format=False,
        dynamic_axes={
            "x": {0: "batch"},
            "u": {0: "batch"},
            "x_next": {0: "batch"},
        },
    )

    fix_ir_version(save_path)

    print("✅ full model exported")


# -------------------------------
# Export A, B, C matrices
# -------------------------------

def export_matrices(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    A = model.A.weight.detach().cpu().numpy()
    B = model.B.weight.detach().cpu().numpy()

    np.save(os.path.join(save_dir, "A.npy"), A)
    np.save(os.path.join(save_dir, "B.npy"), B)

    print("✅ A,B matrices saved")

    if hasattr(model, "C"):
        C = model.C.weight.detach().cpu().numpy()
        np.save(os.path.join(save_dir, "C.npy"), C)
        print("✅ C matrix saved")


# -------------------------------
# Main
# -------------------------------

def export_all(ckpt_path, output_dir="onnx_export", device="cpu"):
    os.makedirs(output_dir, exist_ok=True)

    model = load_model_from_checkpoint(ckpt_path, device)

    export_encoder(model, os.path.join(output_dir, "encoder.onnx"), device)
    export_decoder(model, os.path.join(output_dir, "decoder.onnx"), device)
    export_latent_dynamics(model, os.path.join(output_dir, "latent_dynamics.onnx"), device)
    export_full_model(model, os.path.join(output_dir, "full_step.onnx"), device)

    export_matrices(model, output_dir)

    print("\n🎉 All exports completed successfully!")


if __name__ == "__main__":
    export_all(
        "checkpoints_horizontal/best_model.pt",
        output_dir="onnx_export",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )