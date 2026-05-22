#!/usr/bin/env python3
"""端到端验证：PT checkpoint → ONNX → C++ rollout / MPC（需在 build 后调用）。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from koopman import evalkit as ek
from koopman import paths as P
from koopman.export import KoopmanRollout, TRACED_HORIZON

SCRIPTS = os.path.dirname(__file__)
BUILD_DIR = P.CPP_MPC_DIR / "build"
WEIGHTS = P.CPP_MPC_DIR / "weights"
ORT_LIB = P.CPP_MPC_DIR / "third_party" / "onnxruntime" / "lib"


def _cpp_env() -> dict[str, str]:
    env = os.environ.copy()
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{ORT_LIB}:{prev}" if prev else str(ORT_LIB)
    return env


def step_export_onnx(ckpt: str, out_dir: str) -> None:
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(SCRIPTS, "export_onnx.py"),
            "--ckpt",
            ckpt,
            "--out_dir",
            out_dir,
        ],
        cwd=REPO_ROOT,
    )


def step_export_refs(ckpt: str, out_dir: str) -> None:
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(SCRIPTS, "export_cpp_test_ref.py"),
            "--ckpt",
            ckpt,
            "--out_dir",
            out_dir,
        ],
        cwd=REPO_ROOT,
    )


def step_python_rollout_check(ckpt: str, weights_dir: str) -> float:
    import onnxruntime as ort

    device = torch.device("cpu")
    model, stats = ek.load_model_from_ckpt(
        ckpt if os.path.isabs(ckpt) else str(REPO_ROOT / ckpt), device
    )
    rollout = KoopmanRollout(model, stats).eval()
    npz_path = os.path.join(weights_dir, "rollout_check.npz")
    d = np.load(npz_path)
    s0 = torch.tensor(d["state0"], dtype=torch.float32)
    u = torch.tensor(d["u_seq"], dtype=torch.float32)
    dt = torch.tensor(0.1, dtype=torch.float32)
    with torch.no_grad():
        pt = rollout(s0, u, dt).numpy()

    onnx_path = os.path.join(weights_dir, "koopman_rollout.onnx")
    ort_out = ort.InferenceSession(onnx_path).run(
        None,
        {"state0": s0.numpy(), "u_seq": u.numpy(), "dt": np.array(0.1, np.float32)},
    )[0]
    err = float(np.max(np.abs(pt - ort_out)))
    print(f"[python] rollout_check ONNX max_abs_err={err:.6e}")
    if err > 1e-3:
        raise RuntimeError("rollout_check ONNX error too large")
    return err


def step_write_rollout_txt(weights_dir: str) -> None:
    subprocess.check_call(
        [sys.executable, os.path.join(SCRIPTS, "write_rollout_check_txt.py")],
        cwd=REPO_ROOT,
    )


def step_cpp_verify_rollout(weights_dir: Path) -> None:
    exe = BUILD_DIR / "verify_rollout"
    if not exe.is_file():
        raise FileNotFoundError(f"Missing {exe}; run cpp/koopman_mpc/build.sh first")
    subprocess.check_call(
        [
            str(exe),
            str(weights_dir / "koopman_rollout.onnx"),
            str(weights_dir / "rollout_check.npz"),
        ],
        cwd=REPO_ROOT,
        env=_cpp_env(),
    )


def step_cpp_mpc_smoke(weights_dir: Path) -> None:
    exe = BUILD_DIR / "koopman_mpc_cpp"
    ref = weights_dir / "cpp_test_ref.json"
    subprocess.check_call(
        [str(exe), "--smoketest", "--weights", str(weights_dir), "--ref", str(ref)],
        cwd=REPO_ROOT,
        env=_cpp_env(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(P.CKPT_V3A_BEST))
    parser.add_argument("--weights_dir", default=str(WEIGHTS))
    parser.add_argument("--skip_cpp", action="store_true")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    os.makedirs(weights_dir, exist_ok=True)
    print("=== 1/5 Export ONNX + PT accuracy ===")
    step_export_onnx(args.ckpt, str(weights_dir))

    print("=== 2/5 Export test references ===")
    step_export_refs(args.ckpt, str(weights_dir))

    print("=== 3/5 Python rollout_check vs ONNX ===")
    step_python_rollout_check(args.ckpt, str(weights_dir))

    if args.skip_cpp:
        print("Skipped C++ steps (--skip_cpp)")
        return

    print("=== 4/5 C++ verify_rollout ===")
    step_write_rollout_txt(str(weights_dir))
    step_cpp_verify_rollout(weights_dir)

    print("=== 5/5 C++ MPC smoketest ===")
    step_cpp_mpc_smoke(weights_dir)

    meta = json.loads((P.CPP_MPC_DIR / "weights" / "model_meta.json").read_text(encoding="utf-8"))
    print("=== PIPELINE OK ===")
    print(json.dumps({"onnx_verify_max_abs_err": meta.get("onnx_verify_max_abs_err")}, indent=2))


if __name__ == "__main__":
    main()
