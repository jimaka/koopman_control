#!/usr/bin/env python3
"""从 v4 dict-input checkpoint 导出 ONNX，并在 test 集上对比 PT/ONNX 精度与作图。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koopman import evalkit as ek  # noqa: E402
from koopman import paths as P  # noqa: E402
from koopman.export import KoopmanRollout, TRACED_HORIZON, TRACED_HORIZON_V4  # noqa: E402
from koopman.paths import setup_repo  # noqa: E402
from new_v4_dict_input.model_v4_dict_input import (  # noqa: E402
    FEATURE_DICT_ATOMS_16,
    HorizontalKoopmanModelV4DictInput,
)

setup_repo()

CHANNEL_NAMES = ["x", "y", "yaw", "u", "v", "r"]
DYN_NAMES = ["u", "v", "r"]


def load_v4_model(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats = ckpt["stats"]
    sd = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    args_d = ckpt.get("args", {}) or {}

    model_class = ckpt.get("model_class", "HorizontalKoopmanModelV4DictInput")
    if model_class != "HorizontalKoopmanModelV4DictInput":
        raise ValueError(
            f"checkpoint model_class={model_class!r}，本脚本仅支持 v4 dict-input 模型。"
        )

    model = HorizontalKoopmanModelV4DictInput(
        hidden_dim=int(args_d.get("hidden_dim", 32)),
        clamp_pif=float(args_d.get("clamp_pif", 5.0)),
    )
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, stats


def export_onnx(rollout: nn.Module, out_path: str, pred_len: int, opset: int = 18) -> None:
    rollout.eval()
    s0 = torch.zeros(6, dtype=torch.float32)
    u = torch.zeros(pred_len, 4, dtype=torch.float32)
    dt = torch.tensor(0.1, dtype=torch.float32)
    torch.onnx.export(
        rollout,
        (s0, u, dt),
        out_path,
        input_names=["state0", "u_seq", "dt"],
        output_names=["states"],
        opset_version=opset,
        dynamo=False,
    )


def _pick_quantile_indices(values: np.ndarray, n: int) -> np.ndarray:
    m = int(values.shape[0])
    n = min(max(int(n), 1), m)
    sorted_idx = np.argsort(values)
    q = np.linspace(0, m - 1, n).astype(int)
    return sorted_idx[q]


@torch.no_grad()
def rollout_pytorch_batch(
    rollout: nn.Module,
    state0: np.ndarray,
    u_seq: np.ndarray,
    dt: float,
) -> np.ndarray:
    s0 = torch.from_numpy(state0.astype(np.float32))
    u = torch.from_numpy(u_seq.astype(np.float32))
    dt_t = torch.tensor(dt, dtype=torch.float32)
    out = rollout(s0, u, dt_t).numpy()
    return out


def rollout_onnx_batch(
    sess,
    state0: np.ndarray,
    u_seq: np.ndarray,
    dt: float,
) -> np.ndarray:
    out = sess.run(
        None,
        {
            "state0": state0.astype(np.float32),
            "u_seq": u_seq.astype(np.float32),
            "dt": np.array(dt, dtype=np.float32),
        },
    )[0]
    return out


def collect_test_cases(
    data_path: str,
    pred_len: int,
    max_samples: int | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    states_full, ctrls_full, _, _, t0g, _, _ = ek._flatten_segments(
        data_path, pred_len=pred_len, stride=1
    )
    if max_samples is not None and t0g.shape[0] > max_samples:
        sel = np.linspace(0, t0g.shape[0] - 1, max_samples).astype(int)
        t0g = t0g[sel]

    state0_batch = states_full[t0g]
    u_seq_batch = np.stack([ctrls_full[t0 : t0 + pred_len] for t0 in t0g], axis=0)
    return state0_batch, u_seq_batch, t0g


def compare_pt_vs_onnx_on_test(
    rollout: nn.Module,
    onnx_path: str,
    data_path: str,
    dt: float,
    pred_len: int,
    max_samples: int | None,
    compare_atol: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], float, np.ndarray, np.ndarray, np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    state0_batch, u_seq_batch, t0g = collect_test_cases(data_path, pred_len, max_samples)
    m = state0_batch.shape[0]
    k = pred_len

    pt_out = np.empty((m, k + 1, 6), dtype=np.float32)
    ort_out = np.empty((m, k + 1, 6), dtype=np.float32)
    max_err = 0.0

    for i in range(m):
        pt = rollout_pytorch_batch(rollout, state0_batch[i], u_seq_batch[i], dt)
        ort = rollout_onnx_batch(sess, state0_batch[i], u_seq_batch[i], dt)
        pt_out[i] = pt
        ort_out[i] = ort
        max_err = max(max_err, float(np.max(np.abs(pt - ort))))

    diff = ort_out - pt_out
    abs_diff = np.abs(diff)

    per_step: Dict[str, np.ndarray] = {
        "step": np.arange(0, k + 1, dtype=np.int64),
        "n_samples": np.full(k + 1, m, dtype=np.int64),
    }
    for ci, name in enumerate(CHANNEL_NAMES):
        per_step[f"{name}_rmse"] = np.sqrt(np.mean(diff[:, :, ci] ** 2, axis=0))
        per_step[f"{name}_mae"] = np.mean(abs_diff[:, :, ci], axis=0)
        per_step[f"{name}_max_abs"] = np.max(abs_diff[:, :, ci], axis=0)

    vel_err = np.sqrt(diff[:, :, 3] ** 2 + diff[:, :, 4] ** 2)
    per_step["vel_rmse"] = np.sqrt(np.mean(vel_err ** 2, axis=0))
    per_step["vel_mae"] = np.mean(vel_err, axis=0)
    per_step["vel_max_abs"] = np.max(vel_err, axis=0)
    per_step["state_max_abs"] = np.max(abs_diff, axis=(0, 2))

    summary = {
        "n_samples": int(m),
        "pred_len": int(pred_len),
        "data_path": data_path,
        "max_abs_err_all": float(max_err),
        "mean_abs_err_all": float(np.mean(abs_diff)),
        "vel_rmse_mean": float(np.mean(per_step["vel_rmse"])),
        f"vel_rmse_step_{k}": float(per_step["vel_rmse"][-1]),
        f"u_rmse_step_{k}": float(per_step["u_rmse"][-1]),
        f"v_rmse_step_{k}": float(per_step["v_rmse"][-1]),
        f"r_rmse_step_{k}": float(per_step["r_rmse"][-1]),
        "state_max_abs_stepK": float(per_step["state_max_abs"][-1]),
        "passed": bool(max_err <= compare_atol),
        "compare_atol": float(compare_atol),
    }
    return per_step, summary, max_err, pt_out, ort_out, t0g


def write_per_step_csv(per_step: Dict[str, np.ndarray], path: str) -> None:
    cols = [
        "step",
        "n_samples",
        "state_max_abs",
        "vel_rmse",
        "vel_mae",
        "vel_max_abs",
        "u_rmse",
        "v_rmse",
        "r_rmse",
        "u_mae",
        "v_mae",
        "r_mae",
        "u_max_abs",
        "v_max_abs",
        "r_max_abs",
    ]
    k1 = int(per_step["step"].shape[0])
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for i in range(k1):
            row = []
            for c in cols:
                v = per_step[c][i]
                if isinstance(v, (np.integer, int)):
                    row.append(str(int(v)))
                elif isinstance(v, (np.floating, float)):
                    row.append("nan" if np.isnan(v) else f"{float(v):.8g}")
                else:
                    row.append(str(v))
            f.write(",".join(row) + "\n")


def plot_channel_rmse_vs_step(
    per_step: Dict[str, np.ndarray],
    channel: str,
    path: str,
    title: str,
) -> None:
    steps = per_step["step"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(steps, per_step[f"{channel}_rmse"], "o-", lw=1.6, label=f"{channel} rmse")
    ax.plot(steps, per_step[f"{channel}_mae"], "s--", lw=1.2, alpha=0.8, label=f"{channel} mae")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("PT vs ONNX error [m/s]" if channel in ("u", "v") else "PT vs ONNX error")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_state_max_err_vs_step(per_step: Dict[str, np.ndarray], path: str, title: str) -> None:
    steps = per_step["step"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(steps, per_step["state_max_abs"], "o-", color="C3", lw=1.6, label="max |PT-ONNX| over 6 states")
    ax.plot(steps, per_step["vel_rmse"], "s--", color="C0", lw=1.2, label="vel rmse (u,v)")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("error")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pt_onnx_scatter(
    pt_out: np.ndarray,
    ort_out: np.ndarray,
    channel_idx: int,
    channel_name: str,
    path: str,
    title: str,
    max_points: int = 30000,
) -> None:
    pt_ch = pt_out[..., channel_idx].reshape(-1)
    ort_ch = ort_out[..., channel_idx].reshape(-1)
    n = pt_ch.shape[0]
    if n > max_points:
        sel = np.linspace(0, n - 1, max_points).astype(int)
        pt_ch, ort_ch = pt_ch[sel], ort_ch[sel]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(pt_ch, ort_ch, s=6, alpha=0.25)
    lo = float(min(pt_ch.min(), ort_ch.min()))
    hi = float(max(pt_ch.max(), ort_ch.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="ideal y=x")
    ax.set_xlabel(f"PyTorch {channel_name}")
    ax.set_ylabel(f"ONNX {channel_name}")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pt_onnx_curve_samples(
    pt_out: np.ndarray,
    ort_out: np.ndarray,
    channel_idx: int,
    channel_name: str,
    path: str,
    title: str,
    n_samples: int = 6,
) -> None:
    diff = ort_out - pt_out
    err_k = np.max(np.abs(diff[:, -1, :]), axis=1)
    pick = _pick_quantile_indices(err_k, n_samples)
    steps = np.arange(pt_out.shape[1])
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, sidx in enumerate(pick):
        ax = axes[i // 2, i % 2]
        ax.plot(steps, pt_out[sidx, :, channel_idx], "g-", lw=1.4, label="PyTorch")
        ax.plot(steps, ort_out[sidx, :, channel_idx], "r--", lw=1.2, label="ONNX")
        ax.set_title(f"sample #{sidx} | max_err@K={err_k[sidx]:.3e}")
        ax.set_xlabel("step")
        ax.set_ylabel(channel_name)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_err_hist_step_k(
    pt_out: np.ndarray,
    ort_out: np.ndarray,
    path: str,
    title: str,
    pred_len: int,
) -> None:
    diff = np.abs(ort_out[:, -1, 3:6] - pt_out[:, -1, 3:6])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ci, name in enumerate(DYN_NAMES):
        ax = axes[ci]
        vals = diff[:, ci]
        ax.hist(vals, bins=50, color=f"C{ci}", alpha=0.7)
        ax.axvline(float(np.mean(vals)), color="k", ls="--", lw=1.0, label=f"mean={np.mean(vals):.2e}")
        ax.set_title(f"|PT-ONNX| @ step {pred_len}: {name}")
        ax.set_xlabel("abs error")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def generate_compare_report(
    report_dir: str,
    tag: str,
    per_step: Dict[str, np.ndarray],
    summary: Dict[str, float],
    pt_out: np.ndarray,
    ort_out: np.ndarray,
    pred_len: int,
) -> List[str]:
    os.makedirs(report_dir, exist_ok=True)
    outputs: List[str] = []

    csv_path = os.path.join(report_dir, f"{tag}_pt_onnx_per_step.csv")
    write_per_step_csv(per_step, csv_path)
    outputs.append(csv_path)

    summary_path = os.path.join(report_dir, f"{tag}_pt_onnx_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    outputs.append(summary_path)

    plots = [
        (
            f"{tag}_state_max_err_vs_step.png",
            lambda p: plot_state_max_err_vs_step(
                per_step, p, f"{tag}: PT vs ONNX max error vs step (test set)"
            ),
        ),
        (
            f"{tag}_u_pt_onnx_rmse_vs_step.png",
            lambda p: plot_channel_rmse_vs_step(
                per_step, "u", p, f"{tag}: u (surge) PT vs ONNX error vs step"
            ),
        ),
        (
            f"{tag}_v_pt_onnx_rmse_vs_step.png",
            lambda p: plot_channel_rmse_vs_step(
                per_step, "v", p, f"{tag}: v (sway) PT vs ONNX error vs step"
            ),
        ),
        (
            f"{tag}_r_pt_onnx_rmse_vs_step.png",
            lambda p: plot_channel_rmse_vs_step(
                per_step, "r", p, f"{tag}: r PT vs ONNX error vs step"
            ),
        ),
        (
            f"{tag}_u_pt_onnx_scatter.png",
            lambda p: plot_pt_onnx_scatter(
                pt_out, ort_out, 3, "u [m/s]", p, f"{tag}: PyTorch vs ONNX u scatter"
            ),
        ),
        (
            f"{tag}_v_pt_onnx_scatter.png",
            lambda p: plot_pt_onnx_scatter(
                pt_out, ort_out, 4, "v [m/s]", p, f"{tag}: PyTorch vs ONNX v scatter"
            ),
        ),
        (
            f"{tag}_u_pt_onnx_curve_samples.png",
            lambda p: plot_pt_onnx_curve_samples(
                pt_out, ort_out, 3, "u [m/s]", p, f"{tag}: u curve samples (PT vs ONNX)"
            ),
        ),
        (
            f"{tag}_v_pt_onnx_curve_samples.png",
            lambda p: plot_pt_onnx_curve_samples(
                pt_out, ort_out, 4, "v [m/s]", p, f"{tag}: v curve samples (PT vs ONNX)"
            ),
        ),
        (
            f"{tag}_pt_onnx_err_hist_step{pred_len}.png",
            lambda p: plot_err_hist_step_k(
                pt_out, ort_out, p, f"{tag}: PT vs ONNX abs error @ step {pred_len}", pred_len
            ),
        ),
    ]
    for fname, fn in plots:
        p = os.path.join(report_dir, fname)
        fn(p)
        outputs.append(p)
    return outputs


def verify_onnx_vs_pytorch_random(
    rollout: nn.Module,
    onnx_path: str,
    *,
    pred_len: int,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    n_random: int = 8,
) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rollout.eval()
    max_err = 0.0
    rng = np.random.default_rng(42)

    cases = [(torch.zeros(6), torch.zeros(pred_len, 4))]
    for _ in range(n_random):
        s0 = torch.tensor(rng.normal(0, 1, size=6).astype(np.float32))
        s0[2] = float(rng.uniform(-0.5, 0.5))
        u = torch.tensor(rng.uniform(-5, 5, size=(pred_len, 4)).astype(np.float32))
        cases.append((s0, u))

    with torch.no_grad():
        for s0, u in cases:
            dt = torch.tensor(0.1, dtype=torch.float32)
            pt_out = rollout(s0, u, dt).numpy()
            ort_out = rollout_onnx_batch(sess, s0.numpy(), u.numpy(), 0.1)
            err = float(np.max(np.abs(pt_out - ort_out)))
            max_err = max(max_err, err)

    if max_err > atol + rtol:
        raise RuntimeError(
            f"random-case ONNX vs PyTorch max_abs_err={max_err:.6e} exceeds tol atol={atol} rtol={rtol}"
        )
    return max_err


def default_report_dir(tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(P.EVAL_OUT_DIR / "v4_onnx_compare" / f"{tag}_{ts}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="v4 dict-input Koopman 模型 ONNX 导出 + test 集 PT/ONNX 精度对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str, default=str(P.CKPT_DIR / "koopman_v4_best.pth"))
    p.add_argument("--out_dir", type=str, default=str(P.CPP_MPC_DIR / "weights"))
    p.add_argument("--onnx_name", type=str, default="koopman_rollout.onnx")
    p.add_argument("--report_dir", type=str, default=None, help="PT/ONNX 对比报告与图片输出目录")
    p.add_argument("--data", type=str, default=str(P.TEST), help="精度对比使用的数据集")
    p.add_argument("--pred_len", type=int, default=TRACED_HORIZON_V4, help="ONNX rollout 步数（v4 20s 默认 200）")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=512, help="test 集对比样本数上限")
    p.add_argument("--tag", type=str, default="v4")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--atol", type=float, default=1e-4, help="随机样本 PT/ONNX 对照阈值")
    p.add_argument("--compare_atol", type=float, default=1e-3, help="test 集 PT/ONNX 最大误差阈值（仅记录/告警）")
    p.add_argument("--write_rollout_check", action="store_true")
    p.add_argument("--skip_test_compare", action="store_true", help="跳过 test 集对比与作图")
    return p


def main() -> int:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else str(REPO_ROOT / args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
    if not args.skip_test_compare and not os.path.isfile(args.data):
        raise FileNotFoundError(f"dataset 不存在: {args.data}")

    report_dir = args.report_dir or default_report_dir(args.tag)
    os.makedirs(report_dir, exist_ok=True)

    device = torch.device("cpu")
    model, stats = load_v4_model(ckpt_path, device)
    rollout = KoopmanRollout(model, stats).cpu().eval()

    onnx_path = os.path.join(args.out_dir, args.onnx_name)
    export_onnx(rollout, onnx_path, pred_len=args.pred_len, opset=args.opset)
    print(f"[OK] Saved ONNX -> {onnx_path} (pred_len={args.pred_len})")

    random_max_err = verify_onnx_vs_pytorch_random(
        rollout, onnx_path, pred_len=args.pred_len, atol=args.atol
    )
    print(f"[OK] random-case ONNX vs PyTorch max_abs_err={random_max_err:.6e}")

    compare_summary: Dict | None = None
    report_files: List[str] = []
    test_max_err = float("nan")

    if not args.skip_test_compare:
        per_step, compare_summary, test_max_err, pt_out, ort_out, _ = compare_pt_vs_onnx_on_test(
            rollout=rollout,
            onnx_path=onnx_path,
            data_path=args.data,
            dt=args.dt,
            pred_len=args.pred_len,
            max_samples=args.max_samples,
            compare_atol=args.compare_atol,
        )
        report_files = generate_compare_report(
            report_dir=report_dir,
            tag=args.tag,
            per_step=per_step,
            summary=compare_summary,
            pt_out=pt_out,
            ort_out=ort_out,
            pred_len=args.pred_len,
        )
        print(f"[OK] test-set PT vs ONNX max_abs_err={test_max_err:.6e}")
        print(f"[OK] compare report -> {report_dir}")
        for p in report_files:
            print(f"       {p}")
        if compare_summary["passed"]:
            print(f"[OK] test-set compare passed (max_abs_err <= {args.compare_atol})")
        else:
            print(
                f"[WARN] test-set max_abs_err={test_max_err:.6e} > compare_atol={args.compare_atol}; "
                "report still saved"
            )

    meta = {
        "ckpt": args.ckpt,
        "onnx": args.onnx_name,
        "format": "onnx",
        "model_class": "HorizontalKoopmanModelV4DictInput",
        "input_mode": "dict16_only",
        "feature_dict_atoms": list(FEATURE_DICT_ATOMS_16),
        "latent_dim": int(getattr(model, "latent_dim")),
        "hidden_dim": int(getattr(model, "hidden_dim")),
        "clamp_pif": float(getattr(model, "clamp_pif")),
        "dt": args.dt,
        "horizon_default": args.pred_len,
        "onnx_verify_random_max_abs_err": random_max_err,
        "onnx_verify_test_max_abs_err": test_max_err,
        "compare_report_dir": report_dir,
        "compare_data": args.data,
        "compare_n_samples": compare_summary["n_samples"] if compare_summary else None,
        "u_min": [-100.0, -35.0, -100.0, -35.0],
        "u_max": [100.0, 35.0, 100.0, 35.0],
        "dyn_mean": stats["state_mean"][3:6].tolist(),
        "dyn_std": stats["state_std"][3:6].tolist(),
        "ctrl_mean": stats["ctrl_mean"].tolist(),
        "ctrl_std": stats["ctrl_std"].tolist(),
    }
    meta_path = os.path.join(args.out_dir, "model_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            prev = json.load(f)
        meta = {**prev, **meta}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OK] Updated meta -> {meta_path}")

    if args.write_rollout_check:
        s0 = torch.tensor([0.0, 0.0, 0.0, 1.5, 0.0, 0.0], dtype=torch.float32)
        u_seq = torch.zeros(args.pred_len, 4, dtype=torch.float32)
        with torch.no_grad():
            states = rollout(s0, u_seq, torch.tensor(args.dt, dtype=torch.float32))
        npz_path = os.path.join(args.out_dir, "rollout_check.npz")
        np.savez_compressed(
            npz_path,
            state0=s0.numpy(),
            u_seq=u_seq.numpy(),
            states=states.numpy(),
        )
        print(f"[OK] Wrote rollout_check -> {npz_path}")

    print("=== V4 ONNX EXPORT + COMPARE DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
