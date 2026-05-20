#!/usr/bin/env python3
"""ONNX rollout 单步推理耗时 benchmark（state0 + u_seq + dt -> states）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koopman import evalkit as ek  # noqa: E402
from koopman import paths as P  # noqa: E402
from koopman.export import TRACED_HORIZON  # noqa: E402
from koopman.paths import setup_repo  # noqa: E402

setup_repo()


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def _resolve_providers(provider: str) -> Tuple[List[str], str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if provider == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"
        return ["CPUExecutionProvider"], "CPUExecutionProvider"
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider 不可用，请改用 --provider cpu")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"
    return ["CPUExecutionProvider"], "CPUExecutionProvider"


def _load_inputs(
    data_path: str | None,
    pred_len: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    dt = 0.1
    if data_path is None:
        rng = np.random.default_rng(seed)
        state0 = rng.standard_normal(6, dtype=np.float32)
        u_seq = rng.standard_normal((pred_len, 4), dtype=np.float32)
        return state0, u_seq, dt

    states_full, ctrls_full, _, _, t0g, _, _ = ek._flatten_segments(
        data_path, pred_len=pred_len, stride=1
    )
    if t0g.shape[0] == 0:
        raise RuntimeError(f"数据集 {data_path} 无有效样本（pred_len={pred_len}）")
    t0 = int(t0g[0])
    state0 = states_full[t0].astype(np.float32)
    u_seq = ctrls_full[t0 : t0 + pred_len].astype(np.float32)
    return state0, u_seq, dt


def _make_feed(state0: np.ndarray, u_seq: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    return {
        "state0": state0,
        "u_seq": u_seq,
        "dt": np.array(dt, dtype=np.float32),
    }


def _run_once(sess, feed: Dict[str, np.ndarray]) -> np.ndarray:
    return sess.run(None, feed)[0]


def benchmark_session(
    sess,
    feed: Dict[str, np.ndarray],
    warmup: int,
    iters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(warmup):
        _run_once(sess, feed)

    lat_ms = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        out = _run_once(sess, feed)
        lat_ms[i] = (time.perf_counter() - t0) * 1e3
    return lat_ms, out


def summarize_latency(lat_ms: np.ndarray) -> Dict[str, float]:
    return {
        "count": int(lat_ms.shape[0]),
        "mean_ms": float(np.mean(lat_ms)),
        "std_ms": float(np.std(lat_ms)),
        "min_ms": float(np.min(lat_ms)),
        "max_ms": float(np.max(lat_ms)),
        "median_ms": float(np.median(lat_ms)),
        "p50_ms": _percentile(lat_ms, 50),
        "p90_ms": _percentile(lat_ms, 90),
        "p95_ms": _percentile(lat_ms, 95),
        "p99_ms": _percentile(lat_ms, 99),
        "throughput_fps": float(1000.0 / np.mean(lat_ms)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Koopman v4 ONNX 单步 rollout 耗时测试")
    p.add_argument(
        "--onnx",
        type=str,
        default=str(P.CPP_MPC_DIR / "weights" / "koopman_rollout.onnx"),
        help="ONNX 模型路径",
    )
    p.add_argument("--data", type=str, default=str(P.TEST), help="输入样本来源；设 none 则用随机输入")
    p.add_argument("--pred_len", type=int, default=TRACED_HORIZON)
    p.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--warmup", type=int, default=50, help="预热次数（不计入统计）")
    p.add_argument("--iters", type=int, default=1000, help="计时次数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=str(P.EVAL_OUT_DIR / "v4_onnx_benchmark"))
    p.add_argument("--tag", type=str, default="v4_onnx")
    args = p.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 不存在: {onnx_path}")

    import onnxruntime as ort

    providers, active_provider = _resolve_providers(args.provider)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)

    data_path = None if str(args.data).lower() == "none" else args.data
    state0, u_seq, dt = _load_inputs(data_path, args.pred_len, args.seed)
    feed = _make_feed(state0, u_seq, dt)

    lat_ms, out = benchmark_session(sess, feed, warmup=args.warmup, iters=args.iters)
    stats = summarize_latency(lat_ms)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"{args.tag}_{active_provider.lower()}_{ts}.json"

    report = {
        "onnx_path": str(onnx_path.resolve()),
        "provider_requested": args.provider,
        "provider_active": active_provider,
        "providers_available": ort.get_available_providers(),
        "pred_len": args.pred_len,
        "output_shape": list(out.shape),
        "warmup": args.warmup,
        "iters": args.iters,
        "data_path": data_path,
        "input_state0": state0.tolist(),
        "latency_ms": stats,
        "latency_ms_samples": lat_ms.tolist(),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Koopman ONNX Benchmark ===")
    print(f"model      : {onnx_path}")
    print(f"provider   : {active_provider}")
    print(f"output     : shape={tuple(out.shape)} dtype={out.dtype}")
    print(f"warmup/iters: {args.warmup}/{args.iters}")
    print("--- single-step latency (ms) ---")
    print(f"mean   : {stats['mean_ms']:.4f}")
    print(f"median : {stats['median_ms']:.4f}")
    print(f"min    : {stats['min_ms']:.4f}")
    print(f"max    : {stats['max_ms']:.4f}")
    print(f"p95    : {stats['p95_ms']:.4f}")
    print(f"p99    : {stats['p99_ms']:.4f}")
    print(f"fps    : {stats['throughput_fps']:.1f}")
    print(f"[OK] report -> {report_path}")


if __name__ == "__main__":
    main()
