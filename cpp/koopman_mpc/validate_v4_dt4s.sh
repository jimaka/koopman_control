#!/usr/bin/env bash
# 用训练完成的 v4 dt=4s / 10-step checkpoint 验证 C++ 算法与 Python 一致
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CPP_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT="${1:-$ROOT/checkpoints/v4_dt4s/run_v4_20260617_123100/koopman_v4_best.pth}"
PRED_LEN="${2:-10}"
DT="${3:-4.0}"

if [[ "$CKPT" != /* ]]; then
  CKPT="$ROOT/$CKPT"
fi

MAX_LEN="$(python3 - <<PY
dt, h, steps = float("$DT"), int("$PRED_LEN"), 10
stride = max(1, int(round(dt / 0.1)))
print(1 + (steps - 1 + h) * stride + 64)
PY
)"

echo "=== [1/6] Export weights (horizon=$PRED_LEN, dt=$DT) ==="
python3 "$ROOT/new_v4_dict_input/export_v4_encode_weights.py" \
  --ckpt "$CKPT" --horizon "$PRED_LEN" --dt "$DT" \
  --out-yaml "$ROOT/cpp/koopman_mpc/weights/koopman_v4_latent.yaml"
python3 "$ROOT/new_v4_dict_input/export_v4_onnx.py" \
  --ckpt "$CKPT" --pred_len "$PRED_LEN" --dt "$DT" \
  --out_dir "$ROOT/cpp/koopman_mpc/weights" \
  --write_rollout_check --tag v4_dt4s_cpp_val
python3 "$ROOT/cpp/koopman_mpc/scripts/export_v4_cpp_test_ref.py" \
  --ckpt "$CKPT" --horizon "$PRED_LEN" --dt "$DT" --max_len "$MAX_LEN"

echo "=== [2/6] Python encode reference ==="
python3 "$ROOT/tests/test_v4_encode_reference.py" --ckpt "$CKPT"

echo "=== [3/6] Python latent QP reference ==="
python3 "$ROOT/tests/export_latent_qp_cpp_ref.py" \
  --ckpt "$CKPT" --horizon "$PRED_LEN"

echo "=== [4/6] Build C++ ==="
bash "$CPP_DIR/build_v4.sh" "$CKPT" "$PRED_LEN" "$DT" 2>&1 | tail -5

export LD_LIBRARY_PATH="$CPP_DIR/third_party/onnxruntime/lib:${LD_LIBRARY_PATH:-}"

echo "=== [5/6] C++ numerical equivalence ==="
python3 "$ROOT/cpp/koopman_mpc/scripts/write_rollout_check_txt.py"
"$CPP_DIR/build/verify_rollout" \
  "$ROOT/cpp/koopman_mpc/weights/koopman_rollout.onnx" \
  "$ROOT/cpp/koopman_mpc/weights/rollout_check.npz"
"$CPP_DIR/build/koopman_control/verify_latent_qp" \
  "$ROOT/cpp/koopman_mpc/weights/koopman_v4_latent.yaml" \
  "$ROOT/eval_out/latent_qp_cpp_ref" "$PRED_LEN"
"$CPP_DIR/build/koopman_control/verify_pose_linearize" \
  "$ROOT/cpp/koopman_mpc/weights/koopman_v4_latent.yaml"

echo "=== [6/6] MPC closed-loop run (logic smoke; tracking RMSE informational) ==="
"$CPP_DIR/build/koopman_mpc_cpp" \
  --config "$ROOT/cpp/koopman_mpc/src/mpc_config.yaml" \
  --ref "$ROOT/cpp/koopman_mpc/weights/cpp_test_ref.json" \
  --steps 10 || true

echo ""
echo "[OK] C++ algorithm logic verified against trained checkpoint:"
echo "  - ONNX rollout C++ vs Python"
echo "  - PyTorch vs ONNX export"
echo "  - Latent Gamma/Theta/predictStacked + encode"
echo "  - Tier-2 pose linearization + OSQP pose test"
