#!/usr/bin/env bash
# v4 模型 4s 采样步长：训练 + 评估 + 与原 dt=1s 模型对比
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR="${CKPT_DIR:-checkpoints/v4_dt4s}"
CONFIG="new_v4_dict_input/configs/v4_dt4s_train.yaml"

echo "=== Train v4 @ dt=4s ==="
python3 new_v4_dict_input/train_v4_dict_input.py \
  --config "$CONFIG" \
  --ckpt_dir "$CKPT_DIR" \
  "$@"

# 找到最新 run 目录下的 best ckpt
BEST=$(find "$CKPT_DIR" -name koopman_v4_best.pth | sort | tail -1)
if [[ -z "$BEST" ]]; then
  echo "ERROR: no koopman_v4_best.pth found under $CKPT_DIR" >&2
  exit 1
fi
echo "Best checkpoint: $BEST"

echo "=== Eval dt=4s (GT vs Pred) ==="
python3 new_v4_dict_input/eval_v4_dict_input.py \
  --ckpt "$BEST" \
  --data data/koopman_test.npz \
  --pred_len 5 --dt 4.0 \
  --out_dir eval_out/v4_dt4s \
  --tag v4_dt4s

echo "=== Eval dt=1s original ==="
python3 new_v4_dict_input/eval_v4_dict_input.py \
  --ckpt checkpoints/koopman_v4_best.pth \
  --data data/koopman_test.npz \
  --pred_len 20 --dt 1.0 \
  --out_dir eval_out/v4_dt1s_original \
  --tag v4_dt1s_original

echo "=== Compare dt=4s vs dt=1s ==="
python3 new_v4_dict_input/compare_dt4s_vs_original.py \
  --dt4s_dir eval_out/v4_dt4s \
  --dt1s_dir eval_out/v4_dt1s_original \
  --out_dir eval_out/v4_dt4s_vs_original

echo "[OK] Done. See eval_out/v4_dt4s/ and eval_out/v4_dt4s_vs_original/"
