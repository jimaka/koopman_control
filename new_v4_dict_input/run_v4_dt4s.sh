#!/usr/bin/env bash
# v4 模型 4s 采样步长：训练 + 评估 + 与原 dt=1s 模型对比
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR="${CKPT_DIR:-checkpoints/v4_dt4s}"
CONFIG="${CONFIG:-new_v4_dict_input/configs/v4_dt4s_train.yaml}"
RESUME="${RESUME:-}"

echo "=== Train v4 @ dt=4s ==="
TRAIN_ARGS=(--config "$CONFIG" --ckpt_dir "$CKPT_DIR")
if [[ -n "$RESUME" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi
python3 new_v4_dict_input/train_v4_dict_input.py "${TRAIN_ARGS[@]}" "$@"

# 找到最新 run 目录下的 best/latest ckpt
RUN_DIR=$(find "$CKPT_DIR" -name koopman_v4_latest.pth | sort | tail -1 | xargs dirname)
BEST="$RUN_DIR/koopman_v4_best.pth"
if [[ ! -f "$BEST" ]]; then
  echo "WARN: no best.pth in $RUN_DIR, promoting latest.pth"
  python3 - <<PY
import torch
src = "$RUN_DIR/koopman_v4_latest.pth"
dst = "$RUN_DIR/koopman_v4_best.pth"
ck = torch.load(src, map_location="cpu", weights_only=False)
torch.save(ck, dst)
PY
fi
echo "Checkpoint: $BEST"

# 从配置推断 pred_len（默认 5 步 / 20s）
PRED_LEN=$(python3 - <<'PY'
import yaml, os
cfg = yaml.safe_load(open(os.environ.get("CONFIG", "new_v4_dict_input/configs/v4_dt4s_train.yaml")))
dt = float(cfg.get("dt", 4.0))
if cfg.get("pred_len_max"):
    print(int(cfg["pred_len_max"]))
else:
    print(max(1, int(round(float(cfg.get("pred_time_sec", 20.0)) / dt))))
PY
)

echo "=== Eval dt=4s pred_len=$PRED_LEN (GT vs Pred) ==="
python3 new_v4_dict_input/eval_v4_dict_input.py \
  --ckpt "$BEST" \
  --data data/koopman_test.npz \
  --pred_len "$PRED_LEN" --dt 4.0 \
  --out_dir "eval_out/v4_dt4s_${PRED_LEN}step" \
  --tag "v4_dt4s_${PRED_LEN}step"

echo "[OK] Done. See eval_out/v4_dt4s_${PRED_LEN}step/"
