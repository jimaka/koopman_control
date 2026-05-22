#!/usr/bin/env bash
# 服务器多卡 DDP 训练启动脚本（需 torchrun + 多 GPU）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

usage() {
  cat <<EOF
Usage:
  bash new_v4_dict_input/run_v4_ddp.sh [--nproc N] [-- extra train args]

Env:
  NPROC         GPU 数量（默认 nvidia-smi 检测）
  MASTER_PORT   DDP 主端口（默认 29500）
  CUDA_VISIBLE_DEVICES  可选，限定使用的 GPU

Examples:
  # 4 卡训练，每卡 batch=512
  bash new_v4_dict_input/run_v4_ddp.sh --nproc 4 -- --epochs 120 --run_tag v4_ddp

  # 使用全部可见 GPU
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash new_v4_dict_input/run_v4_ddp.sh -- --batch_size 384

  # 单卡（与 python3 train_v4_dict_input.py 等价，无需 torchrun）
  bash new_v4_dict_input/run_v4_ddp.sh --nproc 1 -- --smoketest
EOF
}

extra_args=()
while (($#)); do
  case "$1" in
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      usage
      exit 2
      ;;
  esac
done

if ((NPROC <= 1)); then
  echo "[INFO] NPROC=${NPROC}, single-process training"
  exec python3 new_v4_dict_input/train_v4_dict_input.py "${extra_args[@]}"
fi

echo "[INFO] DDP training: nproc=${NPROC} master_port=${MASTER_PORT}"
echo "[INFO] global_batch = batch_size × grad_accum_steps × ${NPROC}"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  new_v4_dict_input/train_v4_dict_input.py \
  "${extra_args[@]}"
