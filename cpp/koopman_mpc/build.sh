#!/usr/bin/env bash
# 构建 C++ Koopman MPC（依赖 pip torch 提供的 LibTorch）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$(dirname "$0")"

# 重新导出（若已存在可跳过以加快迭代）
echo ">>> Export TorchScript weights..."
python3 scripts/export_torchscript.py \
    --ckpt "$ROOT/checkpoints/koopman_v3a_best.pth" \
    --out_dir "$ROOT/cpp/koopman_mpc/weights"

python3 scripts/export_cpp_test_ref.py

echo ">>> CMake configure & build..."
TORCH_CMAKE="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" -DCMAKE_CXX_COMPILER=g++
cmake --build . -j"$(nproc)"

echo ">>> Verify rollout (C++ vs Python)..."
python3 "$ROOT/cpp/koopman_mpc/scripts/write_rollout_check_txt.py"
./verify_rollout "$ROOT/cpp/koopman_mpc/weights/koopman_rollout.pt" \
    "$ROOT/cpp/koopman_mpc/weights/rollout_check.npz"

echo ">>> Run MPC smoketest..."
./koopman_mpc_cpp --smoketest \
    --weights "$ROOT/cpp/koopman_mpc/weights" \
    --ref "$ROOT/cpp/koopman_mpc/weights/cpp_test_ref.json"

echo ">>> All C++ checks passed."
