#!/usr/bin/env bash
# 构建 C++ Koopman MPC（ONNX Runtime + 导出权重 + 全流程验证）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CPP_DIR="$(cd "$(dirname "$0")" && pwd)"
ORT_VERSION="1.26.0"
ORT_DIR="$CPP_DIR/third_party/onnxruntime"
ORT_TGZ="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TGZ}"

cd "$CPP_DIR"

bash "$CPP_DIR/scripts/setup_cloud_deps.sh"

if [[ ! -f "$ORT_DIR/include/onnxruntime_cxx_api.h" ]]; then
    echo ">>> Download ONNX Runtime C++ ${ORT_VERSION}..."
    mkdir -p third_party
    tmp="$(mktemp -d)"
    curl -fsSL "$ORT_URL" -o "$tmp/$ORT_TGZ"
    tar -xzf "$tmp/$ORT_TGZ" -C "$tmp"
    rm -rf "$ORT_DIR"
    mv "$tmp/onnxruntime-linux-x64-${ORT_VERSION}" "$ORT_DIR"
    rm -rf "$tmp"
fi

echo ">>> Install Python export deps..."
pip install -q onnx onnxruntime onnxscript

echo ">>> Export ONNX weights + accuracy check..."
python3 scripts/export_onnx.py \
    --ckpt "$ROOT/checkpoints/koopman_v3a_best.pth" \
    --out_dir "$ROOT/cpp/koopman_mpc/weights"

python3 scripts/export_cpp_test_ref.py

echo ">>> CMake configure & build..."
mkdir -p build
cd build
# 云端默认 c++ 可能指向 clang 且缺少 libstdc++ 链接；优先 g++
CXX_COMPILER="${CXX:-g++}"
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
    -DONNXRUNTIME_ROOT="$ORT_DIR"
cmake --build . -j"$(nproc)"

echo ">>> Verify rollout (C++ ONNX vs Python)..."
python3 "$ROOT/cpp/koopman_mpc/scripts/write_rollout_check_txt.py"
export LD_LIBRARY_PATH="$ORT_DIR/lib:${LD_LIBRARY_PATH:-}"
./verify_rollout "$ROOT/cpp/koopman_mpc/weights/koopman_rollout.onnx" \
    "$ROOT/cpp/koopman_mpc/weights/rollout_check.npz"

echo ">>> Run MPC smoketest (ONNX)..."
./koopman_mpc_cpp --smoketest \
    --weights "$ROOT/cpp/koopman_mpc/weights" \
    --ref "$ROOT/cpp/koopman_mpc/weights/cpp_test_ref.json"

echo ">>> Python pipeline re-check..."
python3 "$ROOT/cpp/koopman_mpc/scripts/verify_pipeline.py" \
    --ckpt "$ROOT/checkpoints/koopman_v3a_best.pth"

echo ">>> All C++ / ONNX checks passed."
