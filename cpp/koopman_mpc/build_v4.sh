#!/usr/bin/env bash
# v4 20s 模型（dt=1s, H=20）：导出 ONNX + C++ 参考航迹 + 编译 + 验证
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CPP_DIR="$(cd "$(dirname "$0")" && pwd)"
ORT_VERSION="1.26.0"
ORT_DIR="$CPP_DIR/third_party/onnxruntime"
ORT_TGZ="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_URL="${ORT_URL:-https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TGZ}}"

CKPT="${1:-$ROOT/checkpoints/run_v4_20260520_034545/koopman_v4_best.pth}"
PRED_LEN="${2:-20}"
MODEL_DT="${3:-1.0}"

cd "$CPP_DIR"
bash "$CPP_DIR/scripts/setup_cloud_deps.sh"

if [[ ! -f "$ORT_DIR/include/onnxruntime_cxx_api.h" ]]; then
    echo ">>> Download ONNX Runtime C++ ${ORT_VERSION}..."
    mkdir -p third_party
    tmp="$(mktemp -d)"
    # 支持本地离线包 ORT_TGZ_PATH / 镜像 ORT_URL；自动重试 + 断点续传缓解 curl 56 断流
    if [[ -n "${ORT_TGZ_PATH:-}" && -f "${ORT_TGZ_PATH}" ]]; then
        cp "$ORT_TGZ_PATH" "$tmp/$ORT_TGZ"
    else
        curl -fL --retry 5 --retry-delay 3 --retry-all-errors --connect-timeout 20 -C - \
            "${ORT_URL}" -o "$tmp/$ORT_TGZ"
    fi
    tar -xzf "$tmp/$ORT_TGZ" -C "$tmp"
    rm -rf "$ORT_DIR"
    mv "$tmp/onnxruntime-linux-x64-${ORT_VERSION}" "$ORT_DIR"
    rm -rf "$tmp"
fi

echo ">>> Export v4 latent weights (OSQP MPC: Ā/B + encoder + decoder)..."
python3 "$ROOT/new_v4_dict_input/export_v4_encode_weights.py" \
    --ckpt "$CKPT" \
    --horizon "$PRED_LEN" \
    --out "$ROOT/cpp/koopman_mpc/weights/koopman_v4_latent.json" \
    --out-yaml "$ROOT/cpp/koopman_mpc/weights/koopman_v4_latent.yaml"

echo ">>> Export v4 ONNX plant (pred_len=${PRED_LEN}, dt=${MODEL_DT})..."
python3 "$ROOT/new_v4_dict_input/export_v4_onnx.py" \
    --ckpt "$CKPT" \
    --out_dir "$ROOT/cpp/koopman_mpc/weights" \
    --pred_len "$PRED_LEN" \
    --dt "$MODEL_DT" \
    --write_rollout_check \
    --skip_test_compare

echo ">>> Export v4 C++ test ref..."
python3 "$ROOT/cpp/koopman_mpc/scripts/export_v4_cpp_test_ref.py" \
    --ckpt "$CKPT" \
    --horizon "$PRED_LEN" \
    --dt "$MODEL_DT"

echo ">>> CMake configure & build..."
# 清理陈旧/跨机器的 CMake 缓存
if [[ -f build/CMakeCache.txt ]]; then
    cached_home="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' build/CMakeCache.txt | head -1)"
    if [[ -n "$cached_home" && "$cached_home" != "$CPP_DIR" ]]; then
        echo ">>> stale CMakeCache (home=$cached_home), wiping build/"
        rm -rf build
    fi
fi
mkdir -p build
cd build
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

echo ">>> Run MPC smoketest (ONNX H=${PRED_LEN})..."
./koopman_mpc_cpp --smoketest \
    --config "$ROOT/cpp/koopman_mpc/src/mpc_config.yaml" \
    --weights "$ROOT/cpp/koopman_mpc/weights" \
    --ref "$ROOT/cpp/koopman_mpc/weights/cpp_test_ref.json" \
    --steps 10

echo ">>> v4 C++ / ONNX checks passed."
