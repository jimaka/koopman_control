#!/usr/bin/env bash
# =============================================================================
# 在 Docker 容器内**直接运行**的 v4 C++ OSQP-MPC 编译脚本。
#
# 用法（已在容器内 / 已 docker exec 进入后）：
#   bash cpp/koopman_mpc/build_in_docker.sh [选项]
#
# 默认行为：检测依赖(按需安装) → 下载 ONNX Runtime(缺失时) → CMake 构建库+demo+验证工具。
#
# 选项：
#   --weights CKPT   额外用该 ckpt 导出 latent YAML(+ONNX plant) 到 weights/
#   --smoketest      构建后跑 MPC 冒烟（需先有 weights，可配合 --weights）
#   --skip-deps      跳过依赖安装（仅检测并告警）
#   --jobs N         并行编译数（默认 nproc）
#   -h, --help       显示帮助
#
# 说明：
#   - 仅“直接编译”时无需 Python/ckpt；--weights / --smoketest 才需要 torch/onnx 与 ckpt。
#   - OSQP 由 CMake FetchContent 自动拉取（需 git + 网络）。
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPP_DIR="$ROOT/cpp/koopman_mpc"
ORT_VERSION="1.26.0"
ORT_DIR="$CPP_DIR/third_party/onnxruntime"
ORT_TGZ="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TGZ}"

CKPT=""
DO_SMOKETEST=0
SKIP_DEPS=0
JOBS="$(nproc 2>/dev/null || echo 4)"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"; }

while (($#)); do
  case "$1" in
    --weights) CKPT="$2"; shift 2 ;;
    --smoketest) DO_SMOKETEST=1; shift ;;
    --skip-deps) SKIP_DEPS=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 未知参数: $1"; usage; exit 2 ;;
  esac
done

cd "$CPP_DIR"

# ----------------------------------------------------------------------------
# 1) 检测依赖（仅缺失才安装）
# ----------------------------------------------------------------------------
echo ">>> [1/4] 检测构建依赖..."
need_python=$([[ -n "$CKPT" || "$DO_SMOKETEST" -eq 1 ]] && echo 1 || echo 0)

deps_ok=1
command -v g++ >/dev/null 2>&1 && echo 'int main(){}' | g++ -x c++ - -o /tmp/_gxx_t 2>/dev/null && rm -f /tmp/_gxx_t || { echo "  缺少 g++"; deps_ok=0; }
command -v cmake >/dev/null 2>&1 || { echo "  缺少 cmake"; deps_ok=0; }
command -v curl  >/dev/null 2>&1 || { echo "  缺少 curl"; deps_ok=0; }
command -v git   >/dev/null 2>&1 || { echo "  缺少 git (OSQP FetchContent 需要)"; deps_ok=0; }
if [[ "$need_python" -eq 1 ]]; then
  python3 -c "import torch, onnx, onnxruntime, onnxscript, yaml, numpy" 2>/dev/null \
    || { echo "  缺少 python 依赖 (torch/onnx/onnxruntime/onnxscript/pyyaml/numpy)"; deps_ok=0; }
fi

if [[ "$SKIP_DEPS" -eq 1 ]]; then
  echo "    --skip-deps：跳过安装（依赖齐全=${deps_ok}）"
elif [[ "$deps_ok" -eq 0 ]]; then
  echo "    安装缺失依赖..."
  SUDO=""; command -v sudo >/dev/null 2>&1 && SUDO="sudo"
  if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -qq || true
    $SUDO apt-get install -y -qq build-essential g++ cmake curl git || true
  fi
  if [[ "$need_python" -eq 1 ]]; then
    python3 - <<'PY' || true
import importlib.util, subprocess, sys
pkgs = {"torch":"torch","onnx":"onnx","onnxruntime":"onnxruntime",
        "onnxscript":"onnxscript","yaml":"pyyaml","numpy":"numpy"}
miss = [pip for mod, pip in pkgs.items() if importlib.util.find_spec(mod) is None]
if miss:
    print("pip install:", miss)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *miss], check=False)
else:
    print("python deps OK")
PY
  fi
else
  echo "    依赖齐全。"
fi

# ----------------------------------------------------------------------------
# 2) ONNX Runtime（缺失时下载）
# ----------------------------------------------------------------------------
echo ">>> [2/4] 准备 ONNX Runtime ${ORT_VERSION}..."
if [[ ! -f "$ORT_DIR/include/onnxruntime_cxx_api.h" ]]; then
  echo "    下载 ORT..."
  mkdir -p third_party
  tmp="$(mktemp -d)"
  curl -fsSL "$ORT_URL" -o "$tmp/$ORT_TGZ"
  tar -xzf "$tmp/$ORT_TGZ" -C "$tmp"
  rm -rf "$ORT_DIR"
  mv "$tmp/onnxruntime-linux-x64-${ORT_VERSION}" "$ORT_DIR"
  rm -rf "$tmp"
else
  echo "    已存在，跳过下载。"
fi

# ----------------------------------------------------------------------------
# 3) （可选）导出权重
# ----------------------------------------------------------------------------
if [[ -n "$CKPT" ]]; then
  echo ">>> [3/4] 导出 latent YAML + ONNX plant（ckpt=$CKPT）..."
  python3 "$ROOT/new_v4_dict_input/export_v4_encode_weights.py" \
    --ckpt "$CKPT" --horizon 20 \
    --out "$CPP_DIR/weights/koopman_v4_latent.json" \
    --out-yaml "$CPP_DIR/weights/koopman_v4_latent.yaml"
  python3 "$ROOT/new_v4_dict_input/export_v4_onnx.py" \
    --ckpt "$CKPT" --out_dir "$CPP_DIR/weights" \
    --pred_len 20 --dt 1.0 --write_rollout_check --skip_test_compare
else
  echo ">>> [3/4] 跳过权重导出（未传 --weights；仅编译 C++）。"
fi

# ----------------------------------------------------------------------------
# 4) CMake 配置 + 构建
# ----------------------------------------------------------------------------
echo ">>> [4/4] CMake 配置与构建（-j${JOBS}）..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
  -DONNXRUNTIME_ROOT="$ORT_DIR"
cmake --build . -j"$JOBS"

echo ">>> 构建完成：$(ls -1 koopman_mpc_cpp 2>/dev/null || echo '(见 build/ 目录)')"

if [[ "$DO_SMOKETEST" -eq 1 ]]; then
  echo ">>> 运行 MPC 冒烟..."
  export LD_LIBRARY_PATH="$ORT_DIR/lib:${LD_LIBRARY_PATH:-}"
  ./koopman_mpc_cpp --smoketest \
    --config "$ROOT/cpp/koopman_mpc/src/mpc_config.yaml" \
    --ref "$ROOT/cpp/koopman_mpc/weights/cpp_test_ref.json" \
    --steps 10
fi

echo ">>> 完成。"
