#!/usr/bin/env bash
# =============================================================================
# 仅编译 MPC（OSQP）核心 —— 在 Docker 容器内直接运行，**不依赖 ONNX**。
#
# 与 build_v4.sh / build_in_docker.sh 的区别：
#   - 不下载 ONNX Runtime（规避 curl 56 等网络问题），不编译 ONNX plant / demo；
#   - 只构建 koopman_control 库（Tier-1/Tier-2 OSQP 求解）+ verify 工具；
#   - 依赖仅：g++ / cmake / git(OSQP FetchContent) / yaml-cpp。
#
# 用法（容器内、仓库根目录）：
#   bash cpp/koopman_control/build_mpc_only.sh [选项]
#
# 选项：
#   --yaml PATH    用于跑验证的 latent YAML（默认 cpp/koopman_mpc/weights/koopman_v4_latent.yaml）
#   --no-run       只编译，不运行 verify_pose_linearize
#   --skip-deps    跳过依赖安装（仅检测并告警）
#   --jobs N       并行编译数（默认 nproc）
#   -h, --help     显示帮助
# =============================================================================
set -euo pipefail

CONTROL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${CONTROL_DIR}/../.." && pwd)"
YAML="${ROOT}/cpp/koopman_mpc/weights/koopman_v4_latent.yaml"
DO_RUN=1
SKIP_DEPS=0
JOBS="$(nproc 2>/dev/null || echo 4)"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"; }

while (($#)); do
  case "$1" in
    --yaml) YAML="$2"; shift 2 ;;
    --no-run) DO_RUN=0; shift ;;
    --skip-deps) SKIP_DEPS=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 未知参数: $1"; usage; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------
# 1) 依赖检测（仅缺失才安装）—— 无需 Python / ONNX
# ----------------------------------------------------------------------------
echo ">>> [1/3] 检测依赖（g++ / cmake / git / yaml-cpp）..."
deps_ok=1
command -v g++ >/dev/null 2>&1 && echo 'int main(){}' | g++ -x c++ - -o /tmp/_gxx_t 2>/dev/null && rm -f /tmp/_gxx_t || { echo "  缺少 g++"; deps_ok=0; }
command -v cmake >/dev/null 2>&1 || { echo "  缺少 cmake"; deps_ok=0; }
command -v git   >/dev/null 2>&1 || { echo "  缺少 git (OSQP FetchContent 需要)"; deps_ok=0; }
# yaml-cpp：头文件或 pkg-config 任一可见即可
if ! { [[ -f /usr/include/yaml-cpp/yaml.h ]] || [[ -f /usr/local/include/yaml-cpp/yaml.h ]] || pkg-config --exists yaml-cpp 2>/dev/null; }; then
  echo "  缺少 yaml-cpp 开发包"; deps_ok=0
fi

if [[ "$SKIP_DEPS" -eq 1 ]]; then
  echo "    --skip-deps：跳过安装（依赖齐全=${deps_ok}）"
elif [[ "$deps_ok" -eq 0 ]]; then
  echo "    安装缺失依赖..."
  SUDO=""; command -v sudo >/dev/null 2>&1 && SUDO="sudo"
  if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -qq || true
    $SUDO apt-get install -y -qq build-essential g++ cmake git libyaml-cpp-dev pkg-config || true
  else
    echo "    [WARN] 无 apt-get，请手动安装 g++ / cmake / git / yaml-cpp 后重试。"
  fi
else
  echo "    依赖齐全。"
fi

# ----------------------------------------------------------------------------
# 2) CMake 配置 + 构建（KOOPMAN_ENABLE_ONNX=OFF）
# ----------------------------------------------------------------------------
echo ">>> [2/3] 构建 MPC 核心（-DKOOPMAN_ENABLE_ONNX=OFF, -j${JOBS}）..."
cd "$CONTROL_DIR"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
  -DKOOPMAN_ENABLE_ONNX=OFF
cmake --build . -j"$JOBS"
echo "    产物: $(ls -1 libkoopman_control.a verify_latent_qp verify_pose_linearize 2>/dev/null | tr '\n' ' ')"

# ----------------------------------------------------------------------------
# 3) 运行 MPC 验证（OSQP，无 ONNX）
# ----------------------------------------------------------------------------
if [[ "$DO_RUN" -eq 1 ]]; then
  if [[ -f "$YAML" ]]; then
    echo ">>> [3/3] 运行 verify_pose_linearize（Tier-2 + OSQP 端到端）..."
    ./verify_pose_linearize "$YAML"
  else
    echo ">>> [3/3] 跳过运行：未找到 latent YAML ($YAML)"
    echo "    生成方式：python3 new_v4_dict_input/export_v4_encode_weights.py --ckpt <ckpt> --out-yaml $YAML"
  fi
else
  echo ">>> [3/3] --no-run：仅编译完成。"
fi

echo ">>> 完成（仅 MPC 核心，无 ONNX）。"
