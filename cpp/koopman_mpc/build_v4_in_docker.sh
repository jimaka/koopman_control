#!/usr/bin/env bash
# =============================================================================
# 在本地 Docker 容器内编译 v4 C++ OSQP-MPC（参考 build_v4.sh）。
#
# 流程：
#   1) 检测本地 Docker 环境（CLI / 容器存在 / 运行中）
#   2) 在容器内检测构建依赖（g++ / cmake / curl / python:torch,onnx...）
#      —— 仅当缺失时才安装（--force-deps 强制，--skip-deps 跳过）
#   3) 在容器内执行 build_v4.sh（导出权重 + CMake 构建 + 验证 + 冒烟）
#
# 用法：
#   bash cpp/koopman_mpc/build_v4_in_docker.sh [选项]
#
# 选项：
#   --container NAME   容器名（默认 $CONTAINER_NAME 或 koopman_latest_sm120_martin）
#   --ckpt PATH        容器内 ckpt 路径（默认沿用 build_v4.sh 内默认）
#   --pred_len N       ONNX/MPC horizon（默认 20）
#   --dt X             模型步长（默认 1.0）
#   --sync             若容器内缺少源码，则 docker cp 同步 cpp/ 与 new_v4_dict_input/
#   --force-deps       不论检测结果都安装依赖
#   --skip-deps        跳过依赖安装（仅检测并告警）
#   -h, --help         显示帮助
#
# 环境变量：
#   CONTAINER_NAME     同 --container
# =============================================================================
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-koopman_latest_sm120_martin}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT=""
PRED_LEN="20"
MODEL_DT="1.0"
DO_SYNC=0
FORCE_DEPS=0
SKIP_DEPS=0

usage() {
  # 打印文件头部注释块（第 2 行起，遇到首个非 # 行停止）
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

while (($#)); do
  case "$1" in
    --container) CONTAINER_NAME="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --pred_len) PRED_LEN="$2"; shift 2 ;;
    --dt) MODEL_DT="$2"; shift 2 ;;
    --sync) DO_SYNC=1; shift ;;
    --force-deps) FORCE_DEPS=1; shift ;;
    --skip-deps) SKIP_DEPS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 未知参数: $1"; usage; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------
# 1) 检测本地 Docker 环境
# ----------------------------------------------------------------------------
echo ">>> [1/3] 检测本地 Docker 环境..."
if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] 未找到 docker 命令，请先安装 Docker。"
  exit 1
fi
if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "[ERROR] 容器不存在: ${CONTAINER_NAME}"
  echo "        可用 --container NAME 指定，或先创建/启动容器。"
  exit 1
fi
running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || echo false)"
if [[ "${running}" != "true" ]]; then
  echo "[ERROR] 容器未运行: ${CONTAINER_NAME}（执行: docker start ${CONTAINER_NAME}）"
  exit 1
fi
echo "    容器就绪: ${CONTAINER_NAME}"

# 解析容器内仓库根（以 build_v4.sh 是否存在为准）
resolve_repo_root() {
  docker exec "${CONTAINER_NAME}" bash -lc '
set -e
for p in "/workspace" "/workspace/koopman_control" "/root/workspace" "/app"; do
  if [[ -f "${p}/cpp/koopman_mpc/build_v4.sh" ]]; then echo "${p}"; exit 0; fi
done
[[ -d "/workspace" ]] && { echo "/workspace"; exit 0; }
exit 1
'
}
REPO_ROOT="$(resolve_repo_root || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  if [[ "${DO_SYNC}" -eq 1 ]]; then
    REPO_ROOT="/workspace"   # 将在下方 --sync 步骤把源码复制到此处
  else
    echo "[ERROR] 无法在容器内定位仓库根（未找到 cpp/koopman_mpc/build_v4.sh）。"
    echo "        使用 --sync 同步源码，或确认仓库已挂载到容器。"
    exit 1
  fi
fi
echo "    容器内仓库根: ${REPO_ROOT}"

# 可选：同步源码（缺少 build_v4.sh 时）
if [[ "${DO_SYNC}" -eq 1 ]]; then
  if ! docker exec "${CONTAINER_NAME}" bash -lc "[[ -f '${REPO_ROOT}/cpp/koopman_mpc/build_v4.sh' ]]"; then
    echo "    [sync] 复制 cpp/ 与 new_v4_dict_input/ 到容器 ${REPO_ROOT}"
    docker exec "${CONTAINER_NAME}" bash -lc "mkdir -p '${REPO_ROOT}/cpp'"
    docker cp "${HOST_REPO_ROOT}/cpp/." "${CONTAINER_NAME}:${REPO_ROOT}/cpp/"
    docker cp "${HOST_REPO_ROOT}/new_v4_dict_input" "${CONTAINER_NAME}:${REPO_ROOT}/new_v4_dict_input"
  fi
fi

# ----------------------------------------------------------------------------
# 2) 在容器内检测依赖，按需安装
# ----------------------------------------------------------------------------
echo ">>> [2/3] 检测容器内构建依赖..."
# 返回码：0=依赖齐全；1=有缺失
docker exec "${CONTAINER_NAME}" bash -lc '
missing=0
command -v g++   >/dev/null 2>&1 && echo "int main(){}" | g++ -x c++ - -o /tmp/_gxx_t 2>/dev/null && rm -f /tmp/_gxx_t || { echo "  缺少: g++"; missing=1; }
command -v cmake >/dev/null 2>&1 || { echo "  缺少: cmake"; missing=1; }
command -v curl  >/dev/null 2>&1 || { echo "  缺少: curl"; missing=1; }
command -v git   >/dev/null 2>&1 || { echo "  缺少: git(OSQP FetchContent 需要)"; missing=1; }
python3 -c "import torch, onnx, onnxruntime, onnxscript, yaml, numpy" 2>/dev/null || { echo "  缺少: python 依赖(torch/onnx/onnxruntime/onnxscript/pyyaml/numpy)"; missing=1; }
exit $missing
' && DEPS_OK=1 || DEPS_OK=0

if [[ "${SKIP_DEPS}" -eq 1 ]]; then
  echo "    --skip-deps：跳过安装（依赖齐全=${DEPS_OK}）"
elif [[ "${FORCE_DEPS}" -eq 1 || "${DEPS_OK}" -eq 0 ]]; then
  echo "    安装缺失依赖（force=${FORCE_DEPS}）..."
  docker exec "${CONTAINER_NAME}" bash -lc '
set -e
SUDO=""; command -v sudo >/dev/null 2>&1 && SUDO="sudo"
if command -v apt-get >/dev/null 2>&1; then
  $SUDO apt-get update -qq || true
  $SUDO apt-get install -y -qq build-essential g++ cmake curl git || true
fi
# Python 依赖：仅安装缺失项
python3 - <<"PY" || true
import importlib, subprocess, sys
pkgs = {"torch":"torch","onnx":"onnx","onnxruntime":"onnxruntime","onnxscript":"onnxscript","yaml":"pyyaml","numpy":"numpy"}
miss = [pip for mod, pip in pkgs.items() if importlib.util.find_spec(mod) is None]
if miss:
    print("pip install:", miss)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *miss], check=False)
else:
    print("python deps OK")
PY
'
else
  echo "    依赖齐全，跳过安装。"
fi

# ----------------------------------------------------------------------------
# 3) 在容器内执行 build_v4.sh
# ----------------------------------------------------------------------------
echo ">>> [3/3] 容器内编译（build_v4.sh）..."
BUILD_ARGS=""
[[ -n "${CKPT}" ]] && BUILD_ARGS="${CKPT} ${PRED_LEN} ${MODEL_DT}"

docker exec -i "${CONTAINER_NAME}" bash -lc "
set -e
cd '${REPO_ROOT}'
bash cpp/koopman_mpc/build_v4.sh ${BUILD_ARGS}
"

echo ">>> 完成：v4 C++ OSQP-MPC 已在容器 ${CONTAINER_NAME} 内编译并通过冒烟。"
