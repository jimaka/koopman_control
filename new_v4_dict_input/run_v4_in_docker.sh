#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for v4 dict-input training in Docker.
# Default container: koopman_latest_sm120_martin

CONTAINER_NAME="${CONTAINER_NAME:-koopman_latest_sm120_martin}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST_NEW_DIR="${HOST_REPO_ROOT}/new_v4_dict_input"

usage() {
  cat <<'EOF'
Usage:
  bash new_v4_dict_input/run_v4_in_docker.sh [mode] [-- extra train args]

Modes:
  --smoketest      Run smoketest
  --help-train     Show train script help
  --train          Run normal training (default, single GPU)
  --ddp            Multi-GPU DDP via torchrun (pass --nproc N before --)

Examples:
  bash new_v4_dict_input/run_v4_in_docker.sh --smoketest
  bash new_v4_dict_input/run_v4_in_docker.sh --help-train
  bash new_v4_dict_input/run_v4_in_docker.sh --train -- --epochs 60 --run_tag v4_dict_input
  bash new_v4_dict_input/run_v4_in_docker.sh --ddp --nproc 4 -- --epochs 120 --batch_size 512

Env:
  CONTAINER_NAME   Docker container name (default: koopman_latest_sm120_martin)
EOF
}

mode="train"
nproc=""
extra_args=()
while (($#)); do
  case "$1" in
    --smoketest)
      mode="smoketest"
      shift
      ;;
    --help-train)
      mode="help"
      shift
      ;;
    --train)
      mode="train"
      shift
      ;;
    --ddp)
      mode="ddp"
      shift
      ;;
    --nproc)
      nproc="$2"
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

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker command not found."
  exit 1
fi

if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "[ERROR] container not found: ${CONTAINER_NAME}"
  exit 1
fi

running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")"
if [[ "${running}" != "true" ]]; then
  echo "[ERROR] container is not running: ${CONTAINER_NAME}"
  echo "Run: docker start ${CONTAINER_NAME}"
  exit 1
fi

resolve_container_repo_root() {
  docker exec "${CONTAINER_NAME}" bash -lc '
set -e
candidates=(
  "/workspace"
  "/workspace/koopman_control"
  "/root/workspace"
  "/app"
)
for p in "${candidates[@]}"; do
  if [[ -f "${p}/new_v4_dict_input/train_v4_dict_input.py" ]]; then
    echo "${p}"
    exit 0
  fi
done
if [[ -d "/workspace" ]]; then
  echo "/workspace"
  exit 0
fi
exit 1
'
}

CONTAINER_REPO_ROOT="$(resolve_container_repo_root || true)"
if [[ -z "${CONTAINER_REPO_ROOT}" ]]; then
  echo "[ERROR] failed to detect repo root inside container ${CONTAINER_NAME}"
  exit 1
fi

if ! docker exec "${CONTAINER_NAME}" bash -lc "[[ -f \"${CONTAINER_REPO_ROOT}/new_v4_dict_input/train_v4_dict_input.py\" ]]"; then
  echo "[INFO] sync new_v4_dict_input -> ${CONTAINER_NAME}:${CONTAINER_REPO_ROOT}/new_v4_dict_input"
  docker cp "${HOST_NEW_DIR}" "${CONTAINER_NAME}:${CONTAINER_REPO_ROOT}/new_v4_dict_input"
fi

run_cmd=(python3 new_v4_dict_input/train_v4_dict_input.py)
case "${mode}" in
  smoketest) run_cmd+=(--smoketest) ;;
  help) run_cmd+=(--help) ;;
  train) ;;
  ddp)
    if [[ -z "${nproc}" ]]; then
      nproc="$(docker exec "${CONTAINER_NAME}" bash -lc 'nvidia-smi -L 2>/dev/null | wc -l' || echo 2)"
    fi
    run_cmd=(
      torchrun --standalone --nproc_per_node="${nproc}" --master_port="${MASTER_PORT:-29500}"
      new_v4_dict_input/train_v4_dict_input.py
    )
    ;;
esac
if ((${#extra_args[@]})); then
  run_cmd+=("${extra_args[@]}")
fi

echo "[INFO] container=${CONTAINER_NAME}"
echo "[INFO] repo_root=${CONTAINER_REPO_ROOT}"
echo "[INFO] running: ${run_cmd[*]}"

docker exec -it "${CONTAINER_NAME}" bash -lc "
set -e
cd \"${CONTAINER_REPO_ROOT}\"
${run_cmd[*]}
"
