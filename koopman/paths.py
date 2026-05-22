"""仓库路径常量：所有脚本默认从此解析 data / checkpoints。"""
from __future__ import annotations

from pathlib import Path

# koopman/paths.py -> 仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
CKPT_DIR = REPO_ROOT / "checkpoints/run_v4_20260520_034545"
LOG_DIR = REPO_ROOT / "logs"
EVAL_OUT_DIR = REPO_ROOT / "eval_out"
CPP_MPC_DIR = REPO_ROOT / "cpp" / "koopman_mpc"

# 主数据集
TRAIN_MERGED = DATA_DIR / "koopman_train_merged.npz"
VAL = DATA_DIR / "koopman_val.npz"
TEST = DATA_DIR / "koopman_test.npz"
TRAIN = DATA_DIR / "koopman_train.npz"
TRAIN_LEFT_TURN = DATA_DIR / "koopman_train_left_turn.npz"
DATASET_V1 = DATA_DIR / "koopman_dataset_v1.npz"
SIM_10HZ = DATA_DIR / "sim_10HZ.npz"
TEST_SUPPLEMENT = DATA_DIR / "koopman_test_dataset.npz"

# 默认 checkpoint
CKPT_V1_BEST = CKPT_DIR / "koopman_v1_best.pth"
CKPT_V2_BEST = CKPT_DIR / "koopman_v2_best.pth"
CKPT_V3_BEST = CKPT_DIR / "koopman_v3_best.pth"
CKPT_V3A_BEST = CKPT_DIR / "koopman_v3a_best.pth"
CKPT_DEPLOY = CKPT_DIR / "koopman_best.pth"


def setup_repo() -> Path:
    """将仓库根加入 sys.path 并 chdir 到根目录（供 scripts/ 下 CLI 启动时调用）。"""
    import os
    import sys

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    return REPO_ROOT


ensure_repo_cwd = setup_repo  # 别名
