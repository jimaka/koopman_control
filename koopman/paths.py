"""仓库路径常量：所有脚本默认从此解析 data / checkpoints。"""
from __future__ import annotations

from pathlib import Path

# koopman/paths.py -> 仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
# 注意：CKPT_DIR 必须指向 checkpoints 根目录。
# 历史上曾被误改为某次 v4 run 子目录（checkpoints/run_v4_xxx），导致
# CKPT_V1_BEST 等常量失效、train_v1/v2 默认把权重写进 v4 run 目录。
# v4 训练脚本会在 CKPT_DIR 下自动创建 run_v4_<timestamp>/ 子目录，无需在此硬编码。
CKPT_DIR = REPO_ROOT / "checkpoints"
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
