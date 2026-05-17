# Deep-Koopman 船舶速度跟踪

基于物理先验 Koopman 算子的多步速度预测、量化评估与 MPC 航迹跟踪。

**文档**：

- [docs/项目指南.md](docs/项目指南.md) — 项目总览与全流程
- [docs/MPC使用指南.md](docs/MPC使用指南.md) — MPC 航迹跟踪原理、用法与验证

## 仓库结构

```
.
├── README.md                 # 本文件（索引）
├── requirements.txt
├── koopman/                  # Python 库：模型、评估工具、路径常量
│   ├── model_v1_v2.py        # v1/v2 模型（5 阶字典）
│   ├── model_v3.py           # v3 模型（16 阶字典）
│   ├── evalkit.py            # 评估与 rollout 核心逻辑
│   ├── paths.py              # data/、checkpoints/ 等默认路径
│   └── mpc/                  # MPC 控制器
├── scripts/                  # 命令行入口（推荐从此运行）
│   ├── train_v2.py
│   ├── train_v1.py
│   ├── eval.py
│   ├── mpc_track.py
│   ├── reselect_v3a.py
│   └── data/                 # 数据集处理（rosbag → npz）
├── data/                     # 所有 .npz 数据集
├── checkpoints/              # 预训练权重
├── cpp/koopman_mpc/          # C++ MPC（ONNX Runtime）
│   └── scripts/              # PT→ONNX 导出与验证
├── docs/                     # 项目指南
├── logs/                     # 训练日志（gitignore）
└── eval_out/                 # 评估 / MPC 输出（gitignore）
```

## 快速开始

```bash
pip install -r requirements.txt

# 冒烟自测（约 1–2 分钟）
python3 scripts/train_v2.py --smoketest
python3 scripts/eval.py --smoketest
python3 scripts/mpc_track.py --smoketest

# 训练 v3a
python3 scripts/train_v2.py --model v3 --run_tag v3a --epochs 60 \
    --w_bias_v 100.0 --best_metric composite_v3a

# 评估
python3 scripts/eval.py --ckpt checkpoints/koopman_v3a_best.pth \
    --tag v3a --out_dir eval_out/v3a

# MPC 航迹跟踪
python3 scripts/mpc_track.py --segment 0 --steps 150 --out_dir eval_out/mpc
```

## C++ MPC（ONNX）

```bash
bash cpp/koopman_mpc/build.sh
# 导出 ONNX 并验证 PT/ONNX/C++ 精度
python3 cpp/koopman_mpc/scripts/export_onnx.py
python3 cpp/koopman_mpc/scripts/verify_pipeline.py
```

见 [cpp/koopman_mpc/README.md](cpp/koopman_mpc/README.md)。
