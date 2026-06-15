# Deep-Koopman 船舶速度跟踪

基于物理先验 Koopman 算子的多步速度预测、量化评估与 MPC 航迹跟踪。

**文档**：

- [docs/工程分析与优化.md](docs/工程分析与优化.md) — 工程分析、可优化点与「基线 vs 优化」对照训练结果
- [docs/项目指南.md](docs/项目指南.md) — 项目总览与全流程
- [docs/训练流程指南.md](docs/训练流程指南.md) — 训练流程详解、性能优化说明与新参数（早停 / AMP / 断点续训 / DDP）
- [docs/MPC使用指南.md](docs/MPC使用指南.md) — OSQP 潜空间 MPC 快速上手
- [docs/潜空间QP-MPC实现.md](docs/潜空间QP-MPC实现.md) — QP 推导与 C++ 模块说明
- [cpp/koopman_control/模型输入输出接口说明.md](cpp/koopman_control/模型输入输出接口说明.md) — v4 ONNX / MPC 接口（中文）
- [cpp/koopman_control/README_CN.md](cpp/koopman_control/README_CN.md) — C++ 库与 motion 集成

## 仓库结构

```
.
├── README.md                 # 本文件（索引）
├── requirements.txt
├── koopman/                  # Python 库：模型、评估工具、路径常量
│   ├── model_v1_v2.py        # v1/v2 模型（5 阶字典）
│   ├── model_v3.py           # v3 模型（16 阶字典）
│   ├── evalkit.py            # 评估与 rollout 核心逻辑
│   ├── export/               # 可导出 rollout（ONNX / TorchScript）
│   ├── paths.py              # data/、checkpoints/ 等默认路径
│   └── mpc/                  # MPC 数据工具（参考构造、指标；求解器在 C++）
├── scripts/                  # 命令行入口（推荐从此运行）
│   ├── train_v2.py
│   ├── train_v1.py
│   ├── eval.py
│   ├── reselect_v3a.py
│   └── data/                 # 数据集处理（rosbag → npz）
├── data/                     # 所有 .npz 数据集
├── checkpoints/              # 预训练权重
├── cpp/koopman_control/      # C++ OSQP MPC 控制库（v4 主推，motion 桥接）
├── cpp/koopman_mpc/          # C++ demo / 构建脚本（ONNX Runtime）
├── new_v4_dict_input/        # v4 训练、导出、ONNX benchmark
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

# 训练 v3a
python3 scripts/train_v2.py --model v3 --run_tag v3a --epochs 60 \
    --w_bias_v 100.0 --best_metric composite_v3a

# 评估
python3 scripts/eval.py --ckpt checkpoints/koopman_v3a_best.pth \
    --tag v3a --out_dir eval_out/v3a
```

## C++ MPC（OSQP / v4）

MPC 求解**仅保留 C++ OSQP 潜空间路径**（无 Python Adam MPC）。

```bash
# 1. 导出潜空间权重（MPC 优化用）
python3 new_v4_dict_input/export_v4_encode_weights.py \
  --ckpt checkpoints/run_v4_20260520_034545/koopman_v4_best.pth \
  --horizon 20 \
  --out cpp/koopman_mpc/weights/koopman_v4_latent.yaml

# 2. 导出 ONNX（闭环仿真 plant，可选）
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/run_v4_20260520_034545/koopman_v4_best.pth \
  --pred_len 200 --out_dir cpp/koopman_mpc/weights

# 3. 编译与冒烟
bash cpp/koopman_mpc/build_v4.sh
export LD_LIBRARY_PATH=cpp/koopman_mpc/third_party/onnxruntime/lib:$LD_LIBRARY_PATH
./cpp/koopman_mpc/build/koopman_mpc_cpp \
  --config cpp/koopman_control/config/mpc_config.yaml --smoketest
```

见 [docs/MPC使用指南.md](docs/MPC使用指南.md)、[cpp/koopman_mpc/README.md](cpp/koopman_mpc/README.md) 与 [cpp/koopman_control/config/mpc_config.yaml](cpp/koopman_control/config/mpc_config.yaml)。
