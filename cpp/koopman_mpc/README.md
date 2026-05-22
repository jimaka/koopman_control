# Koopman MPC — C++ 实现（ONNX Runtime）

> **控制库已独立至 [`../koopman_control/`](../koopman_control/README_CN.md)**（含 motion.cpp 对接、YAML 配置与 [`模型输入输出接口说明.md`](../koopman_control/模型输入输出接口说明.md)）。  
> 本目录保留 demo 程序、构建脚本与 ONNX 权重。

基于 **ONNX Runtime** 的船舶航迹跟踪 MPC，动力学 rollout 与 Python 版 `koopman/export/rollout.py` 对齐。

## 目录结构

```
cpp/koopman_mpc/
├── CMakeLists.txt           # 链接 koopman_control 库
├── build.sh                 # v3：下载 ORT、导出 H=20 ONNX、编译、验证
├── build_v4.sh              # v4：H=200 ONNX + 编译（推荐）
├── include/                 # 兼容头文件（转发至 koopman_control）
├── src/
│   ├── main.cpp             # demo：koopman_mpc_cpp
│   └── mpc_config.yaml      # demo 用配置副本
├── tools/verify_rollout.cpp
├── scripts/
│   ├── export_onnx.py       # v3 .pth → .onnx
│   └── verify_pipeline.py
├── third_party/onnxruntime/ # build.sh 自动下载（gitignore）
└── weights/                 # koopman_rollout.onnx 等（gitignore）
```

## 依赖

- C++17（g++）、CMake ≥ 3.18
- **ONNX Runtime** C++、**yaml-cpp**（MPC 配置）
- Python 3 + `torch`、`onnx`、`onnxruntime`、`onnxscript`（导出与验证）

> **Adam 优化器**在 `koopman_control` 源码内实现，**不**单独链接；编译产物主要依赖 `libonnxruntime.so`。

## v4 构建与验证（H=200，推荐）

在仓库根目录：

```bash
# 1. 导出 v4 ONNX（20 s）
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/run_v4_20260520_034545/koopman_v4_best.pth \
  --pred_len 200 \
  --out_dir cpp/koopman_mpc/weights

# 2. 下载 ORT、编译 demo
bash cpp/koopman_mpc/build_v4.sh
```

MPC 参数（控制块、速率限制、权重）：[`../koopman_control/config/mpc_config.yaml`](../koopman_control/config/mpc_config.yaml)

| 关键参数 | v4 默认 | 含义 |
|----------|---------|------|
| `horizon` | 200 | 与 ONNX H 一致 |
| `control_hold_steps` | 10 | 控制每 1 s 变一次 |
| `opt_control_steps` | 40 | 优化前 4 s（4 块） |
| `throttle_du_max` / `rudder_du_max` | 15 / 3.5 | 块间变化速率 |

## v3 构建（H=20，历史）

```bash
bash cpp/koopman_mpc/build.sh
python3 cpp/koopman_mpc/scripts/verify_pipeline.py
```

## 运行 demo

```bash
export LD_LIBRARY_PATH=cpp/koopman_mpc/third_party/onnxruntime/lib:$LD_LIBRARY_PATH

# v3（H=20）
./cpp/koopman_mpc/build/koopman_mpc_cpp \
    --weights cpp/koopman_mpc/weights \
    --ref cpp/koopman_mpc/weights/cpp_test_ref.json \
    --steps 40 --horizon 20 --opt_iters 25

# 冒烟
./cpp/koopman_mpc/build/koopman_mpc_cpp --smoketest \
    --weights cpp/koopman_mpc/weights \
    --ref cpp/koopman_mpc/weights/cpp_test_ref.json
```

| 参数 | 说明 |
|------|------|
| `--weights` | 含 `koopman_rollout.onnx` 的目录 |
| `--ref` | 参考航迹 JSON |
| `--horizon` | **须与 ONNX 导出 H 一致**（v3=20，v4=200） |
| `--opt_control_steps` | 可选，覆盖 yaml |
| `--smoketest` | 快速自检 |

## 与 Python 版关系

| 组件 | Python | C++ |
|------|--------|-----|
| 动力学 | `KoopmanRollout` / PyTorch | `KoopmanOnnxModel`（ONNX，一次 Run = H 步） |
| 优化 | `torch.optim.Adam` + autograd | 源码内 Adam + 前向差分 |
| 控制 blocking | `MPCConfig.control_hold_steps` | `mpc_config.yaml` |
| 速率约束 | `throttle_du_max` 等 | 同左（块间 transition） |
| NLP 求解器 | 无 | 无（非 Ipopt/OSQP） |

重新训练后请重新导出 ONNX 并同步 yaml 中 `horizon`。详见 [docs/MPC使用指南.md](../../docs/MPC使用指南.md)。
