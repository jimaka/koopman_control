# Koopman MPC — C++ 实现（OSQP + ONNX plant）

> **控制库已独立至 [`../koopman_control/`](../koopman_control/README_CN.md)**（含 motion.cpp 对接、YAML 配置与 [`模型输入输出接口说明.md`](../koopman_control/模型输入输出接口说明.md)）。  
> 本目录保留 demo 程序、构建脚本与权重文件。

MPC **优化**通过潜空间 **OSQP** 求解；ONNX 仅作闭环仿真 **被控对象**（`simulate`），与 Python `koopman/export/rollout.py` 对齐。

## 目录结构

```
cpp/koopman_mpc/
├── CMakeLists.txt           # 链接 koopman_control 库
├── build.sh                 # v3：下载 ORT、导出 H=20 ONNX、编译、验证
├── build_v4.sh              # v4：导出权重 + 编译（推荐）
├── include/                 # 兼容头文件（转发至 koopman_control）
├── src/
│   ├── main.cpp             # demo：koopman_mpc_cpp（OSQP MPC + ONNX plant）
│   └── mpc_config.yaml      # demo 用配置副本
├── tools/verify_rollout.cpp
├── scripts/
│   ├── export_onnx.py       # v3 .pth → .onnx
│   └── verify_pipeline.py
├── third_party/onnxruntime/ # build.sh 自动下载（gitignore）
└── weights/                 # koopman_v4_latent.yaml、koopman_rollout.onnx（gitignore）
```

## 依赖

- C++17（g++）、CMake ≥ 3.18
- **OSQP**（由 `koopman_control` CMake FetchContent 拉取）、**yaml-cpp**
- **ONNX Runtime** C++（plant 仿真 / rollout 验证）
- Python 3 + `torch`、`onnx`、`onnxruntime`、`onnxscript`（导出与验证）

## v4 构建与验证（推荐）

在仓库根目录：

```bash
# 1. 潜空间权重（MPC 必需）
python3 new_v4_dict_input/export_v4_encode_weights.py \
  --ckpt checkpoints/run_v4_20260520_034545/koopman_v4_best.pth \
  --horizon 20 \
  --out cpp/koopman_mpc/weights/koopman_v4_latent.yaml

# 2. ONNX plant（闭环仿真）
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/run_v4_20260520_034545/koopman_v4_best.pth \
  --pred_len 200 \
  --out_dir cpp/koopman_mpc/weights

# 3. 下载 ORT、编译 demo
bash cpp/koopman_mpc/build_v4.sh
```

MPC 参数：[`../koopman_control/config/mpc_config.yaml`](../koopman_control/config/mpc_config.yaml)

| 关键参数 | v4 默认 | 含义 |
|----------|---------|------|
| `latent_model` | `koopman_v4_latent.yaml` | 潜空间 Ā,B + encoder + decoder |
| `horizon` | 20 | MPC 预测步数（粗步长 dt=1 s） |
| `opt_control_steps` | 2 | 优化前 2 步控制 |
| `w_xy` / `w_yaw` | 0 / 0 | Tier-2 位姿跟踪权重（>0 启用） |
| `sqp_iters` | 2 | Tier-2 SQP 外迭代次数 |
| `w_z` / `w_u` / `w_du` | 1.0 / 1e-4 / 0.05 | QP 代价权重 |
| `throttle_du_max` / `rudder_du_max` | 15 / 3.5 | 变化速率约束 |

## v3 构建（H=20，历史）

```bash
bash cpp/koopman_mpc/build.sh
python3 cpp/koopman_mpc/scripts/verify_pipeline.py
```

## 运行 demo

```bash
export LD_LIBRARY_PATH=cpp/koopman_mpc/third_party/onnxruntime/lib:$LD_LIBRARY_PATH

./cpp/koopman_mpc/build/koopman_mpc_cpp \
  --config cpp/koopman_control/config/mpc_config.yaml \
  --ref cpp/koopman_mpc/weights/cpp_test_ref.json \
  --steps 40

# 冒烟
./cpp/koopman_mpc/build/koopman_mpc_cpp \
  --config cpp/koopman_control/config/mpc_config.yaml --smoketest
```

| 参数 | 说明 |
|------|------|
| `--config` | `mpc_config.yaml`（含 `latent_model`、`w_z`、`osqp_*`） |
| `--ref` | 参考航迹 JSON |
| `--steps` | 闭环仿真步数 |
| `--horizon` | 可选，覆盖 yaml 中 MPC horizon |
| `--opt_control_steps` | 可选，覆盖 yaml |
| `--smoketest` | 快速自检（`xy_rmse < 8 m`） |

## 与 Python 版关系

| 组件 | Python | C++ |
|------|--------|-----|
| 潜空间动力学 | `model_v4_dict_input.py` | `KoopmanLatentModel`（YAML） |
| 编码 | `encode()` | `KoopmanEncoder` |
| 解码（Tier-2） | `reconstruct_state()` | `KoopmanDecoder` + `pose_linearize` |
| MPC 求解 | （已移除 Adam MPC） | **OSQP** condensed QP（Tier-1/2） |
| 闭环 plant | PyTorch / ONNX rollout | `KoopmanOnnxModel`（可选） |
| 参考工具 | `koopman.mpc.data_utils` | demo JSON / motion 桥接 |

重新训练后请重新运行 `export_v4_encode_weights.py`（及可选 `export_v4_onnx.py`）。详见 [docs/MPC使用指南.md](../../docs/MPC使用指南.md) 与 [docs/潜空间QP-MPC实现.md](../../docs/潜空间QP-MPC实现.md)。
