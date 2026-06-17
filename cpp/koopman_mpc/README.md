# Koopman MPC — C++ 实现（OSQP + ONNX plant）

> **控制库已独立至 [`../koopman_control/`](../koopman_control/README_CN.md)**（含 motion.cpp 对接、YAML 配置与 [`模型输入输出接口说明.md`](../koopman_control/模型输入输出接口说明.md)）。  
> 本目录保留 demo 程序、构建脚本与权重文件。

MPC **优化**通过潜空间 **OSQP** 求解；ONNX 仅作闭环仿真 **被控对象**（`simulate`），与 Python `koopman/export/rollout.py` 对齐。

## 目录结构

```
cpp/koopman_mpc/
├── CMakeLists.txt           # 链接 koopman_control 库
├── build.sh                 # v3：下载 ORT、导出 H=20 ONNX、编译、验证
├── build_v4.sh              # v4：导出 latent YAML + ONNX + 编译 + 验证（推荐）
├── build_in_docker.sh       # v4：容器内直接运行、直接编译（按需装依赖）
├── build_v4_in_docker.sh    # v4：宿主机启动器（docker exec 进容器再编译）
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

> `build_v4.sh` 现在会**同时导出** latent YAML（`export_v4_encode_weights.py`，OSQP MPC 必需）
> 与 ONNX plant，再 CMake 构建并跑冒烟。

### 在 Docker 容器内编译

提供两种方式：

**A. 容器内直接运行（推荐，最简单）** —— `build_in_docker.sh`
已 `docker exec -it <容器> bash` 进入容器后，直接在仓库根执行；它会检测依赖（缺失才装）、
按需下载 ONNX Runtime，然后 CMake 直接编译：

```bash
# 仅编译 C++（无需 Python/ckpt）
bash cpp/koopman_mpc/build_in_docker.sh

# 顺带导出权重并跑冒烟
bash cpp/koopman_mpc/build_in_docker.sh \
  --weights checkpoints/run_v4_xxx/koopman_v4_best.pth --smoketest

# 仅检测不装依赖 / 指定并行数
bash cpp/koopman_mpc/build_in_docker.sh --skip-deps --jobs 8
```

**B. 宿主机一键启动** —— `build_v4_in_docker.sh`
在宿主机执行，先检测本地 Docker（CLI/容器存在且运行中），再 `docker exec` 进容器
检测依赖并执行 `build_v4.sh`：

```bash
CONTAINER_NAME=my_ctr bash cpp/koopman_mpc/build_v4_in_docker.sh \
  --ckpt /workspace/checkpoints/run_v4_xxx/koopman_v4_best.pth --pred_len 20 --sync
```

**C. 仅编译 MPC 核心（不依赖 ONNX）** —— `cpp/koopman_control/build_mpc_only.sh`
只跑 OSQP 优化、想规避 ONNX Runtime 下载时使用（`-DKOOPMAN_ENABLE_ONNX=OFF`）：

```bash
bash cpp/koopman_control/build_mpc_only.sh   # 依赖仅 g++/cmake/git/yaml-cpp
```

### 网络受限 / 跨机器注意

- **ORT 下载失败（curl 56 等）**：脚本已内置重试 + 断点续传；仍失败时
  - 离线包：`bash ... build_in_docker.sh --ort-tgz /path/onnxruntime-linux-x64-1.26.0.tgz`
  - 镜像：`ORT_URL=<镜像tgz> bash ...`
  - 或干脆用方式 C 跳过 ONNX。
- **跨机器复用 checkout 报 `CMakeCache.txt` 源目录不一致**：脚本会自动清理陈旧
  `build/`；手动可 `rm -rf cpp/koopman_*/build` 后重跑。

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
