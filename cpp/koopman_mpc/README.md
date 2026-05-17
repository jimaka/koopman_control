# Koopman MPC — C++ 实现

基于 **LibTorch**（与 Python 版相同的 TorchScript rollout）的船舶航迹跟踪 MPC，逻辑对齐仓库根目录 `mpc_koopman.py` / `run_mpc_tracking.py`。

## 目录结构

```
cpp/koopman_mpc/
├── CMakeLists.txt
├── build.sh                 # 一键导出权重、编译、验证
├── include/                 # 头文件
├── src/                     # LibTorch 模型加载 + MPC
├── tools/verify_rollout.cpp # rollout 数值对照
├── scripts/                 # 导出 TorchScript / 测试数据
└── weights/                 # 生成物（gitignore）
    ├── koopman_rollout.pt
    ├── cpp_test_ref.json
    └── ...
```

## 依赖

- C++17 编译器（g++）
- CMake ≥ 3.18
- Python 3 + PyTorch（`pip install torch`）— 提供 LibTorch 与导出脚本

## 构建与验证

在仓库根目录执行：

```bash
bash cpp/koopman_mpc/build.sh
```

脚本将依次：

1. 导出 `weights/koopman_rollout.pt`（TorchScript）
2. 导出 `weights/cpp_test_ref.json`（测试航迹）
3. `cmake` 编译 `koopman_mpc_cpp` 与 `verify_rollout`
4. 对照 Python rollout（误差 &lt; 1e-3）
5. 运行 MPC 冒烟测试

## 运行

```bash
./cpp/koopman_mpc/build/koopman_mpc_cpp \
    --weights cpp/koopman_mpc/weights \
    --ref cpp/koopman_mpc/weights/cpp_test_ref.json \
    --steps 40 --horizon 20 --opt_iters 25
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--weights` | 含 `koopman_rollout.pt` 的目录 |
| `--ref` | 文本参考航迹（`export_cpp_test_ref.py` 生成） |
| `--steps` | 闭环仿真步数 |
| `--horizon` | MPC 预测步长 |
| `--opt_iters` | 每步 Adam 迭代次数 |
| `--smoketest` | 快速自检 |

## 与 Python 版关系

| 组件 | Python | C++ |
|------|--------|-----|
| 动力学预测 | `KoopmanMPC.rollout` | `KoopmanTorchModel::rollout`（同一 TorchScript） |
| 优化器 | `torch.optim.Adam` | LibTorch Adam |
| 代价函数 | 相同权重 `w_xy/w_yaw/w_vel/...` | `MpcConfig` |

重新训练模型后，请重新运行 `build.sh` 以更新 `koopman_rollout.pt`。
