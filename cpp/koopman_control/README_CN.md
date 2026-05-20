# Koopman MPC 控制库 — 中文说明

## 1. 概述

`cpp/koopman_control/` 是将 **Koopman v4 + ONNX Runtime** 的 MPC 控制逻辑独立封装后的 C++ 库，供：

- 本仓库离线仿真 / 测试（`cpp/koopman_mpc/` 中的 demo 程序）
- **Elane 实船 ROS 节点 `motion.cpp`** 通过桥接层 `motion_bridge.hpp` 调用

与原先 `motion.cpp` 中使用的水动力 MPC（`MpcTaSlove` / 西城 Azimuthing）不同，本库使用 **训练得到的 Koopman 神经网络 rollout** 作为预测模型。

```
cpp/koopman_control/
├── CMakeLists.txt              # 构建静态/动态库 koopman_control
├── README_CN.md                # 本文档
├── config/mpc_config.yaml      # MPC 默认参数
├── include/koopman_control/
│   ├── koopman_onnx_model.hpp  # ONNX 推理
│   ├── mpc_config.hpp          # 代价权重、horizon 等
│   ├── mpc_controller.hpp      # 滚动时域 MPC 求解器
│   ├── mpc_config_loader.hpp   # YAML 配置加载
│   └── motion_bridge.hpp       # ★ motion.cpp 对接入口
├── src/                        # 实现
└── examples/
    └── motion_integration_example.cpp
```

---

## 2. 依赖

| 依赖 | 用途 |
|------|------|
| C++17 | 语言标准 |
| ONNX Runtime C++ ≥ 1.26 | 加载 `koopman_rollout.onnx` |
| yaml-cpp | 读取 `config/mpc_config.yaml` |

ONNX 权重由 Python 导出：

```bash
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/run_v4_20260520_034545/koopman_v4_best.pth \
  --pred_len 200 \
  --out_dir cpp/koopman_mpc/weights
```

---

## 3. 核心 API

### 3.1 底层：`KoopmanMpcController`

```cpp
#include "koopman_control/mpc_controller.hpp"

koopman_control::KoopmanOnnxModel model("/path/to/koopman_rollout.onnx");
koopman_control::MpcConfig cfg;
cfg.horizon = model.horizon();  // 通常 200

koopman_control::KoopmanMpcController mpc(std::move(model), cfg);

std::array<float, 6> state0{0, 0, 0, u, v, r};
std::vector<std::array<float, 6>> ref_window;  // 长度 horizon+1
auto [u_opt, cost] = mpc.solveStep(state0, ref_window);
// u_opt: 4 维控制量
```

状态定义：`[x, y, yaw, u, v, r]`，与训练数据一致。

### 3.2 上层：`KoopmanMotionMpc`（供 motion.cpp 使用）

```cpp
#include "koopman_control/motion_bridge.hpp"

koopman_control::MpcConfig cfg =
    koopman_control::loadMpcConfigFromYaml("cpp/koopman_control/config/mpc_config.yaml");

koopman_control::MotionBridgeConfig bridge;
bridge.ref_dt = 0.1f;           // 对应 motion 中 mpc_during
bridge.ref_time_offset = 0.5f;  // 对应 PointChange 中 +0.5

koopman_control::KoopmanMotionMpc solver(onnx_path, cfg, bridge);

koopman_control::MotionSolveInput in;
in.u = fusion_pose->velocity.x;
in.v = fusion_pose->velocity.y;
in.r = yaw_rate_;
for (const auto& t : mpc_states_gl.targets) {
    koopman_control::MotionRefPoint p;
    p.x = t.x; p.y = t.y; p.psi = t.psi;
    p.u = t.u; p.v = t.v; p.r = 0.f;
    in.ref.push_back(p);
}

koopman_control::MotionSolveOutput out;
if (solver.solve(in, out)) {
    // out.control[0..3] 为 Koopman 4 维控制量
}
```

桥接层会自动将 motion 侧较短/较稀疏的参考序列 **重采样** 为 ONNX 所需的 `horizon+1` 个点。

---

## 4. 在 motion.cpp 中集成

### 4.1 CMake 增加库

在 `ship_control` 包的 `CMakeLists.txt` 中：

```cmake
add_subdirectory(/path/to/koopman_control/repo/cpp/koopman_control)
target_link_libraries(ship_control_node PRIVATE koopman_control)
target_include_directories(ship_control_node PRIVATE
    /path/to/koopman_control/repo/cpp/koopman_control/include)
target_compile_definitions(ship_control_node PRIVATE USE_KOOPMAN_MPC=1)
```

并设置 `LD_LIBRARY_PATH` 包含 ONNX Runtime 的 `lib` 目录。

### 4.2 头文件 `motion.h`

在 `ControlNode` 中增加（参见仓库内已提供的 `cpp/motion_koopman_mpc.hpp`）：

```cpp
#ifdef USE_KOOPMAN_MPC
#include "koopman_control/motion_bridge.hpp"
#endif

// private:
#ifdef USE_KOOPMAN_MPC
  void MpcTaRunKoopman();
  std::unique_ptr<koopman_control::KoopmanMotionMpc> koopman_mpc_;
#endif
```

### 4.3 初始化 `Init()`

```cpp
#ifdef USE_KOOPMAN_MPC
  koopman_control::MpcConfig cfg = koopman_control::loadMpcConfigFromYaml(
      "/opt/elane/ros/share/ship_control/config/koopman_mpc.yaml");
  koopman_control::MotionBridgeConfig bridge;
  bridge.ref_dt = mpc_during;
  bridge.ref_time_offset = 0.5f;
  koopman_mpc_ = std::make_unique<koopman_control::KoopmanMotionMpc>(
      cfg_onnx_path, cfg, bridge);
#endif
```

### 4.4 主循环 `Run()`

增加船型分支或替换西城 MPC：

```cpp
case static_cast<int>(ThrustModeId::DOUBLE_THRUST_XI_CHENG):
#ifdef USE_KOOPMAN_MPC
  MpcTaRunKoopman();
#else
  MpcTaRunXiCheng();
#endif
  break;
```

### 4.5 实现 `MpcTaRunKoopman()`

完整示例见：

- `cpp/koopman_control/examples/motion_integration_example.cpp`
- `cpp/motion_koopman_mpc.cpp`（本仓库提供的可拷贝实现）

---

## 5. 控制量与推进器的映射

Koopman 模型输出 **4 维抽象控制** `u[0..3]`（与训练数据集 `ctrl` 字段一致），**不是**直接的左右推力/舵角。

`motion.cpp` 当前通过 `thrust_command_send` 发布：

- `port_thruster_throttle` / `starboard_thruster_throttle`
- `port_thruster_angle` / `starboard_thruster_angle`

接入 Koopman MPC 后需要增加一层 **控制分配（Control Allocation）**：

1. **短期**：在仿真中直接将 `u[0..3]` 映射到训练数据的物理含义（需对照数据集文档）
2. **长期**：仿照现有 `xicheng_azimuthing_MpcTaRudderSlove_`，编写 4 维 u → 双推进器 (force, angle) 的分配矩阵

在未完成分配前，可先将 `u[0]`, `u[1]` 线性映射到左右油门做联调。

---

## 6. 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `horizon` | 200 | 与 ONNX 一致，20s @ dt=0.1 |
| `opt_control_steps` | 40 | 仅优化前 4s 控制，降低计算量 |
| `opt_iters` | 15 | Adam 迭代次数 |
| `dt` | 0.1 | 与训练/ONNX 一致 |
| `motion_ref_dt` | 0.1 | motion `PointChange` 参考点时间间隔 |

单步 ONNX rollout（H=200）在 CPU 上约 **数毫秒～数十毫秒**；完整 MPC 求解取决于 `opt_control_steps × opt_iters`，建议实船控制周期 **≥ 0.5s** 或减小 `opt_control_steps`。

---

## 7. 构建

### 7.1 仅构建库

```bash
cd cpp/koopman_control
mkdir -p build && cd build
cmake .. -DONNXRUNTIME_ROOT=../koopman_mpc/third_party/onnxruntime
cmake --build . -j
```

### 7.2 与 demo 一起构建

```bash
bash cpp/koopman_mpc/build_v4.sh
```

`cpp/koopman_mpc/CMakeLists.txt` 已通过 `add_subdirectory` 链接本库。

---

## 8. 与 motion.cpp 的数据对应关系

| motion.cpp | KoopmanMotionMpc |
|------------|------------------|
| `FusionPoseStamped_->velocity.x/y` | `MotionSolveInput.u / .v` |
| `yaw_rate_` | `MotionSolveInput.r` |
| `ini_state << 0,0,0,u,v,r` | 船体系原点，桥接层内部构造 `state0` |
| `mpc_states_gl.targets[i].x/y/psi/u/v` | `MotionRefPoint` |
| `PointChange()` + `mpc_during` | `MotionBridgeConfig.ref_dt` |
| `xicheng_...Solve()` 输出 | `MotionSolveOutput.control[4]` |

---

## 9. 常见问题

**Q: horizon 必须为 200 吗？**  
A: 须与 ONNX 导出时的 `--pred_len` 一致。更换 horizon 需重新导出 ONNX。

**Q: motion 参考点只有 20 个，horizon 是 200 怎么办？**  
A: `motion_bridge` 按时间线性插值并重采样，末端 hold 最后一点。

**Q: 编译找不到 onnxruntime？**  
A: 先运行 `bash cpp/koopman_mpc/build.sh` 或 `build_v4.sh` 下载 ORT，或手动指定 `ONNXRUNTIME_ROOT`。

---

## 10. 相关文件

| 文件 | 说明 |
|------|------|
| `new_v4_dict_input/export_v4_onnx.py` | 导出 ONNX |
| `cpp/koopman_mpc/build_v4.sh` | v4 全流程构建脚本 |
| `cpp/motion.cpp` | Elane ROS 控制节点（集成目标） |
| `cpp/motion_koopman_mpc.cpp` | 可拷贝的 `MpcTaRunKoopman` 实现 |
