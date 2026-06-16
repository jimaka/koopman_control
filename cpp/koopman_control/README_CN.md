# Koopman MPC 控制库 — 中文说明

> **模型 I/O 接口（图文并茂）** → [`模型输入输出接口说明.md`](模型输入输出接口说明.md)

## 1. 概述

`cpp/koopman_control/` 将 **v4 潜空间 OSQP-MPC** 独立封装为 C++ 库，供：

- 本仓库离线仿真 / 测试（`cpp/koopman_mpc/` 中的 demo 程序）
- **Elane 实船 ROS 节点 `motion.cpp`** 通过桥接层 `motion_bridge.hpp` 调用

MPC 优化路径：**encode → condensed QP（Γ,Θ,ξ）→ OSQP**。ONNX 仅作闭环仿真 **plant**（`simulate` / demo），**不参与** `solveStep` 内的优化前向。

支持两层跟踪目标：
- **Tier-1（默认）**：潜空间 `z` 跟踪（速度）。
- **Tier-2（可选，`w_xy>0`/`w_yaw>0`）**：物理位姿 `(x,y,ψ)` 跟踪，经 decoder + 欧拉积分线性化 + SQP 外迭代叠加进同一 OSQP。

```
cpp/koopman_control/
├── CMakeLists.txt              # 构建库；FetchContent 拉取 OSQP v0.6.3
├── README_CN.md                # 本文档
├── config/mpc_config.yaml      # MPC 默认参数
├── include/koopman_control/
│   ├── koopman_latent_model.hpp # Ā,B,Γ,Θ 预计算
│   ├── koopman_encode.hpp      # dict16 + res_mlp 编码
│   ├── koopman_decoder.hpp     # decoder 前向 + Jacobian（Tier-2）
│   ├── pose_linearize.hpp      # 位姿灵敏度 Φ（Tier-2）
│   ├── latent_mpc_qp.hpp       # OSQP 组装与求解（含位姿项）
│   ├── koopman_onnx_model.hpp  # ONNX plant（仿真用）
│   ├── mpc_controller.hpp      # KoopmanMpcController（SQP 外迭代）
│   ├── mpc_config_loader.hpp   # YAML 配置加载
│   └── motion_bridge.hpp       # ★ motion.cpp 对接入口
├── src/
└── examples/
    └── motion_integration_example.cpp
```

---

## 2. 依赖

| 依赖 | 用途 |
|------|------|
| C++17 | 语言标准 |
| **OSQP v0.6.3** | 凸 QP 求解（CMake FetchContent 自动获取） |
| yaml-cpp | 读取 `config/mpc_config.yaml` |
| ONNX Runtime C++ ≥ 1.26 | 可选；仅 `simulate` / demo 闭环 plant |

权重由 Python 导出：

```bash
# MPC 优化必需（同时导出 encoder 与 decoder；decoder 供 Tier-2 位姿跟踪）
python3 new_v4_dict_input/export_v4_encode_weights.py \
  --ckpt checkpoints/run_v4_20260520_034545/koopman_v4_best.pth \
  --horizon 20 \
  --out cpp/koopman_mpc/weights/koopman_v4_latent.yaml

# 闭环仿真 plant（可选）
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
#include "koopman_control/mpc_config_loader.hpp"

koopman_control::MpcConfig cfg =
    koopman_control::loadMpcConfigFromYaml("cpp/koopman_control/config/mpc_config.yaml");
koopman_control::LatentMpcQpConfig qp_cfg = koopman_control::latentQpConfigFromMpc(cfg);

// 实船 / motion：仅需潜空间 YAML
koopman_control::KoopmanMpcController mpc(cfg.latent_model, cfg, qp_cfg);

// demo 闭环仿真：附加 ONNX plant
koopman_control::KoopmanMpcController mpc_sim(cfg.latent_model, cfg.onnx_plant, cfg, qp_cfg);

std::array<float, 6> state0{0, 0, 0, u, v, r};
std::vector<std::array<float, 6>> ref_window;  // 长度 horizon+1
auto [u_opt, cost] = mpc.solveStep(state0, ref_window);
// u_opt: 4 维控制量
```

状态定义：`[x, y, yaw, u, v, r]`，与训练数据一致。motion 侧通常 `state0 = [0,0,0,u,v,r]`。

### 3.2 上层：`KoopmanMotionMpc`（供 motion.cpp 使用）

```cpp
#include "koopman_control/motion_bridge.hpp"

koopman_control::MpcConfig cfg =
    koopman_control::loadMpcConfigFromYaml("cpp/koopman_control/config/mpc_config.yaml");

koopman_control::MotionBridgeConfig bridge;
bridge.ref_dt = 1.0f;           // 参考点时间间隔
bridge.ref_time_offset = 0.5f;  // 对应 PointChange 中 +0.5

koopman_control::KoopmanMotionMpc solver(cfg.latent_model, cfg, bridge);

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
    // out.timing.qp_solve_ms / osqp_iters 可观测
}
```

桥接层将 motion 侧较短参考序列 **重采样** 为 `horizon+1` 个点（步长 `cfg.dt`）。

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

并设置 `LD_LIBRARY_PATH` 包含 ONNX Runtime 的 `lib` 目录（若使用 ONNX plant 仿真）。

### 4.2 初始化 `Init()`

```cpp
#ifdef USE_KOOPMAN_MPC
  koopman_control::MpcConfig cfg = koopman_control::loadMpcConfigFromYaml(
      "/opt/elane/ros/share/ship_control/config/koopman_mpc.yaml");
  koopman_control::MotionBridgeConfig bridge;
  bridge.ref_dt = mpc_during;
  bridge.ref_time_offset = 0.5f;
  koopman_mpc_ = std::make_unique<koopman_control::KoopmanMotionMpc>(
      cfg.latent_model, cfg, bridge);
#endif
```

完整示例见 `cpp/koopman_control/examples/motion_integration_example.cpp` 与 `cpp/motion_koopman_mpc.cpp`。

---

## 5. 控制量与推进器的映射

Koopman 模型输出 **4 维抽象控制** `u[0..3]`（与训练数据集 `ctrl` 字段一致），**不是**直接的左右推力/舵角。

`motion.cpp` 当前通过 `thrust_command_send` 发布左右油门与舵角。接入 Koopman MPC 后需要 **控制分配（Control Allocation）** 将 4 维 u 映射到双推进器指令。

---

## 6. 参数说明

配置文件：`config/mpc_config.yaml`（由 `loadMpcConfigFromYaml()` 加载）。

| 参数 | 默认 | 说明 |
|------|------|------|
| `latent_model` | `koopman_v4_latent.yaml` | 潜空间动力学 + encoder（+ decoder）权重 |
| `horizon` | 20 | MPC 预测步数 |
| `dt` | 1.0 | MPC 时间步（秒） |
| `opt_control_steps` | 2 | 参与优化的控制步数 |
| `control_hold_steps` | 1 | 控制零阶保持块大小 |
| `w_z` / `w_u` / `w_du` | 1.0 / 1e-4 / 0.05 | 潜空间跟踪 / 控制幅值 / 增量惩罚 |
| `w_xy` / `w_yaw` | 0 / 0 | **Tier-2** 位姿跟踪权重（>0 启用，需 decoder） |
| `sqp_iters` | 2 | **Tier-2** SQP 外迭代次数 |
| `throttle_du_max` / `rudder_du_max` | 15 / 3.5 | 块间变化速率硬约束；≤0 不限制 |
| `osqp_eps_abs` / `osqp_eps_rel` | 1e-4 | OSQP 容差 |
| `osqp_max_iter` | 4000 | OSQP 最大迭代 |
| `u_min` / `u_max` | ±100 / ±35 | 4 维控制盒约束 |
| `onnx_plant` | `koopman_rollout.onnx` | 仅 `simulate` / demo 使用 |
| `motion_ref_dt` | 1.0 | motion 参考点时间间隔 |

**依赖库**：编译链接 **OSQP**、**yaml-cpp**；使用 ONNX plant 时额外链接 **ONNX Runtime**。

单步 `solveStep` 为毫秒级 QP（无迭代 rollout）；适合实船 **≥ 0.5 s** 控制周期。

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

---

## 8. 与 motion.cpp 的数据对应关系

| motion.cpp | KoopmanMotionMpc |
|------------|------------------|
| `FusionPoseStamped_->velocity.x/y` | `MotionSolveInput.u / .v` |
| `yaw_rate_` | `MotionSolveInput.r` |
| `ini_state << 0,0,0,u,v,r` | 桥接层内部构造 `state0` |
| `mpc_states_gl.targets[i].x/y/psi/u/v` | `MotionRefPoint` |
| 当前全局位姿（Tier-2） | `MotionSolveInput.{x,y,psi}` + `has_pose=true` |
| `PointChange()` + `mpc_during` | `MotionBridgeConfig.ref_dt` |
| `xicheng_...Solve()` 输出 | `MotionSolveOutput.control[4]` |

---

## 9. 常见问题

**Q: horizon 必须为 20 吗？**  
A: 须与 `export_v4_encode_weights.py --horizon` 及 yaml 中 `horizon` 一致。更换后需重新导出 latent YAML。

**Q: motion 参考点只有 20 个，horizon 是 20 怎么办？**  
A: `motion_bridge` 按时间线性插值并重采样为 `horizon+1` 点，末端 hold 最后一点。

**Q: ONNX 还用吗？**  
A: **MPC 优化不用**。仅 demo `simulate` 或需要 ONNX 闭环仿真时加载 `onnx_plant`。

**Q: 编译找不到 onnxruntime？**  
A: 先运行 `bash cpp/koopman_mpc/build_v4.sh` 下载 ORT，或手动指定 `ONNXRUNTIME_ROOT`。

**Q: 如何启用位姿 (x,y,yaw) 跟踪？**  
A: 在 `mpc_config.yaml` 设 `w_xy>0`/`w_yaw>0`，并确保 latent YAML 含 `decoder`
（新版 `export_v4_encode_weights.py` 自动导出）。实船经 motion 桥接时须填 `MotionSolveInput.{x,y,psi}` 并置 `has_pose=true`。

---

## 10. 相关文件

| 文件 | 说明 |
|------|------|
| `new_v4_dict_input/export_v4_encode_weights.py` | 导出潜空间 YAML（含 encoder + decoder） |
| `new_v4_dict_input/export_v4_onnx.py` | 导出 ONNX plant |
| `src/koopman_decoder.cpp` / `src/pose_linearize.cpp` | Tier-2 decoder Jacobian / 位姿线性化 |
| `tools/verify_pose_linearize.cpp` | Tier-2 验证（Φ 精度 + OSQP 端到端） |
| `docs/潜空间QP-MPC实现.md` | QP 推导与模块索引 |
| `cpp/koopman_mpc/build_v4.sh` | v4 全流程构建脚本 |
| `cpp/motion_koopman_mpc.cpp` | 可拷贝的 `MpcTaRunKoopman` 实现 |
