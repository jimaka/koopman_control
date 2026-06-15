# MPC 航迹跟踪（OSQP / C++）

MPC 求解**仅保留 C++ OSQP 潜空间路径**。Python Adam MPC、`compare_mpc_tracking.py` 已删除。

## 组件

| 组件 | 路径 |
|------|------|
| OSQP MPC 控制器 | `cpp/koopman_control/mpc_controller.cpp` |
| motion 桥接 | `cpp/koopman_control/motion_bridge.cpp` |
| 默认配置 | `cpp/koopman_control/config/mpc_config.yaml` |
| 权重导出 | `new_v4_dict_input/export_v4_encode_weights.py` |
| 文档 | `docs/潜空间QP-MPC实现.md` |

## 快速开始

```bash
python3 new_v4_dict_input/export_v4_encode_weights.py
cd cpp/koopman_control/build && cmake .. && cmake --build .
cd ../../koopman_mpc/build && cmake .. && cmake --build .
./koopman_mpc_cpp --config ../koopman_control/config/mpc_config.yaml --smoketest
```

## 求解器

| 项目 | 实现 |
|------|------|
| 类型 | 凸 QP（潜空间 Tier-1） |
| 库 | **OSQP v0.6.3**（CMake FetchContent） |
| 预测 | 预计算 Γ,Θ + encode；优化内无 ONNX rollout |
| Plant（仿真） | ONNX rollout（仅 `simulate` / demo） |

## 配置要点

```yaml
latent_model: cpp/koopman_mpc/weights/koopman_v4_latent.yaml
horizon: 20
w_z: 1.0
w_u: 0.0001
w_du: 0.05
osqp_max_iter: 4000
```

完整历史说明见 git 历史中的 Adam/ONNX 迭代版文档。
