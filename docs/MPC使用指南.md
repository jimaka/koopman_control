# MPC 航迹跟踪（OSQP / C++）

MPC 求解**仅保留 C++ OSQP 潜空间路径**。Python Adam MPC（`KoopmanMPC`、`scripts/mpc_track.py`）、`compare_mpc_tracking.py` 已删除。

## 架构

```text
export_v4_encode_weights.py → koopman_v4_latent.yaml (Ā, B, encoder)
encode(z₀) + 预计算 Γ,Θ,ξ
OSQP: min ½ U'PU + q'U   (Tier-1 潜空间跟踪)
KoopmanMpcController::solveStep() → u₀
ONNX rollout: 仅 plant（simulate / demo），不在优化器内
```

## 组件

| 组件 | 路径 |
|------|------|
| OSQP MPC 控制器 | `cpp/koopman_control/src/mpc_controller.cpp` |
| QP 组装与求解 | `cpp/koopman_control/src/latent_mpc_qp.cpp` |
| motion 桥接 | `cpp/koopman_control/src/motion_bridge.cpp` |
| 默认配置 | `cpp/koopman_control/config/mpc_config.yaml` |
| 潜空间权重导出 | `new_v4_dict_input/export_v4_encode_weights.py` |
| ONNX plant 导出 | `new_v4_dict_input/export_v4_onnx.py` |
| QP 推导文档 | `docs/潜空间QP-MPC实现.md` |
| Python 参考工具 | `koopman/mpc/data_utils.py` |

## 快速开始

```bash
# 1. 导出潜空间权重（MPC 必需；权重目录默认 gitignore）
python3 new_v4_dict_input/export_v4_encode_weights.py \
  --ckpt checkpoints/run_v4_20260520_034545/koopman_v4_best.pth \
  --horizon 20 \
  --out cpp/koopman_mpc/weights/koopman_v4_latent.yaml

# 2. 导出 ONNX plant（闭环 demo 用，可选）
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/run_v4_20260520_034545/koopman_v4_best.pth \
  --pred_len 200 --out_dir cpp/koopman_mpc/weights

# 3. 编译
bash cpp/koopman_mpc/build_v4.sh

# 4. 冒烟
export LD_LIBRARY_PATH=cpp/koopman_mpc/third_party/onnxruntime/lib:$LD_LIBRARY_PATH
./cpp/koopman_mpc/build/koopman_mpc_cpp \
  --config cpp/koopman_control/config/mpc_config.yaml --smoketest
```

## 求解器

| 项目 | 实现 |
|------|------|
| 类型 | 凸 QP（潜空间 Tier-1） |
| 库 | **OSQP v0.6.3**（CMake FetchContent） |
| 预测 | 预计算 Γ,Θ + encode；优化内无 ONNX rollout |
| Plant（仿真） | ONNX rollout（仅 `simulate` / demo） |
| 实船 | `KoopmanMotionMpc(latent_yaml)`，无需 ONNX |

## 配置要点

```yaml
latent_model: cpp/koopman_mpc/weights/koopman_v4_latent.yaml
horizon: 20          # 与 export --horizon 一致
dt: 1.0
opt_control_steps: 2
w_z: 1.0
w_u: 0.0001
w_du: 0.05
throttle_du_max: 15.0
rudder_du_max: 3.5
osqp_max_iter: 4000
onnx_plant: cpp/koopman_mpc/weights/koopman_rollout.onnx  # 仅 simulate
```

## 验证

```bash
python3 tests/test_latent_qp_matrices.py
python3 tests/test_v4_encode_reference.py
```

## 说明

- **Tier-1**：潜空间 `z` 跟踪；参考由 `encode(ref [u,v,r])` 构造。
- **Tier-2**（物理 `(x,y,ψ)` 跟踪）：已实现，`w_xy>0` 或 `w_yaw>0` 启用，
  通过 decoder + 欧拉积分线性化 + SQP 外迭代，详见 [潜空间QP-MPC实现.md](./潜空间QP-MPC实现.md)。
  实船经 motion 桥接时需在 `MotionSolveInput` 提供当前全局位姿 `x,y,psi` 并置 `has_pose=true`。
- 历史 Adam / ONNX 迭代优化说明见 git 历史。
