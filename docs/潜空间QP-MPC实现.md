# v4 潜空间 OSQP-MPC

MPC **仅**通过 C++ [OSQP](https://osqp.org/) 求解潜空间 condensed QP；已移除 Adam / 投影梯度 / Python MPC 求解路径。

## 架构

```text
encode(dyn) → z0
Z_ref = stack(encode(ref_dyn_k))
min_U  0.5 U'PU + q'U   (P=2H, q=2Θ'Q(z_free-Z_ref))
s.t.   盒约束 + 变化率约束
OSQP → u0 → 下发控制
```

## 模块

| 文件 | 作用 |
|------|------|
| `latent_mpc_qp.cpp` | OSQP 组装与求解 |
| `mpc_controller.cpp` | `KoopmanMpcController`（唯一 MPC 类） |
| `motion_bridge.cpp` | motion.cpp 桥接 |
| `export_v4_encode_weights.py` | 导出 `koopman_v4_latent.yaml` |

## 使用

```bash
python3 new_v4_dict_input/export_v4_encode_weights.py
cd cpp/koopman_control/build && cmake .. && cmake --build .
./koopman_mpc_cpp --config ../koopman_control/config/mpc_config.yaml --smoketest
```

配置：`cpp/koopman_control/config/mpc_config.yaml`（`latent_model`, `w_z`, `osqp_*`）。

## 验证

```bash
python3 tests/test_latent_qp_matrices.py
python3 tests/test_v4_encode_reference.py
```

## 说明

- **Tier-1**：潜空间跟踪；参考由 `encode(ref [u,v,r])` 构造。
- **闭环仿真 plant**：demo 仍用 ONNX rollout 推进被控对象；**优化器不用 ONNX**。

## Tier-2：物理位姿 \((x,y,\psi)\) 跟踪（已实现）

当 `w_xy > 0` 或 `w_yaw > 0` 且 latent YAML 含 `decoder` 时启用。

**原理**：位姿来自 `decoder(z) → (u,v,r)` 的船体系欧拉积分，含三角非线性。
每步在标称轨迹处一阶线性化，得到位姿对控制序列的仿射灵敏度：

\[
P_{xy\psi} \approx p_{free} + \Phi\,U,\qquad
\text{cost} \mathrel{+}= w_{xy}\|x,y - ref\|^2 + w_{yaw}\,\|\psi - ref\|^2
\]

叠加进 QP：`P += 2ΦᵀQ_pΦ`，`q += 2ΦᵀQ_p b`（仍是凸 QP，复用 OSQP）。

**关键工程处理**：

| 问题 | 处理 |
|------|------|
| decoder 非线性 | 导出 `decoder` 权重 + 解析 Jacobian（含 `dyn_std` 缩放） |
| 欧拉积分 `sin/cos` 非线性 | 在标称 `ψ_k^0` 处线性化 |
| 长 horizon 线性化失效 | **SQP 外迭代**（`sqp_iters`，默认 2）更新标称轨迹 |
| yaw 角度环绕 | 标称角误差 `wrapAngle` 后再线性化 |
| 坐标系不一致 | `motion_bridge` 把全局参考变换到当前船体系（`MotionSolveInput` 增加 `x,y,psi,has_pose`） |
| 位姿不可行 | 仅作**软约束**（代价项），不加硬约束 |

**配置**（`mpc_config.yaml`）：

```yaml
w_xy: 1.0      # 平面位置跟踪（默认 0=关闭）
w_yaw: 0.5     # 艏向跟踪
sqp_iters: 2   # SQP 外迭代次数
```

**模块**：

| 文件 | 作用 |
|------|------|
| `koopman_decoder.cpp` | decoder 前向 + 物理 Jacobian |
| `pose_linearize.cpp` | 位姿灵敏度 Φ、偏置 b、权重 wq |
| `latent_mpc_qp.cpp` | QP 叠加位姿项 |
| `mpc_controller.cpp` | SQP 外迭代 + 标称 rollout |

**验证**：

```bash
python3 tests/test_pose_linearize.py            # Jacobian + Φ 二阶收敛（float64）
# C++（含 OSQP 端到端）：构建后运行
./verify_pose_linearize cpp/koopman_mpc/weights/koopman_v4_latent.yaml
```
