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
- Tier-2（物理 \((x,y,\psi)\) 跟踪）未实现。
