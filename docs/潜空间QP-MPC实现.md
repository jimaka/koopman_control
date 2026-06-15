# v4 潜空间 QP-MPC 实现说明

本文档是 [潜空间 QP 推导](上一节对话) 的 **落地实现**：Tier-1 潜空间跟踪、矩阵预计算、C++ 模块与验证流程。

---

## 1. 已实现模块

| 文件 | 作用 |
|------|------|
| `new_v4_dict_input/export_v4_encode_weights.py` | 从 ckpt 导出 `koopman_v4_latent.yaml/json`（Ā,B,bias, encoder 权重, 归一化） |
| `cpp/koopman_control/include/koopman_control/koopman_latent_model.hpp` | 加载 YAML，预计算 Γ,Θ,ξ |
| `cpp/koopman_control/include/koopman_control/koopman_encode.hpp` | dict16 + res_mlp → z₀ |
| `cpp/koopman_control/include/koopman_control/latent_mpc_qp.hpp` | Tier-1 QP 代价 + 投影梯度求解 |
| `cpp/koopman_control/include/koopman_control/mpc_controller_qp.hpp` | `KoopmanQpMpcController`（API 对齐 Adam MPC） |
| `cpp/koopman_control/tools/verify_latent_qp.cpp` | C++ 矩阵/encode 对照工具 |
| `tests/test_latent_qp_matrices.py` | Python condensed 预测 ≡ step rollout |
| `tests/test_v4_encode_reference.py` | C++ encode 算法 ≡ PyTorch |
| `tests/export_latent_qp_cpp_ref.py` | 生成 C++ verify 用参考向量 |

---

## 2. QP 形式（Tier-1，已实现）

决策变量：归一化控制堆叠 \(\tilde U \in \mathbb{R}^{4N}\)（默认 \(N=20\) → 80 维）。

\[
\min_{\tilde U}\; \|\Theta \tilde U + \Gamma z_0 + \xi - Z^{\mathrm{ref}}\|_{Q_z}^2
+ \|\tilde U\|_R^2 + \|D_u \tilde U\|_S^2
\quad \text{s.t. 盒约束 + 速率约束（投影）}
\]

- \(H = \Theta^\top Q \Theta + R + D_u^\top S D_u\) 在 `LatentMpcQpSolver::buildHessian()` 中预计算（\(Q,R,S\) 为标量对角缩放）。
- 求解器：**投影梯度下降**（不依赖 OSQP/Eigen，便于嵌入式集成）；凸问题下可替换为 OSQP 接口而不改上层 API。
- 参考 \(Z^{\mathrm{ref}}\)：对 `ref_window` 每步 \([u,v,r]\) 归一化后 `encode` 再堆叠。

---

## 3. 使用流程

### 3.1 导出权重

```bash
python3 new_v4_dict_input/export_v4_encode_weights.py \
  --ckpt checkpoints/koopman_v4_best.pth \
  --horizon 20
```

产物：

- `cpp/koopman_mpc/weights/koopman_v4_latent.yaml`
- `cpp/koopman_mpc/weights/koopman_v4_latent.json`

### 3.2 Python 验证

```bash
python3 tests/test_latent_qp_matrices.py
python3 tests/test_v4_encode_reference.py
python3 tests/export_latent_qp_cpp_ref.py
```

### 3.3 C++ 构建与验证

```bash
cd cpp/koopman_control && mkdir -p build && cd build
cmake .. -DKOOPMAN_BUILD_QP_VERIFY=ON
cmake --build . -j
./verify_latent_qp \
  ../../koopman_mpc/weights/koopman_v4_latent.yaml \
  ../../../eval_out/latent_qp_cpp_ref 20
```

### 3.4 C++ QP-MPC 调用示例

```cpp
#include "koopman_control/mpc_controller_qp.hpp"

koopman_control::MpcConfig cfg;
cfg.horizon = 20;
cfg.opt_control_steps = 2;
cfg.dt = 1.0f;

koopman_control::LatentMpcQpConfig qp_cfg;
qp_cfg.w_z = 1.f;
qp_cfg.w_u = 1e-4f;
qp_cfg.w_du = 0.05f;

koopman_control::KoopmanQpMpcController mpc(
    "cpp/koopman_mpc/weights/koopman_v4_latent.yaml", cfg, qp_cfg);

std::array<float, 6> state0{0, 0, 0, 1.5f, 0.f, 0.01f};
std::vector<std::array<float, 6>> ref_window(21);
// ... 填充参考 ...
auto [u0, cost] = mpc.solveStep(state0, ref_window);
```

---

## 4. 与 Adam MPC 的关系

| 项目 | `KoopmanMpcController`（现有） | `KoopmanQpMpcController`（新增） |
|------|----------------------------------|----------------------------------|
| 代价空间 | 物理 \((x,y,\psi,u,v,r)\) | 潜空间 \(z\)（Tier-1） |
| 预测 | 每迭代 ONNX rollout | 预计算 Θ 矩阵乘法 |
| 求解器 | Adam + 数值/自动梯度 | 投影梯度 QP |
| 实时性 | 依赖 ONNX 迭代次数 | 主要成本：encode + matvec + PGD |
| 最优性 | 局部最优 | 凸 QP 全局最优（Tier-1） |

**Tier-2**（物理跟踪 + decoder 线性化 + LTV 位姿）尚未实现，见设计文档中的 \(C_{\mathrm{ltv}}\) 小节。

---

## 5. 替换为 OSQP（可选增强）

在 `latent_mpc_qp.cpp` 中将 `solve()` 内 PGD 换为：

1. 组装 `P=H`, `q=f`（\(f = \Theta^\top Q(\Gamma z_0 + \xi - Z^{\mathrm{ref}})\)）；
2. 约束 `l ≤ I \tilde U ≤ u` 与差分矩阵行；
3. 调用 OSQP；接口保持 `LatentMpcQpSolution` 不变。

推荐在 `horizon≥20`、需硬约束严格满足时切换。

---

## 6. 维度速查（v4 默认）

| 符号 | 值 |
|------|-----|
| \(n_z\) | 48 |
| \(n_u\) | 4 |
| \(N\) | 20 |
| \(\bar A\) | \(48\times 48\) |
| \(\bar B\) | \(48\times 4\) |
| \(\Theta\) | \(960 \times 80\) |
| QP 决策维 | 80（或 blocking 后更少） |
