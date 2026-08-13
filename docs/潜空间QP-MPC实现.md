# v4 潜空间 OSQP-MPC

MPC **仅**通过 C++ [OSQP](https://osqp.org/) 求解潜空间 condensed QP；已移除 Adam / 投影梯度 / Python MPC 求解路径。

本文给出完整的原理推导与实现流程，所有公式均与代码逐一对应。涉及的核心文件：

| 文件 | 作用 |
|------|------|
| `cpp/koopman_control/src/koopman_latent_model.cpp` | 潜空间仿射动力学 + condensed 预测矩阵 (Γ, Θ, ξ) |
| `cpp/koopman_control/src/koopman_encode.cpp` | encoder：PIF 字典原子 + 残差 MLP，dyn→z |
| `cpp/koopman_control/src/koopman_decoder.cpp` | decoder：z→(u,v,r) 前向 + 解析 Jacobian |
| `cpp/koopman_control/src/latent_mpc_qp.cpp` | Tier-1/Tier-2 condensed QP 组装与 OSQP 求解 |
| `cpp/koopman_control/src/pose_linearize.cpp` | Tier-2 位姿灵敏度 Φ、偏置 b（标称轨迹线性化） |
| `cpp/koopman_control/src/mpc_controller.cpp` | `KoopmanMpcController`：SQP 外迭代、warm start、闭环仿真 |
| `cpp/koopman_control/src/motion_bridge.cpp` | 与 motion.cpp 桥接：参考重采样 + 全局→船体系变换 |
| `new_v4_dict_input/export_v4_encode_weights.py` | 从 checkpoint 导出 `koopman_v4_latent.yaml` |

## 1. 总体架构

```text
物理状态 (x,y,ψ,u,v,r)
   │ dyn=(u,v,r)，归一化 (x−μ_dyn)/σ_dyn
   ▼
encoder（16 PIF 原子 + 残差 MLP）→ z0 ∈ R^48
参考窗口 (H+1 帧) → 逐帧 encode → Z_ref ∈ R^{48H}
   ▼
condensed QP： min_U  ½ UᵀPU + qᵀU     (P = 2H, q = 2 w_z Θᵀ(z_free − Z_ref) [+ 位姿项])
   s.t.      盒约束（归一化控制幅值）+ 变化率约束
   ▼
OSQP → Ũ* → 截断/保持（move-blocking）→ 反归一化 ũ·σ_ctrl+μ_ctrl → u0 下发
```

维度约定（v4 默认值）：

| 符号 | 值 | 含义 |
|------|-----|------|
| `nz` | 48 | 潜变量维度 = 16（字典原子）+ 32（hidden） |
| `nu` | 4 | 控制通道：[油门, 舵, 油门, 舵]（ch 0/2 油门，ch 1/3 舵） |
| `N` | 10（`horizon`） | 预测步数，`dt = 4.0 s` |
| `nvar` | `N·nu` = 40 | QP 决策变量数（堆叠控制序列 Ũ） |

## 2. 潜空间模型

### 2.1 归一化

训练数据统计量随 checkpoint 保存并写入 YAML（`export_v4_encode_weights.py:120`）：

\[
\tilde{x} = \frac{x - \mu_{dyn}}{\sigma_{dyn}},\quad x=(u,v,r);
\qquad
\tilde{u} = \frac{u - \mu_{ctrl}}{\sigma_{ctrl}},\quad u\in\mathbb{R}^4
\]

C++ 侧由 `KoopmanLatentModel::normalizeControl / denormalizeControl` 与 `KoopmanMpcController::normalizeDyn` 实现。**QP 全程在归一化空间求解**，约束、权重均按归一化量书写。

### 2.2 encoder：dyn → z

`KoopmanEncoder::encode`（`koopman_encode.cpp:151`）：

1. **PIF 字典原子**（`computeAtoms16`，`koopman_encode.cpp:122`）：由 (u,v,r) 构造 16 个多项式/绝对值原子
   （u|u|, v|v|, r|r|, vr, ur, uvr, u²r, v²r, ur², vr², u|v|v, v|u|u, r|u|u, r|v|v, u|u|u, v|v|v），
   每个原子按 `clamp_pif=5` 截断，抑制外推发散；
2. **残差 MLP**：每个 `residual_conv_block` 做 `fc → ×conv 中间 tap + conv_bias → GELU → +shortcut(或恒等)`
   （`koopman_encode.cpp:159-171`；depthwise conv 在时间维 kernel=3，单点前向只取中间 tap）；
3. 输出层 `out_linear` 得 32 维 hidden；
4. 拼接：**z = [atoms(16); hidden(32)] ∈ R^48**。

### 2.3 潜空间仿射动力学

PyTorch 训练模型（`model_v4_dict_input.py:158`）：

```python
def latent_step(self, z, u):
    return z + self.A(z) + self.B(u)   # A: Linear(48,48), B: Linear(4,48,bias=False)
```

展开即**仿射线性系统**：

\[
z_{k+1} = \underbrace{(I + W_A)}_{\bar A}\, z_k + \underbrace{W_B}_{B}\, \tilde u_k + \underbrace{b_A}_{\beta}
\]

导出时 `A_bar = A.weight + I`、`bias = A.bias`、`B = B.weight`（`export_v4_encode_weights.py:105`），
C++ 单步接口为 `KoopmanLatentModel::latentStep`。

### 2.4 condensed 预测矩阵 Γ, Θ, ξ

递推展开 N 步，堆叠 Z = [z₁;…;z_N]、Ũ = [ũ₀;…;ũ_{N−1}]：

\[
z_k = \bar A^k z_0 + \sum_{j=0}^{k-1} \bar A^{k-1-j} B\,\tilde u_j + \sum_{i=0}^{k-1} \bar A^{i}\beta
\;\Longrightarrow\;
Z = \Gamma z_0 + \Theta \tilde U + \xi
\]

其中 Γ（48N×48）的第 k 行块为 Āᵏ；Θ（48N×4N）为下三角 Toeplitz，第 (k,j) 块为 Ā^{k−1−j}B；
ξ（48N）的第 k 块为 Σ_{i<k} Āⁱβ。`KoopmanLatentModel::precomputePredictionMatrices`
（`koopman_latent_model.cpp:77`）先缓存 Ā 的各次幂再填三个矩阵，**模型加载后只算一次**。

由此定义两个运行时使用量：

- **自由响应** `freeResponse(z0) = Γ z0 + ξ`（零输入下的 Z）；
- **堆叠预测** `predictStacked(z0, Ũ) = Γ z0 + ξ + Θ Ũ`。

正确性由 `tests/test_latent_qp_matrices.py` 保证：condensed 结果与 PyTorch 逐步 rollout 的最大误差 < 1e-4。

## 3. Tier-1：潜空间跟踪 QP

### 3.1 代价函数

\[
J = \sum_{k=1}^{N} w_z \|z_k - z_{ref,k}\|^2
  + \sum_{k=0}^{N-1} w_u \|\tilde u_k\|^2
  + \sum_{k=1}^{N-1} w_{du} \|\tilde u_k - \tilde u_{k-1}\|^2
\]

代入 Z = Z_free + ΘŨ（Z_free = Γz₀+ξ），记 e = Z_free − Z_ref：

\[
J = w_z\|\Theta \tilde U + e\|^2 + w_u\|\tilde U\|^2 + w_{du}\|D\tilde U\|^2
  = \tilde U^T H\,\tilde U + 2w_z e^T\Theta\,\tilde U + \text{const}
\]

\[
\boxed{H = w_z\,\Theta^T\Theta + w_u I + w_{du} D^T D}
\]

D 为相邻步差分矩阵（DŨ 的第 k 行为 ũ_k − ũ_{k−1}）。H 不依赖 z0 与参考，
`LatentMpcQpSolver::buildHessian`（`latent_mpc_qp.cpp:120`）在首次 solve 时构建并缓存
（`ensureMats` 惰性触发）。

写成 OSQP 标准形 min ½UᵀPU + qᵀU：

\[
P = 2H,\qquad q = 2 w_z\,\Theta^T (Z_{free} - Z_{ref})
\]

对应 `solve()` 中 `freeResponse → err → matvec(Θᵀ, err) × 2w_z`（`latent_mpc_qp.cpp:229-237`）。
参考潜变量由 `buildRefLatentStack` 对参考窗口逐帧 encode 得到（`mpc_controller.cpp:75`）。

### 3.2 约束

约束矩阵 A（2·nvar × nvar）分两块（`latent_mpc_qp.cpp:275-321`）：

**盒约束**（前 nvar 行，A 为单位阵）——物理幅值约束映射到归一化空间：

\[
\frac{u_{min,c} - \mu_c}{\sigma_c} \le \tilde u_{k,c} \le \frac{u_{max,c} - \mu_c}{\sigma_c}
\]

默认 u_min = [−100,−35,−100,−35]、u_max = [100,35,100,35]（油门 ±100，舵 ±35）。

**变化率约束**（后 nvar 行）——令 dũ_c = du_max,c / σ_c：

\[
k=0:\; |\tilde u_{0,c} - \tilde u_{prev,c}| \le d\tilde u_c
\qquad
k\ge 1:\; |\tilde u_{k,c} - \tilde u_{k-1,c}| \le d\tilde u_c
\]

k=0 的约束锚定**上一周期实际下发的物理控制** u_prev（先归一化），保证周期间平滑。
每通道 du_max 由 `effectiveDuMax` 解析：优先 `du_max[c]`，否则油门（ch 0/2）取
`throttle_du_max=15`、舵（ch 1/3）取 `rudder_du_max=3.5`；du_max≤0 的通道置 ±1e30（OSQP 无穷）松弛。

### 3.3 move-blocking：截断 + 保持

构造时由 `opt_control_steps` 与 `control_hold_steps` 算出 n_opt（`latent_mpc_qp.cpp:99-109`，
要求 horizon 能被 hold 整除）。QP 仍在全 horizon 上求解（代价看到整条预测轨迹），
解出后 `expandToFull`（`latent_mpc_qp.cpp:165`）做后处理：

1. 保留前 n_opt 步；
2. 第 n_opt..N−1 步全部覆写为第 n_opt−1 步的值（尾部常值保持）；
3. 若 hold>1，再把每步对齐到所属块的块首（块内常值）。

默认配置 horizon=10、opt_control_steps=2：OSQP 解出 40 维序列后，仅前 2 步生效，
后 8 步保持第 2 步的控制量——用更保守的尾部策略换取对长 horizon 解质量的鲁棒性。
最终代价 `evalCost` 用**展开后**的完整序列评估。

### 3.4 OSQP 接口细节

- **CSC 转换**：P 只取上三角（`denseUpperTriToCsc`），A 保留全部非零（`denseToCsc`）；
- **设置**：`eps_abs = eps_rel = 1e-4`，`max_iter = 4000`，`warm_start = 1`；
- **暖启动**：若传入 `u_init_tilde_stack`（上一周期解或 SQP 上一迭代），`osqp_warm_start_x`
  直接以它为初始点（`latent_mpc_qp.cpp:354`）；
- 每次 solve 重新 `osqp_setup` / `osqp_cleanup`——nvar=40 的小问题，重建开销可忽略，
  换来实现简单与无非预期状态；
- 求解失败（status≠solved）直接抛异常，由上层决定降级策略。

### 3.5 输出

取展开序列的前 4 维（第 0 步），`denormalizeControl` 反归一化得物理控制 u0
（`mpc_controller.cpp:152`），随即将整条解序列存入 `u_warm_tilde_` 供下一周期暖启动。

## 4. Tier-2：物理位姿 (x,y,ψ) 跟踪

当 `w_xy > 0` 或 `w_yaw > 0` 且 latent YAML 含 `decoder` 时启用
（`poseTrackingEnabled`，`mpc_controller.cpp:50`；旧 YAML 无 decoder 时静默禁用）。

### 4.1 非线性位姿传播

decoder 把潜变量解码回物理速度：d_m = dec_phys(z_m) = diag(σ_dyn)·MLP(z_m) + μ_dyn。
位姿由船体系欧拉积分推进（沿用 rollout 约定：速度 d_m 配艏向 ψ_{m−1}）：

\[
\begin{bmatrix} x_m \\ y_m \end{bmatrix}
= \begin{bmatrix} x_{m-1} \\ y_{m-1} \end{bmatrix}
+ \Delta t\, R(\psi_{m-1})\, \begin{bmatrix} u_m \\ v_m \end{bmatrix},
\qquad
\psi_m = \psi_{m-1} + \Delta t\, r_m
\]

非线性来自两处：decoder（GELU 网络）与旋转矩阵的 sin/cos。

### 4.2 在标称轨迹处一阶线性化

给定标称控制序列 U⁰（优先取 warm start），先算标称潜轨迹 Z⁰ = predictStacked(z0, U⁰)，
再对 m=1..N（`pose_linearize.cpp:36-60`）：

- 标称速度 d_m = `decodePhysical(z_m⁰)`；
- **decoder 解析 Jacobian** Jp_m（3×48）：逐层链式累乘——Linear 层 J←W·J，
  GELU 层 J←diag(gelu′(h))·J，最后乘 diag(σ_dyn) 得物理缩放
  （`koopman_decoder.cpp:128-182`）；
- 速度对 U 的灵敏度 V_m = Jp_m·Θ_m（3×nvar，Θ_m 为 Θ 的第 m 行块）。

位姿灵敏度递推（S^x_m 等表示 ∂p_m/∂U 的 nvar 维行向量，`pose_linearize.cpp:77-114`）：

\[
\begin{aligned}
S^x_m &= S^x_{m-1} + \Delta t(\cos\psi^0_{m-1}\, V_u - \sin\psi^0_{m-1}\, V_v)
        + \underbrace{(-u_m s - v_m c)\,\Delta t}_{\partial x_m/\partial\psi_{m-1}}\, S^\psi_{m-1}\\
S^y_m &= S^y_{m-1} + \Delta t(\sin\psi^0_{m-1}\, V_u + \cos\psi^0_{m-1}\, V_v)
        + \underbrace{(u_m c - v_m s)\,\Delta t}_{\partial y_m/\partial\psi_{m-1}}\, S^\psi_{m-1}\\
S^\psi_m &= S^\psi_{m-1} + \Delta t\, V_r
\end{aligned}
\]

其中第三项是旋转线性化的关键：ψ_{m−1} 的扰动经 ∂p_m/∂ψ_{m−1} 传入位置。
Φ（3N×nvar）的第 m 行块即 (S^x_m; S^y_m; S^ψ_m)。

于是位姿误差有仿射近似：

\[
P_{xy\psi} - P_{ref} \;\approx\; \Phi\,\tilde U + b,
\qquad
b = \underbrace{(p^0 - p_{ref})}_{\text{标称误差，yaw 先 wrapAngle}} - \Phi U^0
\]

（`pose_linearize.cpp:116-131`；yaw 环绕用 `wrapAngle` 处理后再线性化。）

### 4.3 叠加进 QP（仍是凸 QP）

\[
J \mathrel{+}= \sum_{k} \left( w_{xy}(e_{x,k}^2 + e_{y,k}^2) + w_{yaw}\, e_{\psi,k}^2 \right)
= \|W^{1/2}(\Phi \tilde U + b)\|^2
\]

W = diag(wq)，每步 wq = (w_xy, w_xy, w_yaw)。叠加项：

\[
P \mathrel{+}= 2\Phi^T W \Phi,\qquad q \mathrel{+}= 2\Phi^T W b
\]

实现见 `latent_mpc_qp.cpp:246-273`（先按行缩放 W·Φ 再做 Φᵀ(WΦ)，避免显式构造 W）。
位姿只作**软约束**（代价项），不加硬约束，避免不可行。

### 4.4 SQP 外迭代

线性化只在标称轨迹附近有效，长 horizon 下会失效。`KoopmanMpcController::solveStep`
（`mpc_controller.cpp:134-146`）做 `sqp_iters`（默认 2）轮：

```text
U ← warm start（或 0）
repeat sqp_iters 次：
    在 U 处 buildPoseLinearization → (Φ, b)
    solve QP（以 U 暖启动）→ U ← 解
```

每轮用新解更新标称轨迹重新线性化，近似序列二次规划。Tier-1 单独使用时只跑 1 轮。

### 4.5 关键工程处理

| 问题 | 处理 |
|------|------|
| decoder 非线性 | 导出 `decoder` 权重 + 解析 Jacobian（含 `dyn_std` 缩放） |
| 欧拉积分 sin/cos 非线性 | 在标称 ψ_k⁰ 处线性化，∂p/∂ψ 项进递推 |
| 长 horizon 线性化失效 | **SQP 外迭代**（`sqp_iters`，默认 2）更新标称轨迹 |
| yaw 角度环绕 | 标称角误差 `wrapAngle` 后再线性化 |
| 坐标系不一致 | `motion_bridge` 把全局参考变换到当前船体系（见 §5） |
| 位姿不可行 | 仅作**软约束**（代价项），不加硬约束 |

## 5. 坐标系与 motion 桥接

`KoopmanMotionMpc::solve`（`motion_bridge.cpp:99`）是 motion.cpp 的入口：

1. **参考重采样**：`resampleMotionRefToHorizon` 按 `ref_dt=1.0 s`、`ref_time_offset=0.5 s`
   对参考序列线性插值，取 t_k = k·dt（k=0..N）得 H+1 帧 `ref_window`
   （`motion_bridge.cpp:21-78`）；
2. **全局→船体系变换**（`has_pose` 时）：以当前船位为原点、当前艏向为 x 轴，

   \[
   p_{body} = R(-\psi)\,(p_{global} - p_{cur}),\qquad \psi_{body} = \mathrm{wrap}(\psi_{global} - \psi)
   \]

   速度分量 (u,v,r) 本就是船体系量，保持不变（`motion_bridge.cpp:150-170`）。
   相应地 `state0` 的位姿置零——MPC 内部一切位姿都在这个随体坐标系中；
3. 调 `solveStep`，打印耗时分解：
   `total / ref_resample / qp_solve / osqp_iters / status / cost`。

## 6. 端到端求解流程（一次控制周期）

```text
KoopmanMotionMpc::solve(in)
  ├─ buildRefWindow：重采样 + 坐标系变换
  └─ KoopmanMpcController::solveStep(state0, ref_window, u_prev)
       1. dyn=(u,v,r) 归一化 → encoder_.encode → z0            (mpc_controller.cpp:115-116)
       2. 逐帧 encode 参考 → Z_ref（buildRefLatentStack）
       3. Tier-2 开启时另建位姿参考堆叠（buildRefPoseStack）
       4. 标称 U ← u_warm_tilde_（首周期为 0）
       5. SQP 循环（Tier-1 时 1 轮）：
            ├─ buildPoseLinearization(z0, pose0, U, pose_ref)  → Φ, b   (Tier-2)
            └─ LatentMpcQpSolver::solve(z0, Z_ref, u_prev, U, pose)
                 ├─ Z_free = Γ z0 + ξ；q = 2 w_z Θᵀ(Z_free − Z_ref)
                 ├─ P = 2H（缓存）；Tier-2 时 P += 2ΦᵀWΦ, q += 2ΦᵀWb
                 ├─ 组装盒约束 + 变化率约束（归一化空间）
                 ├─ 稠密→CSC，osqp_setup → (暖启动) → osqp_solve
                 └─ expandToFull 截断/保持 + evalCost
       6. u_warm_tilde_ ← 解序列；取第 0 步反归一化 → u0_phys 下发
```

初始化（每个进程一次）：`loadMpcConfigFromYaml` 读 `mpc_config.yaml`
（`control_period` 会换算成 `control_hold_steps`；`syncHorizonWithOnnx` 可把 horizon
对齐到 ONNX plant）；`KoopmanMpcController` 构造时依次 `loadFromYaml`（latent 矩阵 +
归一化统计量）→ `precomputePredictionMatrices` → 加载 encoder / decoder 权重。

## 7. 导出厂物：`koopman_v4_latent.yaml`

由 `export_v4_encode_weights.py` 从 checkpoint 生成（JSON + YAML 双份）：

| 字段 | 内容 |
|------|------|
| `latent_dim` / `control_dim` | 48 / 4 |
| `clamp_pif` | PIF 原子截断阈值（5.0） |
| `normalization.{dyn_mean,dyn_std,ctrl_mean,ctrl_std}` | 训练统计量（dyn 取 state 的 [3:6] 即 u,v,r） |
| `koopman.{A_bar,bias,B}` | Ā = W_A+I（48×48）、β（48）、B（48×4） |
| `encoder.layers` | `residual_conv_block`（fc/conv_weight/conv_bias/shortcut）+ 输出 `linear` |
| `decoder.layers` | Linear/GELU 交替的 MLP（48→3），Tier-2 必需 |
| `horizon_default` / `dt` | 供 C++ 验证工具读取 |

## 8. 配置参数（`cpp/koopman_control/config/mpc_config.yaml`）

| 参数 | 默认 | 含义 |
|------|------|------|
| `horizon` | 10 | 预测步数 N |
| `dt` / `data_dt` | 4.0 / 0.1 | MPC 离散步长 / 数据采集步长（参考重采样跨步用） |
| `opt_control_steps` | 2 | 优化步数 n_opt，其后保持常值 |
| `control_hold_steps` | 1 | 控制保持块长（需整除 horizon） |
| `w_z` / `w_u` / `w_du` | 1.0 / 1e-4 / 0.05 | 潜空间跟踪 / 控制幅值 / 变化率权重 |
| `w_xy` / `w_yaw` | 0 / 0 | Tier-2 位姿/艏向权重（>0 启用） |
| `sqp_iters` | 2 | SQP 外迭代次数 |
| `throttle_du_max` / `rudder_du_max` | 15 / 3.5 | 每控制周期油门/舵最大变化（物理量） |
| `du_max[4]` | 0 | 逐通道覆盖（>0 时优先） |
| `u_min` / `u_max` | ±100 / ±35 | 控制盒约束（物理量） |
| `osqp_eps_abs` / `osqp_eps_rel` / `osqp_max_iter` | 1e-4 / 1e-4 / 4000 | OSQP 收敛容差与迭代上限 |

## 9. 使用

```bash
python3 new_v4_dict_input/export_v4_encode_weights.py
cd cpp/koopman_control/build && cmake .. && cmake --build .
./koopman_mpc_cpp --config ../koopman_control/config/mpc_config.yaml --smoketest
```

## 10. 验证

```bash
python3 tests/test_latent_qp_matrices.py     # condensed (Γ,Θ,ξ) vs 逐步 rollout，max_err < 1e-4
python3 tests/test_v4_encode_reference.py    # C++ encoder vs PyTorch 参考
python3 tests/test_pose_linearize.py         # decoder Jacobian + Φ 二阶收敛（float64）
# C++（含 OSQP 端到端）：构建后运行
./verify_pose_linearize cpp/koopman_mpc/weights/koopman_v4_latent.yaml
```

## 11. 说明

- **Tier-1**：潜空间跟踪；参考由 `encode(ref [u,v,r])` 构造。
- **闭环仿真 plant**：demo 仍用 ONNX rollout 推进被控对象（`simulate` 中
  `rolloutOneStepOnnx`，`mpc_controller.cpp:16`）；**优化器不用 ONNX**。
- **性能**：nvar=40 的 condensed QP，Hessian 预计算、逐周期暖启动，单次 OSQP
  求解为毫秒级（`motion_bridge` 打印的 `qp_solve_ms` 可在线观测）。
