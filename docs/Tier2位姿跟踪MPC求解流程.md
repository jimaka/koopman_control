# Tier-2 位姿跟踪 MPC 求解流程详解

本文详细讲解 v4 潜空间 OSQP-MPC 在**开启 Tier-2 位姿跟踪**后，一个控制周期内从传感器状态到
控制量下发的完整求解过程。总体架构与 Tier-1 推导见《潜空间QP-MPC实现.md》，本文聚焦求解流程本身，
所有公式与代码逐一对应。

**启用条件**：`w_xy > 0` 或 `w_yaw > 0`，且 latent YAML 含 `decoder` 权重
（`KoopmanMpcController::poseTrackingEnabled`，`mpc_controller.cpp:50`；旧 YAML 无 decoder 时静默禁用，退化为纯 Tier-1）。

## 1. 全流程总览

```mermaid
flowchart TD
    subgraph IN["输入（随体坐标系）"]
        S0["当前状态 (u,v,r) + 位姿 (x,y,ψ)"]
        REF["参考序列 + 上一周期控制 u_prev"]
        WARM["暖启动解 U_warm（上周期的解序列）"]
    end

    IN --> P0["① 坐标系准备<br/>motion_bridge.cpp<br/>参考重采样 + 全局→船体系变换"]
    P0 --> P1["② 编码<br/>encoder: (u,v,r) → z0<br/>参考逐帧 encode → Z_ref / 位姿参考堆叠"]
    P1 --> LOOP{"③ SQP 外迭代<br/>共 sqp_iters 轮（默认 2）"}
    LOOP --> NOM["④ 标称轨迹生成<br/>Z⁰=Γz₀+ΘU+ξ → decoder 解码速度<br/>→ 欧拉积分出标称位姿 p⁰"]
    NOM --> LIN["⑤ 逐点线性化<br/>decoder 解析 Jacobian → V_m<br/>→ 灵敏度递推得 Φ、b"]
    LIN --> QP["⑥ QP 组装与求解<br/>P=2H+2ΦᵀWΦ，q=2w_zΘᵀe+2ΦᵀWb<br/>盒约束+变化率约束 → OSQP（暖启动）"]
    QP --> LOOP
    LOOP -->|迭代完成| OUT["⑦ 输出<br/>截断+尾部保持 → 取第 0 步反归一化<br/>→ u0 下发；解序列存为暖启动"]
```

一个控制周期 = ① 坐标系准备 → ② 编码 → ③ SQP 外迭代（每轮含 ④⑤⑥）→ ⑦ 输出。
以下逐节展开。

## 2. 功能模块拆解

求解链路按职责分为四层：**接口层 → 编排层 → 模型层 → 求解层**，外加贯穿始终的配置与数学工具。

```mermaid
flowchart TD
    subgraph L0["接口层"]
        MB["KoopmanMotionMpc<br/>motion_bridge.cpp<br/>参考重采样 / 坐标变换 / 耗时统计"]
    end
    subgraph L1["编排层"]
        CTL["KoopmanMpcController<br/>mpc_controller.cpp<br/>SQP 外迭代 / 暖启动 / 输出"]
    end
    subgraph L2["模型层"]
        ENC["KoopmanEncoder<br/>koopman_encode.cpp<br/>(u,v,r) → z"]
        LAT["KoopmanLatentModel<br/>koopman_latent_model.cpp<br/>Γ/Θ/ξ 预计算与预测"]
        DEC["KoopmanDecoder<br/>koopman_decoder.cpp<br/>z → (u,v,r) + Jacobian"]
    end
    subgraph L3["求解层"]
        PL["buildPoseLinearization<br/>pose_linearize.cpp<br/>标称轨迹 + Φ, b, wq"]
        QP["LatentMpcQpSolver<br/>latent_mpc_qp.cpp<br/>QP 组装 + OSQP"]
    end
    subgraph L4["配置与工具"]
        CFG["MpcConfig / mpc_config_loader<br/>参数载体与 YAML 解析"]
        MAT["detail::Matrix<br/>detail/dense_matrix.hpp<br/>稠密矩阵运算 / gelu / wrapAngle"]
    end

    MB --> CTL
    CTL --> ENC
    CTL --> PL
    CTL --> QP
    PL --> LAT
    PL --> DEC
    QP --> LAT
    CFG --> MB
    CFG --> CTL
    MAT -.-> L2
    MAT -.-> L3
```

### 2.1 模块一览

| 模块 | 文件 | 职责 | 关键接口 | 对应环节 |
|---|---|---|---|---|
| `KoopmanMotionMpc` | `motion_bridge.cpp` | motion.cpp 桥接：参考重采样、全局→船体系变换、耗时统计 | `solve(MotionSolveInput) → MotionSolveOutput` | ① |
| `KoopmanMpcController` | `mpc_controller.cpp` | 求解编排：编码、参考堆叠、SQP 外迭代、暖启动管理 | `solveStep(state0, ref_window, u_prev) → (u0, cost)` | ②③⑦ |
| `KoopmanEncoder` | `koopman_encode.cpp` | (u,v,r) → z₀（16 PIF 原子 + 残差 MLP） | `encode(dyn_norm) → z(48)` | ② |
| `KoopmanLatentModel` | `koopman_latent_model.cpp` | latent YAML 加载、Γ/Θ/ξ 预计算、自由响应/堆叠预测、控制归一化 | `loadFromYaml` / `precomputePredictionMatrices` / `freeResponse` / `predictStacked` | ④⑥ |
| `KoopmanDecoder` | `koopman_decoder.cpp` | z → (u,v,r) 前向解码 + 解析 Jacobian | `decodePhysical(z)` / `jacobianPhysical(z)` | ④⑤ |
| `buildPoseLinearization` | `pose_linearize.cpp` | 标称轨迹生成与逐点线性化 | 返回 `PoseLinearization{Φ, b, wq, valid}` | ④⑤ |
| `LatentMpcQpSolver` | `latent_mpc_qp.cpp` | Hessian 预计算、QP 组装（含位姿叠加）、约束构建、OSQP 调用 | `solve(z0, z_ref, u_prev, U_init, pose) → solution` | ⑥ |
| `MpcConfig` / `LatentMpcQpConfig` | `mpc_config.hpp` + `mpc_config_loader.cpp` | 参数载体与 YAML 解析（`control_period` 换算、horizon 对齐） | `loadMpcConfigFromYaml` / `latentQpConfigFromMpc` | 初始化 |
| `detail::Matrix` | `detail/dense_matrix.hpp` | 稠密矩阵运算（matmul/matvec/transpose）、`gelu`/`geluGrad`、`wrapAngle` | 被模型层与求解层共用 | 贯穿 |
| `KoopmanOnnxModel` | `koopman_onnx_model.cpp` | 闭环仿真 plant（**仅 demo/simulate**，优化器不依赖） | `rollout` | 仅仿真 |

### 2.2 接口层：`KoopmanMotionMpc`

- **职责**：把 motion.cpp 的全局参考与状态翻译成控制器可用的随体系输入，并把求解结果与耗时带回去；
- **输入**：`MotionSolveInput{ref, u/v/r, x/y/psi, u_prev, has_pose/has_u_prev}`；
- **输出**：`MotionSolveOutput{control, cost, horizon, timing}`；
- **内部状态**：`MotionBridgeConfig{ref_dt, ref_time_offset}` + 持有 `KoopmanMpcController` 实例（pImpl）；
- **调用频率**：每控制周期 1 次；不做任何与模型相关的计算。

### 2.3 编排层：`KoopmanMpcController`

- **职责**：整个求解流程的"总线"——编码当前状态与参考、管理标称序列与暖启动、驱动 SQP 外迭代、
  产出物理控制量；另提供 `simulate` 闭环仿真（用 ONNX plant，优化路径不用）；
- **持有**：`KoopmanLatentModel model_`、`KoopmanEncoder encoder_`、`KoopmanDecoder decoder_`、
  `LatentMpcQpSolver solver_`（构造顺序有讲究：solver_ 先于 model_ 加载构造，horizon 以配置为准）；
- **内部状态**：`u_warm_tilde_` / `has_warm_`（暖启动解），`resetWarmStart()` 可清空；
- **每周期调用**：`solveStep` 1 次，内部驱动 ③ 外迭代 `sqp_iters` 轮。

### 2.4 模型层

**`KoopmanEncoder`**：无状态权重容器（pImpl 存残差块与输出层权重）。
`encode` 一次调用 = 16 原子计算 + 2 个残差块 + 输出线性层 + 拼接，纯前向，O(nz) 级。
每周期调用次数：1（z₀）+ N（参考逐帧 encode，`buildRefLatentStack` 循环 k=0..N−1）。

**`KoopmanLatentModel`**：加载 YAML 后 `precomputePredictionMatrices` **一次性**算好
Γ（48N×48）、Θ（48N×4N）、ξ（48N），之后 `freeResponse` / `predictStacked` 只是矩阵乘。
是 Tier-1 代价、约束归一化（`ctrlMean/ctrlStd`）、标称 rollout 的共同数据来源。

**`KoopmanDecoder`**：同样无状态；`decodePhysical` 与 `jacobianPhysical` 共享同一套层遍历逻辑，
后者逐层链式累乘 Jacobian（Linear 层左乘 W，GELU 层按行乘 gelu′）。每个 SQP 轮次内被调用
2N 次（每步各一次前向 + 一次 Jacobian）。

**`buildPoseLinearization`**（自由函数）：无内部状态，输入 `(model, decoder, z0, pose0, U, pose_ref, dt, w_xy, w_yaw)`，
输出 `PoseLinearization{Φ(3N×nvar), b(3N), wq(3N), valid}`。decoder 未加载或输入维度不符时
`valid=false`，求解器自动跳过位姿项（退化 Tier-1）。

### 2.5 求解层：`LatentMpcQpSolver`

- **内部缓存**：Hessian H = w_zΘᵀΘ + w_uI + w_duDᵀD 首次 solve 时构建并缓存（`ensureMats`），
  此后每轮只做 P = 2H (+ 位姿叠加) 的拷贝与加法；
- **每次 solve 的工作**：自由响应 → q₀ →（有位姿时）P/q 叠加 → 盒约束与变化率约束 →
  CSC 转换 → `osqp_setup` /（暖启动）/ `osqp_solve` / `osqp_cleanup` → 截断保持 → `evalCost`；
- **每周期调用**：`sqp_iters` 次；每次求解 nvar=40、约束 80 行的小 QP，毫秒级。

### 2.6 配置与工具

- `MpcConfig`（业务参数：horizon、dt、权重、约束限值、Tier-2 开关）与 `LatentMpcQpConfig`
  （QP/OSQP 参数）分离，`latentQpConfigFromMpc` 负责前者→后者的抽取；
- `loadMpcConfigFromYaml` 支持 `control_period` 自动换算 `control_hold_steps`，
  `syncHorizonWithOnnx` 可把 horizon 对齐到 ONNX plant；
- `detail::Matrix` 是全链路唯一的矩阵实现（无外部 BLAS 依赖），矩阵规模小，三重循环足够。

### 2.7 模块间的数据流

```mermaid
flowchart LR
    subgraph 每周期一次
        A["state/ref/u_prev"] --> B["z0(48), Z_ref(48N),<br/>pose_ref(3N), pose0"]
    end
    subgraph 每 SQP 轮
        B --> C["标称 U"] --> D["Z⁰, d_m, p⁰<br/>Jp_m, V_m"] --> E["Φ(3N×4N), b(3N), wq"]
    end
    subgraph 每 QP 求解
        E --> F["P, q, A, l, u"] --> G["OSQP → Ũ*"]
        G --> H["截断保持 → u0 下发<br/>解序列 → 暖启动"]
    end
    H -.->|下一周期| C
```

三类缓存/复用决定了性能特征：**Γ/Θ/ξ 与 H 跨周期不变**（模型加载与首次求解时各算一次）；
**暖启动解跨周期复用**；每 SQP 轮只有 Φ、b 与 P/q 的位姿部分需要重建。

## 3. ① 坐标系准备（`motion_bridge.cpp`）

`KoopmanMotionMpc::solve` 是 motion.cpp 侧的入口，先做两件事：

**参考重采样**（`resampleMotionRefToHorizon`，`motion_bridge.cpp:58`）：参考序列按
`ref_dt=1.0 s`、`ref_time_offset=0.5 s` 线性插值，取 t_k = k·dt（k=0..N）得到 H+1 帧参考窗口，
每帧含 (x, y, ψ, u, v, r)。

**全局→船体系变换**（`buildRefWindow`，`motion_bridge.cpp:150`）：`has_pose` 时把参考位姿
变换到"以当前船位为原点、当前艏向为 x 轴"的随体坐标系：

\[
p_{body} = R(-\psi_{cur})\,(p_{global} - p_{cur}),\qquad
\psi_{body} = \mathrm{wrap}(\psi_{global} - \psi_{cur})
\]

参考中的速度分量 (u,v,r) 本就是船体系量，保持不变。相应地，送入 `solveStep` 的 `state0`
位姿直接置零（`motion_bridge.cpp:111`）——MPC 内部一切位姿量都在这个随体系中表示，
**每个控制周期重建一次坐标系**，避免全局坐标数值过大带来的精度问题。

## 4. ② 编码（`mpc_controller.cpp:114-126`）

进入 `KoopmanMpcController::solveStep` 后：

- 当前动力学状态 dyn=(u,v,r) 归一化（`normalizeDyn`），经 `encoder_.encode` 得 **z₀ ∈ R⁴⁸**；
- 参考窗口逐帧取 (u,v,r) 归一化后 encode，堆叠成 **Z_ref ∈ R^{48N}**（`buildRefLatentStack`，供 Tier-1 潜空间跟踪项）；
- Tier-2 开启时另建**位姿参考堆叠** pose_ref ∈ R^{3N}（`buildRefPoseStack`），第 k 块对应第 k 步的 (x,y,ψ)；
- 位姿初值 pose0 = (0,0,0)（随体系原点）。

同时准备**标称控制序列 U**（SQP 线性化工作点）：优先取上一周期保存的暖启动解
`u_warm_tilde_`，首周期为全零（`mpc_controller.cpp:130-132`）。

## 5. ③ SQP 外迭代框架

位姿传播含 decoder（GELU 网络）与旋转矩阵 sin/cos 两处非线性，无法直接写进凸 QP。
策略是**序列二次规划（SQP）**：在当前标称控制序列 U 处把位姿约束线性化，解一个凸 QP，
用解更新标称轨迹，重复 `sqp_iters` 轮（默认 2，Tier-1 单独使用时只跑 1 轮）。

```mermaid
flowchart LR
    U0["U ← 暖启动"] --> IT1["第 1 轮<br/>在 U 处线性化 → 解 QP"]
    IT1 --> U1["U ← 解序列"]
    U1 --> IT2["第 2 轮<br/>在新 U 处重新线性化 → 解 QP"]
    IT2 --> U2["U ← 解序列"]
    U2 --> DONE["输出"]
```

每轮 QP 都以上一轮的解暖启动 OSQP，第二轮的线性化工作点已经比暖启动更接近真解，
因此 2 轮即可显著修正长 horizon 下的线性化漂移（继续增加轮数收益递减，且每轮约多花一倍求解时间）。

## 6. ④ 标称轨迹生成（`pose_linearize.cpp:32-75`）

给定当前标称 U，分三步生成标称轨迹：

**潜空间 rollout**（精确线性传播，与 condensed 矩阵口径一致）：

\[
Z^0 = \Gamma z_0 + \Theta U + \xi
\]

**逐步解码物理速度**：对每个预测步 m=1..N，取 z_m⁰ 经 decoder 前向并还原物理量纲：

\[
d_m = \mathrm{diag}(\sigma_{dyn})\cdot \mathrm{MLP}(z_m^0) + \mu_{dyn} \in \mathbb{R}^3
\quad (u_m, v_m, r_m)
\]

**船体系欧拉积分出标称位姿**（沿用 rollout 约定：速度 d_m 配本步开始时的艏向 ψ_{m−1}）：

\[
\begin{bmatrix} x_m^0 \\ y_m^0 \end{bmatrix}
= \begin{bmatrix} x_{m-1}^0 \\ y_{m-1}^0 \end{bmatrix}
+ \Delta t\, R(\psi_{m-1}^0) \begin{bmatrix} u_m \\ v_m \end{bmatrix},
\qquad
\psi_m^0 = \psi_{m-1}^0 + \Delta t\, r_m
\]

## 7. ⑤ 逐点线性化：灵敏度 Φ 与偏置 b

目标：把"位姿误差"写成控制序列 U 的仿射函数。链条是
U →(Θ)→ z →(decoder)→ d →(欧拉积分)→ p，逐环求导再相乘。

```mermaid
flowchart LR
    U["控制序列 U<br/>(nvar = 4N)"] -->|Θ_m：潜空间灵敏度<br/>常矩阵，预计算| Z["潜变量扰动 δz_m"]
    Z -->|"Jp_m：decoder 解析 Jacobian<br/>逐点求值"| D["速度扰动 δd_m = V_m δU"]
    D -->|"欧拉积分递推<br/>含 ∂p/∂ψ 旋转项"| P["位姿扰动 δp_m<br/>Φ 的第 m 行块"]
```

**第一环：decoder 解析 Jacobian**（`koopman_decoder.cpp:128`）。decoder 是 Linear/GELU 交替的 MLP，
Jacobian 逐层链式累乘：Linear 层 J ← W·J；GELU 层 J ← diag(gelu′(h))·J（逐元素导数）；
最后乘 diag(σ_dyn) 还原物理量纲，得 Jp_m（3×48），在标称点 z_m⁰ 处求值。

**第二环：速度灵敏度**。潜空间扰动对 U 的灵敏度就是 Θ 的第 m 行块（常矩阵），故

\[
V_m = J_{p,m}\cdot \Theta_m \quad (3 \times nvar)
\]

**第三环：位姿灵敏度递推**（`pose_linearize.cpp:77-114`）。对欧拉积分逐步求导，
S^x_m、S^y_m、S^ψ_m 表示 ∂p_m/∂U 的 nvar 维行向量（初值为 0）：

\[
\begin{aligned}
S^x_m &= S^x_{m-1} + \Delta t(\cos\psi^0\, V_u - \sin\psi^0\, V_v)
        + \underbrace{(-u_m \sin\psi^0 - v_m \cos\psi^0)\,\Delta t}_{\partial x_m/\partial\psi_{m-1}}\, S^\psi_{m-1}\\
S^y_m &= S^y_{m-1} + \Delta t(\sin\psi^0\, V_u + \cos\psi^0\, V_v)
        + \underbrace{(u_m \cos\psi^0 - v_m \sin\psi^0)\,\Delta t}_{\partial y_m/\partial\psi_{m-1}}\, S^\psi_{m-1}\\
S^\psi_m &= S^\psi_{m-1} + \Delta t\, V_r
\end{aligned}
\]

第三项是关键：ψ_{m−1} 的扰动经旋转矩阵的导数传入位置——正是它把"艏向误差"与"位置误差"耦合起来，
忽略它会导致转弯工况下位姿梯度明显错误。Φ（3N × nvar）的第 m 行块即 (S^x_m; S^y_m; S^ψ_m)。

**偏置 b**（`pose_linearize.cpp:116-131`）：把仿射展开锚定在标称轨迹的实测误差上——

\[
b = \underbrace{(p^0 - p_{ref})}_{\text{标称位姿误差}} - \Phi U^0
\quad\Rightarrow\quad
e_{pose} \approx \Phi U + b
\]

yaw 分量先做 `wrapAngle` 折到 ±π 再相减，避免 2π 跳变污染梯度。
权重向量 wq 逐步取 (w_xy, w_xy, w_yaw)。

## 8. ⑥ QP 组装与 OSQP 求解（`latent_mpc_qp.cpp`）

### 8.1 目标函数：Tier-1 基底 + 位姿叠加

Tier-1 部分（H 预计算一次，不随 SQP 轮次变化）：

\[
H = w_z \Theta^T\Theta + w_u I + w_{du} D^T D,\qquad
P_0 = 2H,\qquad q_0 = 2 w_z \Theta^T (Z_{free} - Z_{ref})
\]

位姿软约束项（每轮重建）：

\[
J_{pose} = \|W^{1/2}(\Phi U + b)\|^2
\;\Longrightarrow\;
\boxed{P = P_0 + 2\Phi^T W \Phi,\qquad q = q_0 + 2\Phi^T W b}
\]

W = diag(wq)。实现上先对 Φ 按行乘 wq 再做 Φᵀ(WΦ)，避免显式构造 W（`latent_mpc_qp.cpp:246-273`）。
ΦᵀWΦ 半正定，叠加后 Hessian 仍半正定，**问题保持凸 QP**。

### 8.2 约束：只管控制量

约束矩阵 A（2·nvar × nvar）只含两类硬约束，位姿不进入约束：

| 约束 | 形式 | 说明 |
|---|---|---|
| 盒约束 | (u_min−μ)/σ ≤ ũ ≤ (u_max−μ)/σ | 物理幅值映射到归一化空间；油门 ±100、舵 ±35 |
| 变化率约束 | k=0：\|ũ₀−ũ_prev\| ≤ du_max/σ；k≥1：\|ũ_k−ũ_{k−1}\| ≤ du_max/σ | 首步锚定上一周期实际下发量；油门 15/周期、舵 3.5/周期 |

### 8.3 OSQP 求解

- P 取上三角转 CSC、A 转 CSC，每次 solve 重建 workspace（nvar=40 的小问题，开销可忽略）；
- `eps_abs=eps_rel=1e-4`、`max_iter=4000`、`warm_start=1`，并以上一轮/上一周期的 U
  `osqp_warm_start_x`——SQP 第二轮和相邻控制周期的求解因此都只需少量迭代；
- 求解失败直接抛异常，由上层决定降级。

## 9. ⑦ 输出与周期衔接（`mpc_controller.cpp:145-166`）

最后一轮 QP 解出后：

1. **截断 + 尾部保持**（`expandToFull`）：保留前 `opt_control_steps`（默认 2）步，
   第 2..N−1 步覆写为最后优化步的常值——用保守的尾部策略换鲁棒性；
2. 解序列存入 `u_warm_tilde_`，作为**下一控制周期**的暖启动与 SQP 标称初值；
3. 取第 0 步的 4 维控制，反归一化（ũ·σ_ctrl + μ_ctrl）得物理量 u0 下发；
4. `motion_bridge` 打印耗时分解：`total / ref_resample / qp_solve / osqp_iters / status / cost`，
   可在线观测单次求解毫秒级耗时。

相邻周期的时间线：

```mermaid
sequenceDiagram
    participant MB as motion_bridge
    participant CTL as KoopmanMpcController
    participant QP as LatentMpcQpSolver
    participant OSQP as OSQP

    Note over MB: 周期 t：状态 (u,v,r,ψ) 到达
    MB->>MB: 参考重采样 + 全局→船体系变换
    MB->>CTL: solveStep(state0, ref_window, u_prev)
    CTL->>CTL: encode → z0；建 Z_ref / pose_ref；U ← 暖启动
    loop sqp_iters 轮（默认 2）
        CTL->>CTL: 标称 rollout + 线性化 → (Φ, b)
        CTL->>QP: solve(z0, Z_ref, u_prev, U, pose)
        QP->>OSQP: setup（P=2H+2ΦᵀWΦ）+ warm_start(U) + solve
        OSQP-->>QP: Ũ*
        QP-->>CTL: 截断保持后的解序列 + cost
        CTL->>CTL: U ← 解序列
    end
    CTL-->>MB: u0（物理量）+ cost
    MB->>MB: 打印耗时分解，下发控制
    Note over CTL: 解序列留存，供周期 t+1 暖启动
```

## 10. 配置与调优

| 参数（`mpc_config.yaml`） | 默认 | 作用与调优 |
|---|---|---|
| `w_xy` | 0（关闭） | 平面位置跟踪权重；开启建议 0.5–2.0，过大易与艏向/潜空间项打架 |
| `w_yaw` | 0（关闭） | 艏向跟踪权重；与 `w_xy` 配合开启 |
| `sqp_iters` | 2 | SQP 外迭代轮数；>2 收益小、耗时近似线性增长 |
| `w_z` / `w_u` / `w_du` | 1.0 / 1e-4 / 0.05 | Tier-1 潜空间跟踪 / 幅值 / 平滑权重，Tier-2 开启后仍生效 |
| `throttle_du_max` / `rudder_du_max` | 15 / 3.5 | 每周期控制变化上限（物理量），限制位姿环的调节速度 |

开启 Tier-2 的前提：`koopman_v4_latent.yaml` 含 `decoder` 段（由 `export_v4_encode_weights.py`
导出）；否则 `decoder_.loaded()` 为 false，自动退化为纯 Tier-1，不报错。

## 11. 验证

```bash
python3 tests/test_pose_linearize.py            # decoder Jacobian 与 Φ 的二阶收敛性（float64 数值微分对照）
# C++ 端到端（含 OSQP）：构建后运行
./verify_pose_linearize cpp/koopman_mpc/weights/koopman_v4_latent.yaml
```

`test_pose_linearize.py` 用数值微分验证解析 Jacobian 与 Φ 的正确性（二阶收敛即梯度正确），
`verify_pose_linearize` 在 C++ 侧对同一份 latent YAML 做端到端复核，保证 Python 训练口径与
C++ 部署口径一致。

## 12. 设计要点回顾

- **位姿只作代价项（软约束）**：欠驱动系统 + 线性化近似下，硬约束可能无可行解；
  软约束永远有解，误差以代价形式被权衡（详见《潜空间QP-MPC实现.md》§4.5）；
- **线性化锚定实测误差**：b = 标称误差 − ΦU⁰，使仿射模型在标称点处零误差，SQP 轮次间逐步外推；
- **yaw 全程 wrap 处理**：参考变换、标称误差两处都先 wrapAngle，杜绝 2π 跳变；
- **H 不变、只重建位姿项**：Tier-1 的 Hessian 一次预计算，SQP 每轮只需重建 ΦᵀWΦ 与 q，
  计算开销集中在 N×nvar 规模的矩阵乘上；
- **暖启动贯穿始终**：SQP 轮次间、控制周期间都以旧解暖启动，OSQP 实际迭代次数很低。
