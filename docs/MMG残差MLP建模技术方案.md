# MMG + 残差 MLP 混合建模技术方案

本文论证在现有 Deep-Koopman 工程的数据基础上，训练一个 **"MMG 物理模型为基线 + 残差 MLP 预测误差"** 的混合动力学模型是否可行，并给出端到端实现流程。配套：[项目指南](./项目指南.md)、[训练流程指南](./训练流程指南.md)、[仿真到实船优化指南](./仿真到实船优化指南.md)、[数据增强技术方案](./数据增强技术方案.md)。

> **结论（先说清楚）**：现有数据集**可以**支撑该方案。数据是为系统辨识设计的（chirp / PRBS / zigzag / 差速回转等激励，10 Hz 严格同步的 `(u,v,r)` + 4 维控制量），唯一需要的改造是：经典 MMG 的"螺旋桨转速 + 舵"输入要改型为**双推进器"油门% + 推力矢量角"**，且由于仓库中无船体物理参数，名义 MMG 参数需**先从数据中辨识**（最小二乘一步可解），残差 MLP 再兜底剩余误差。

---

## 1. 背景与动机

### 1.1 现有路线的局限

当前主线是纯数据驱动的 Deep-Koopman 模型（v1→v4），其"物理先验"仅体现为手工字典特征（二次阻尼 `u|u|`、科氏耦合 `vr, ur` 及三次项，见 [`koopman/model_v1_v2.py`](../koopman/model_v1_v2.py)、[`new_v4_dict_input/model_v4_dict_input.py`](../new_v4_dict_input/model_v4_dict_input.py) `:23-28`）。全工程 grep `MMG|mmg` 零命中——**没有任何参数化水动力模型**。这带来几个已知问题：

| 问题 | 表现 | 出处 |
|------|------|------|
| 外推能力弱 | 训练流形外（新航速段、大舵角）误差快速放大 | 仿真到实船优化指南 P0–P3 |
| 可解释性差 | 潜空间 48 维无法对应物理量，调参靠试错 | — |
| sim2real 差距 | 只能靠噪声增强等工程补丁，结构差异无法覆盖 | 数据增强技术方案 §1.2 |
| 数据效率低 | 任何动态变化（装载、吃水）都要重新采集+全量重训 | — |

### 1.2 混合建模的价值

"物理基线 + 学习残差"是船舶建模的标准灰箱路线：

$$
x_{t+1} = x_t + \Delta t \cdot \big[\, f_{\text{MMG}}(x_t, c_t;\, \theta) + g_{\phi}(x_t, c_t) \,\big]
$$

- $f_{\text{MMG}}$：结构正确、参数可辨识、可外推的刚体水动力模型，解释**主要动态**；
- $g_{\phi}$（残差 MLP）：吸收物理模型未建模的部分——执行器死区/延迟/左右不对称、高阶水动力、风浪流等效扰动，这些正是文档已记录的实船数据质量问题（仿真到实船优化指南 §数据质量）。

两者互补：物理部分保证分布外不发散，MLP 只需要在残差流形上拟合小量，数据需求和过拟合风险都大幅降低。

### 1.3 与现有工程的关系

本方案**不是推翻 Koopman 路线**，而是新增一条可对比、可复用基础设施的支线：

- 训练骨架（Dataset / 课程式 pred_len / EMA / Huber / z-score 统计）直接复用 [`train_v4_dict_input.py`](../new_v4_dict_input/train_v4_dict_input.py)；
- 评估协议复用 [`koopman/evalkit.py`](../koopman/evalkit.py)，与 v4 同口径对标；
- 导出链路仿照 [`export_v4_onnx.py`](../new_v4_dict_input/export_v4_onnx.py)，可先作为仿真 plant 接入 `cpp/koopman_mpc/`；
- v4 的 16 阶物理字典本身就是"摊平的 MMG 水动力项"，特征工程可直接借用。

---

## 2. 现状分析

### 2.1 工程架构

```
rosbag 海试数据 → npz 数据集 → Koopman 模型训练 → 评估 → ONNX/YAML 导出
                                                        → C++ OSQP 潜空间 MPC → ROS 控制节点（motion.cpp）
```

- **全状态 6 维** `[x, y, yaw, u, v, r]`（[`evalkit.py`](../koopman/evalkit.py) `:120-129`），单位 m / rad / m/s / rad/s，船体系 x 前 y 左；
- **所有模型只学动力学子状态 `(u, v, r)`**，位姿由外部欧拉积分恢复（[`rollout.py`](../koopman/export/rollout.py) `:51-53`）——这与 MMG 的 3-DOF 结构完全一致；
- **控制 4 维** `[左油门, 左舵角, 右油门, 右舵角]`，油门百分制 ±100（**无 RPM 概念**），舵角 ±35°（实船）/ ±89°（仿真）；
- 原始数据 10 Hz（`data_dt=0.1`）；v4 模型步长 dt=1.0s/4.0s（stride 下采样）；**实船控制周期约 0.5s**，与模型 dt 的零阶保持矛盾是已记录的最大时序风险（仿真到实船优化指南）。

### 2.2 数据管线

[`scripts/data/bag_test.py`](../scripts/data/bag_test.py) 从两个 rosbag 话题插值到统一 10 Hz 时间轴并按机动切段：

- `/localization/fusion_pose` → `Pos(2,T)`、`Vel(2,T)`、`pqr(1,T)`、`Euler(3,T, yaw)`；
- `/system/chassis_feedback` → `Thrusters_CMD(4,T)`。

切分：train（前进/倒车/差速转/zigzag/chirp/PRBS）、val（8字/急停/U转）、test（随机航行），另有 `koopman_train_left_turn.npz` 补充原地差速左转。海试采集脚本 [`scripts/sea_trial/`](../scripts/sea_trial/) 产出完全相同字段。

### 2.3 数据集实测

对 `data/` 下全部 npz 的实际检查结果（2026-07 复核）：

| 数据集 | 段数 | 时长@10Hz | u [m/s] | v [m/s] | r [°/s] | 舵角 | 油门 |
|--------|------|-----------|---------|---------|---------|------|------|
| koopman_train_merged | 99 | **5.5 h** | [-3.26, 4.20] | ±0.46 | ±3.06 | ±35° | ±100 |
| koopman_val | 3 | 1.0 h | [-4.20, 4.20] | ±0.41 | [-2.96, 1.95] | [-25°, 35°] | ±100 |
| koopman_test | 18 | 1.0 h | [1.99, 4.02] | ±0.44 | ±2.96 | ±34° | [30, 99] |
| sim_10HZ（仿真） | 1 | 4.2 h | [-4.20, 4.20] | ±0.37 | ±3.22 | **±89°** | ±100 |
| koopman_train_left_turn | 9 | 0.5 h | **≈0（原地转）** | ≤0.23 | ≤0.78 | 0° | 差速 ±60 |

激励充分性（train_merged 统计）：

- 右旋样本 10.5% / 左旋样本 10.7%（按 r 符号，阈值 1°/s）——**左右基本对称**；
- |舵角| > 5° 占 63.5%；倒车（油门<0）占 12.1%；
- 覆盖加减速、定速、回转、zigzag、chirp、PRBS、急停、U 转、原地差速转——标准辨识激励全套。

实测发现的注意点：

1. **left_turn 集 u ≈ 0**（max 0.0005 m/s）：是纯原地差速转工况，对辨识低速/零速动态有价值，但不可混入常规航速统计；
2. **左右油门统计逐项一致**（min/max/mean/std 完全相同）：除差速机动外两舷指令同步下发，差速激励仅存在于 diff_turn / left_turn 段；
3. **v 通道幅值小**（±0.46 m/s，约为 u 的 1/9），信噪比低，训练需加权（工程分析与优化.md 已记录该问题）；
4. 实船舵角上限 35°，**>35° 的外推只能依赖 sim 数据**；
5. 文档已记录的数据质量问题：插值外推造假、fusion 定位点 ≠ 旋转中心、执行器延迟/死区/不对称（仿真到实船优化指南 §数据质量）——**这些恰恰是残差 MLP 要吸收的对象**。

---

## 3. 模型设计

### 3.1 总体结构

```mermaid
flowchart LR
  S["状态 (u,v,r)"] --> MMG["MMG 基线 f(·;θ)<br/>参数冻结"]
  C["控制 (th_p,δ_p,th_s,δ_s)"] --> MMG
  S --> MLP["残差 MLP g(·;φ)"]
  C --> MLP
  MMG --> SUM((+))
  MLP --> SUM
  SUM --> NXT["下一状态 (u,v,r)']
```

离散形式（dt 为模型步长，建议 0.5s 或 1.0s）：

$$
\begin{bmatrix} u \\ v \\ r \end{bmatrix}_{t+1}
=
\begin{bmatrix} u \\ v \\ r \end{bmatrix}_{t}
+ \Delta t \cdot f_{\text{MMG}}(u,v,r,c_t;\theta)
+ \Delta t \cdot g_{\phi}\big(\bar{u},\bar{v},\bar{r},\bar{c}_t\big)
$$

其中 $\bar{\cdot}$ 表示 z-score 归一化量（MLP 在归一化空间工作，与 v4 约定一致）；MMG 部分在物理量空间工作。

### 3.2 双推进器 MMG 改型

经典 MMG 的操纵性方程（3-DOF，质心/中点原点差异吸收进参数）：

$$
\begin{aligned}
m(\dot{u} - v r) &= X_H + X_T \\
m(\dot{v} + u r) &= Y_H + Y_T \\
I_{zz}\dot{r} &= N_H + N_T
\end{aligned}
$$

**船体水动力模块**（粘性项，线性 + 二次阻尼，左右对称假设下 X 只含 u 的函数）：

$$
\begin{aligned}
X_H &= X_u u + X_{u|u|}\, u|u| \\
Y_H &= Y_v v + Y_r r + Y_{v|v|}\, v|v| + Y_{r|r|}\, r|r| + Y_{vr}\, v r \\
N_H &= N_v v + N_r r + N_{r|r|}\, r|r| + N_{vr}\, v r
\end{aligned}
$$

**推进器模块**（替代经典 MMG 的"桨 + 舵"两个模块）：每舷推进器推力为油门的函数，推力矢量随舵角旋转：

$$
T_i = k_t \cdot c_i \qquad (\text{先试线性；再试分段/二次以捕捉死区与非线性})
$$

$$
X_{T,i} = T_i \cos\delta_i, \qquad
Y_{T,i} = T_i \sin\delta_i, \qquad
N_{T,i} = \pm\, x_t \cdot T_i \sin\delta_i
$$

- $i \in \{\text{port}, \text{stbd}\}$，$c_i$ 为油门（±100），$\delta_i$ 为舵角（需转弧度）；
- $x_t$ 为推进器纵向力臂（回转中心到推进器的距离），**图纸拿不到就作为自由参数一起辨识**；
- 可辨识性说明：质量 $m$、$I_{zz}$ 与水动力导数、$k_t$ 之间存在尺度耦合，只能辨识"每单位质量/惯量"的组合量（$X_u/m$、$k_t/m$、$k_t x_t / I_{zz}$ 等）——**这对预测没有影响**，残差 MLP 也不需要知道绝对尺度；
- 扩展项（第二阶段再引入）：左右舷不对称系数 $k_t^p \ne k_t^s$、油门死区 $|c| < c_0 \Rightarrow T=0$、附加质量项（$\dot{u}$ 系数）。

### 3.3 名义参数辨识（最小二乘）

把 10 Hz 数据按一步差分 $\dot{x} \approx (x_{t+1} - x_t)/\Delta t_{\text{data}}$ 构造回归。以上方程对所有未知参数是**线性的**，例如 sway 方程：

$$
\underbrace{\dot{v} + u r}_{y_t}
=
\underbrace{\begin{bmatrix} v & r & v|v| & r|r| & vr & \tfrac{1}{m}\big(T_p s_p + T_s s_s\big) \end{bmatrix}}_{\varphi_t^\top}
\cdot \theta_Y
$$

堆叠全部样本得 $y = \Phi \theta$，最小二乘解 $\hat\theta = (\Phi^\top\Phi)^{-1}\Phi^\top y$（用 `np.linalg.lstsq`，可加岭正则）。**这一步本身即给出纯物理基线的精度数字**，是后续一切工作的基准。

工程注意：

- 差分放大噪声 → 先对 `(u,v,r)` 做轻量低通（如 5 点滑动平均）再差分，评估时仍用原始值；
- 剔除插值外推段与长饱和段（仿真到实船优化指南 §质量门控）；
- 辨识用 10 Hz 原始采样，辨识完成后再把参数冻结进目标 dt（0.5s/1.0s）的离散化中。

### 3.4 残差 MLP

- **残差定义**：$r_t = x_{t+1} - \big(x_t + \Delta t\, f_{\text{MMG}}(x_t, c_t)\big)$，在归一化空间回归 $\hat r_t / \Delta t$；
- **输入**：归一化 $[\bar u, \bar v, \bar r, \bar c_1..\bar c_4]$（7 维），可选拼接 v4 的 16 阶物理字典（现成代码 [`model_v4_dict_input.py`](../new_v4_dict_input/model_v4_dict_input.py) `:23-28`）与 1~2 拍 cmd 历史（吸收执行器延迟，见 §6.3）；
- **结构**：2~3 个残差块的 MLP，隐层 64（复用 v4 `encoder_arch="mlp"` 的 residual MLP 实现），输出 $\Delta(\bar u,\bar v,\bar r)$ 3 维。**基线已解释主要动态，MLP 只需学小量，切忌加大**；
- **初始化**：输出层零初始化，使训练起点 = 纯 MMG 基线，训练过程单调改善。

---

## 4. 训练方案

### 4.1 复用与新增文件

| 动作 | 文件 | 说明 |
|------|------|------|
| 新增 | `scripts/data/build_mmg_dataset.py` | npz → `(state, cmd, next_state)` 样本；清洗（去外推/饱和段）；stride 下采样到目标 dt；z-score 统计（沿用 v4 约定，存 ckpt） |
| 新增 | `koopman/mmg_model.py` | PyTorch 实现的 MMG 基线（参数 buffer 化，可冻结/解冻）；含 `least_squares_fit()` 辨识入口与 `step()` 离散积分 |
| 新增 | `scripts/fit_mmg_baseline.py` | 阶段 1：最小二乘辨识 + 基线精度报告（one-step / rollout RMSE） |
| 新增 | `scripts/train_mmg_residual.py` | 阶段 2：复用 `train_v4_dict_input.py` 的 Dataset / 课程 pred_len / EMA / AMP / Huber 骨架，替换模型类 |
| 新增 | `tests/test_mmg_model.py` | 单测：MMG 步骤维度/符号约定、辨识参数在仿真数据上的自洽性（sim 数据辨识→回放误差有界）、残差模型零初始化等价于基线 |
| 复用 | `koopman/evalkit.py`、`new_v4_dict_input/eval_v4_dict_input.py` | 评估协议不变 |
| 仿照 | `export_v4_onnx.py` | 新增 `export_mmg_residual_onnx.py`，rollout 接口对齐 `state0(6,), u_seq(L,4), dt → states` |

### 4.2 训练配置

- **损失**：多步 rollout 的 (u,v,r) Huber（沿用 v4 的 step 加权），**v 通道加权 2~5 倍**补偿低信噪比；可选加速度平滑项（v4 已有）；不加重构/潜空间线性损失（无 encoder）；
- **课程**：pred_len 从 2 步渐增到 20 步（与 v4 相同策略）；
- **MMG 参数**：阶段 2 全程冻结；收敛后可用 ×0.01 学习率与 MLP **联合微调**（可选消融项）；
- **归一化**：统计量只从训练集计算，与 ckpt 一起保存，导出时写入 YAML/ONNX 元数据——与现有导出链路一致；
- **噪声增强**：实船 fine-tune 阶段开启 `--noise_std 0.02 / --ctrl_noise_std 0.003`（数据增强技术方案 §2）。

### 4.3 执行器延迟处理（关键）

录制的是 CMD 而非真实推力，执行器存在延迟/死区（仿真到实船优化指南已记录）。对策按代价从低到高：

1. MLP 输入堆叠 1~2 拍 cmd 历史（增广维数小，首选）；
2. 时间平移标定：网格搜索 cmd 前移 0~5 拍，取 one-step 残差最小者；
3. 一阶执行器模型 $\dot{T} = (T_{\text{cmd}} - T)/\tau$，$\tau$ 作为可辨识参数（侵入 MMG 结构，最后做）。

### 4.4 训练/验证/测试切分

直接沿用现有切分（train_merged / val / test 三个 npz 已按机动类型物理隔离），保证与 v4 的对标同口径。left_turn 集建议**单独报告**而不混入训练损失（u≈0 工况会拉偏统计），或作为低速工况的专项验证集。

---

## 5. 评估方案

三组对照，同数据（`koopman_test.npz`）、同指标（one-step + 20s 开环 rollout 的 (u,v,r) RMSE）、同评估代码：

| 模型 | 目的 |
|------|------|
| 纯 MMG 基线（辨识后） | 物理结构单独能到多少 |
| **MMG + 残差 MLP** | 本方案 |
| v4 Koopman（`checkpoints/koopman_v4_best.yaml`） | 现有主线基准 |

报告维度：

- 分机动类型拆分（直线/回转/zigzag/倒车/原地转）；
- 20s rollout 误差增长曲线（物理模型应在分布外增长更慢）；
- 外推测试：test 集油门下限 30（train 含 -100~100 全程），重点看 **v4 训练流形边缘处两者差距**；
- 消融：cmd 历史有无、联合微调有无、MLP 容量（32/64/128）。

---

## 6. 部署与集成

1. **ONNX 导出**：仿 `export_v4_onnx.py` 导出 rollout 网络（MMG 解析式 + MLP 合成一个图），先作为**仿真 plant** 接入 `cpp/koopman_mpc/` 做闭环验证——这是风险最低的接入点；
2. **接入 MPC 的两条路**：
   - *简单路*：MPC 内部仍用 Koopman 线性模型，仅替换外部 plant，验证"更准的模型是否改善控制"；
   - *完整路*：对 MMG+MLP 逐步线性化得到时变线性模型送入 QP（解析雅可比可对 MMG 部分手写、MLP 部分自动求导），思路与现有 `pose_linearize.hpp` 的 SQP 一致；
3. **sim2real**：`sim_10HZ.npz`（舵角 ±89° 覆盖宽）预训练 → 海试数据微调；MMG 参数在两个域分别辨识、对比差异，本身就是 sim2real 差距的量化诊断；
4. **在线残差观测器**：文档 P3 已有规划（仿真到实船优化指南 §P3），与本方案天然衔接——残差 MLP 的在线输入输出偏差即观测信号，超阈回退备用控制器。

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| v 通道低信噪比 | 横向动态学不准 | 损失加权；低通预处理；参考工程分析与优化.md 的已有结论 |
| 执行器延迟未建模 | 模型学成"超前"预测 | §4.3 三级对策，先做 cmd 历史堆叠 |
| 舵角 >35° 无实船数据 | 大舵角外推不可信 | sim 数据补充；实船部署限幅 ±35° |
| 低速/原地转工况数据少 | 靠泊机动精度差 | left_turn 专项验证；必要时补采（scripts/sea_trial 有现成 profile） |
| 参数可辨识性（尺度耦合） | 只能辨识组合量 | 对预测无影响；不需绝对质量；文档中明确记录约定 |
| 差分噪声放大 | 辨识参数偏差 | 低通后差分；多步 rollout fine-tune 兜底 |
| 0.5s 控制周期 vs 模型 dt | 部署时 ZOH 失配 | 模型 dt 取 0.5s 对齐控制周期（相对 v4 的 4s 反而改善） |

---

## 8. 里程碑

| # | 内容 | 产出 | 预估 |
|---|------|------|------|
| M0 | 数据集构建脚本 + 清洗 | `build_mmg_dataset.py` + 数据报告 | 0.5 d |
| M1 | MMG 模型实现 + 最小二乘辨识 | `koopman/mmg_model.py`、`fit_mmg_baseline.py` + **基线精度数字** | 1~2 d |
| M2 | 残差 MLP 训练 + 对标评估 | `train_mmg_residual.py` + 三组对照表 | 2~3 d |
| M3 | ONNX 导出 + 仿真闭环 | `export_mmg_residual_onnx.py` + `cpp/koopman_mpc/` 闭环结果 | 1~2 d |
| M4 | （可选）联合微调、延迟标定、sim2real fine-tune | 消融报告 | 视情况 |

**M1 是决策点**：纯 MMG 基线精度直接决定残差规模——若基线已接近 v4，则 MLP 只需极小容量，整条路线性价比极高；若基线差距大，需回头检查符号约定/力臂参数/延迟，再决定是否推进。
