# Cursor 提示词：重写 Deep-Koopman 训练脚本（强化速度跟踪）

> 直接把下面整段（从「角色」开始到「交付物」结束）粘贴到 Cursor 的对话框，建议在 Agent 模式下运行，并把仓库内的 `train_multistep_voyage.py`、`koopman.py`、`test_and_plot.py`、`koopman_train_merged.npz`、`koopman_val.npz`、`koopman_test.npz` 一起 @ 进去作为上下文。

---

## 角色

你是一名熟悉 Koopman 算子理论、深度学习以及水面/水下机器人 6-DOF 流体力学建模的资深研究工程师。请用 PyTorch 重写一个 deep-Koopman 多步预测训练脚本，目标是**显著提升速度（u, v, r）的多步跟踪精度，并以可量化的方式证明改进**——尤其要给出能反映「误差是否随预测步数发散」的硬指标，而不是只比一张图。同时保留与现有推理 / 部署管线（YAML 导出 + 外部欧拉积分）兼容的接口。

> 本次改写的两件强制交付物：
>
> 1. 新训练脚本 `train_koopman_v2.py`（详见第 1–8 节）。
> 2. 新评估脚本 `eval_koopman.py`（详见第 9 节）——它生成 CSV / JSON / MD / PNG 四类产物，包含 RMSE、R²、Pearson、bias、分位数、`slope_loglog`、`lyapunov_like`、`instability_score` 等量化指标，并支持多 ckpt 横向 `--compare`。**不允许只输出图、不输出数。**

## 背景与问题

现有代码：

- 模型：`koopman.py::HorizontalKoopmanModel`（物理先验字典 + 黑盒残差，隐空间 z = [x(3), pif(5), h(24)] = 32 维，残差 Koopman 转移 z_{t+1} = z + A·z + B·u）。
- 训练：`train_multistep_voyage.py`，`pred_len=20`，`dt=0.1`，损失 = 100·MSE(加速度) + 15·MSE(latent 一步 vs 编码 GT 序列) + 0.1·spectralRadius 罚。
- 推理：`test_and_plot.py`，把网络预测的 [u,v,r] 作为速度送入外部欧拉积分得到位置/航向。

目前实测**速度跟踪较差，并且经过一轮粗调改善仍不显著**——肉眼看曲线「形似而漂移」。这次我们要把改进做到**可量化、可证伪**：必须用 `eval_koopman.py`（见第 9 节）算出来的发散指标和拟合优度直接对比，而不是只看几张图。我们已经定位到几个关键缺陷，新脚本必须正面解决：

1. **没有直接的速度 MSE 损失**。当前主损失是「相邻步差分（加速度）」MSE，对**绝对速度漂移**不敏感——一条整体偏置的预测曲线照样可以有低加速度误差。
2. `loss_linear` 只在 latent 上比较「一步预测」与「直接 encode 真值」，由于 `reconstruct_state` 只是切片 z 的前 3 维，**latent 中 pif 与 h 的部分几乎没有梯度约束**，编码器容易学崩。
3. 没有 multi-step rollout 在物理量上的直接监督；多步误差只通过加速度间接传递，导致长程发散。
4. `StepLR(gamma=0.5, step_size=30)` + 固定 `pred_len=20` 没有 curriculum，模型在前期就被迫优化长时序，难以收敛到好极小值。
5. Encoder 用了 `dropout=0.1`，会让 encode(x_target_seq) 与 rollout 的 latent 不在同一分布上，进一步污染 `loss_linear`。
6. 数据集 `__getitem__` 里每次都重算 `_get_raw_state` 切片并组 numpy，CPU/IO 是热点；统计量计算 O(N·pred_len) 也偏慢。
7. 验证指标只看「加速度 MSE」，并用它作为 best-checkpoint 的筛选标准——这与「速度跟踪好」并不等价，导致选出来的 best 模型并非速度最好的那个。

## 数据规格（必须严格遵守，不得更改 npz 字段）

输入文件统一为 `np.load(path, allow_pickle=True)['datas']`，是一个 `object` 数组，每个元素是 dict（一段连续航段，10 Hz 采样，`dt = 0.1 s`）：

| 字段 | 形状 | 含义 |
|---|---|---|
| `len` | int | 该段有效长度 T |
| `Pos` | (2, T) float32 | 北/东位置 x, y（米） |
| `Euler` | (3, T) float32 | roll, pitch, yaw（弧度），仅 yaw（索引 2）使用 |
| `Vel` | (2, T) float32 | 体坐标系下的 surge u、sway v（m/s） |
| `pqr` | (1, T) float32 | 体坐标系下的 yaw rate r（rad/s） |
| `Thrusters_CMD` | (4, T) float32 | 4 路推进器指令，范围约 [0, 10] |

> 注：roll/pitch、heave、p、q 在水面任务里基本为 0，**不要**把它们当输入特征。
> 完整 6 维全状态向量约定为 `[x, y, yaw, u, v, r]`；动力学状态（Koopman 隐空间监督对象）为 `[u, v, r]`，控制为 4 路推力。
> 数据集分布：`koopman_train_merged.npz`（99 段，长度 ≈900~11000）/ `koopman_val.npz`（3 段长序列）/ `koopman_test.npz`（18 段，每段 ≈2000）。

## 与现有模型 / 推理脚本的接口约束

新脚本必须**继续使用** `koopman.py::HorizontalKoopmanModel`（`state_dim=3`、`control_dim=4`、`hidden_dim=24`），并保留下列接口与语义不变，从而无需改动 `test_and_plot.py`、YAML 部署、`bag_test.py` 等下游：

- `model.encode(x_dyn_norm) -> z (B, 32)`
- `model.latent_step(z, u_norm) -> z_next`，等价于 `z + A·z + B·u`
- `model.reconstruct_state(z) -> z[..., :3]`
- `model.spectral_radius()`
- checkpoint 字典必须包含 `{'epoch', 'model_state_dict', 'stats'}`，`stats` 字段名沿用现有：`state_mean`(6,), `state_std`(6,), `ctrl_mean`(4,), `ctrl_std`(4,)，使得 `test_and_plot.py` 现行加载逻辑（`stats['state_mean'][3:6]` 取动力学均值）保持可用。
- 保存路径：`checkpoints/koopman_latest.pth` 与 `checkpoints/koopman_best.pth`（best 由**速度指标**决定，见后）。
- 仍然提供一个 `export_params_to_yaml(model, stats, save_path)`，导出键名与现有 `train_multistep_voyage.py` 保持一致：`normalization.{dyn_mean, dyn_std, ctrl_mean, ctrl_std}` + `system_matrices.{A_weight (含 +I), A_bias, B}`。

如果你认为模型本身有更优结构，请**单独**新建 `koopman_v2.py`，但默认训练脚本仍然驱动现有 `HorizontalKoopmanModel`，新增模型放在 `--model {v1,v2}` 开关后面。

## 新训练脚本要求

文件名：`train_koopman_v2.py`。结构清晰、模块化，类型注解完整，关键步骤中文注释。

### 1. Dataset

实现 `KoopmanVoyageDataset(Dataset)`：

- 一次性把所有段拼成 `(N_total, 6)` 的 `states_full` 与 `(N_total, 4)` 的 `ctrls_full`（用 numpy 拼接 + 段起止索引数组），`__getitem__` 用纯整数索引切片，**禁止**在 `__getitem__` 里再做 `np.array([...])` 这种逐元素重组——目标是 CPU 单 worker 也能 ≥ 30k samples/s。
- 索引 `(seg_idx, t)` 仅在「`t + pred_len < seg_len`」处生成，且支持 `stride` 参数（默认 1，可调大做下采样）。
- 标准化统计：用整段 `states_full` / `ctrls_full` 一次性算 `mean/std`（不要按窗口重复累加）。统计字段名沿用 `state_mean/state_std/ctrl_mean/ctrl_std`。
- 返回 `(x_t_norm: (6,), x_target_seq_norm: (pred_len, 6), u_seq_norm: (pred_len, 4))`，dtype float32。
- 支持传入外部 `stats` 给验证 / 测试集复用训练集的归一化。

### 2. 模型加载

```python
from koopman import HorizontalKoopmanModel
model = HorizontalKoopmanModel(state_dim=3, control_dim=4, hidden_dim=24)
```

并提供 `--encoder_dropout 0.0`（默认 0），通过 `for m in model.encoder_mlp.modules(): if isinstance(m, nn.Dropout): m.p = args.encoder_dropout` 关闭/调整 dropout。

### 3. 损失（核心改动，必须全部实现并暴露权重 CLI 参数）

设 rollout 在归一化空间进行：`z0 = encode(x_t_dyn_norm)`，`z_{k+1} = latent_step(z_k, u_k_norm)`，`x_{k+1}_dyn_norm_pred = reconstruct_state(z_{k+1})`。物理量 `x_phys = x_norm * std + mean`。

| 名称 | 含义 | 默认权重 |
|---|---|---|
| `L_vel` | **Huber loss**（β=0.1）on 物理速度，`pred_phys[..., :3]` vs `gt_phys[..., :3]`，**逐步加权** w_k = γ^k（γ 默认 0.97），并对 (u, v, r) 做 per-channel 缩放 1/std_k，使三个通道贡献相当。这是主损失。 | `--w_vel 1.0` |
| `L_acc` | 物理加速度 Huber loss（用 `pred_phys`、`gt_phys` 的相邻差分 / dt），同样 γ-加权。保留是为了平滑性。 | `--w_acc 0.2` |
| `L_lin` | latent 一致性：把 GT 序列 reshape 后 `encode` 得到 `z_gt_seq`（**带 `with torch.no_grad()` 的 detach 版本可作消融**，默认不 detach），与 rollout 的 `z_pred_seq` 做 MSE。 | `--w_lin 1.0` |
| `L_recon` | 自编码恒等性：`MSE(reconstruct_state(encode(x)) , x)`。理论上恒为 0（因为 reconstruct_state 是切片），但加上可在切换到 v2 模型时复用。当前模型这一项实际为 0，不影响梯度。 | `--w_recon 0.0` |
| `L_stab` | `relu(spectral_radius(A_eff) - ρ_max)^2`，`ρ_max` 默认 1.005。 | `--w_stab 0.1` |
| `L_l2_A` | `||A_weight||_F^2 + ||B||_F^2` 轻量 L2，鼓励算子稀疏。 | `--w_l2 1e-4` |

总损失 = 上述加权和。**初始预热阶段**（前 5 epoch）建议把 `L_lin`、`L_stab` 权重线性 ramp-up（在脚本内自动完成，无需 CLI），避免 latent 一致性把 encoder 拉崩。

### 4. Curriculum

- `--pred_len_start 4`、`--pred_len_max 20`、`--pred_len_grow_every 10`（每 N epoch 把窗口 +2，直到 max）。
- 当窗口变化时**重建** Dataset / DataLoader，但**复用**训练集的 stats（保证归一化稳定）。

### 5. 优化器与调度

- `AdamW`（默认 `lr=1e-3`, `weight_decay=1e-4`），`betas=(0.9, 0.999)`。
- `CosineAnnealingWarmRestarts(T_0=20, T_mult=2)` 或 `OneCycleLR`，提供 `--scheduler {cosine, onecycle, step}` 切换；默认 `cosine`。
- 梯度裁剪 `clip_grad_norm_=1.0`。
- 支持 `torch.cuda.amp` 混合精度（`--amp` 开关，默认开），用 `GradScaler`。
- 可选 EMA：`--ema_decay 0.999`（默认开），验证用 EMA 权重，best checkpoint 同时存 EMA 和原权重，部署用 EMA。

### 6. 验证指标（决定 best）

每个 epoch 在 val 上跑完整 rollout（**不 teacher-force**），在物理空间上计算：

- `vel_rmse_step_k`（k = 1..pred_len_max），重点报告 k = 1, 5, 10, 20。
- `vel_rmse_mean = mean_k vel_rmse_step_k`。
- `acc_rmse_mean`（参考用）。
- `traj_xy_rmse_20`：用与 `test_and_plot.py` 相同的外部欧拉积分把预测速度积成位置，再算第 20 步水平距离误差。

**`koopman_best.pth` 由 `vel_rmse_mean` 最小决定**（不再用 acc loss）。所有指标写入 TensorBoard（`scalars`）和文本日志，并把每个 epoch 的 metrics dump 到 `logs/metrics_<timestamp>.jsonl`。

### 7. 工程化

- `setup_logger`、`SummaryWriter` 沿用现有风格；时间戳后缀；同时输出到文件和 stdout。
- 支持 `--resume <ckpt>` 续训（恢复 model、optimizer、scheduler、scaler、ema、epoch、best_metric）。
- 启动时打印：设备、batch、pred_len 课程、各损失权重、数据集大小、A/B 矩阵谱半径、参数量。
- 训练结束后调用 `export_params_to_yaml(...)` 把 best 模型导出为 `checkpoints/koopman_best.yaml`。
- CLI 参数全部用 `argparse`，并提供 `--config xxx.yaml` 覆盖默认值（YAML > CLI 默认；CLI 显式 > YAML）。
- 设置随机种子（`--seed 42`），保证可复现；DataLoader 的 `worker_init_fn` 也要 seed。
- DataLoader：`num_workers` 默认 8、`pin_memory=True`、`persistent_workers=True`、`prefetch_factor=4`。

### 8. 健壮性 / 自检

脚本顶部提供一个 `--smoketest` 模式：
- 只取训练集前 2 段、`pred_len=4`、`epochs=2`、`batch=16`，跑通前向/反向/验证/checkpoint/yaml 导出全流程并退出 0。
- 该模式必须能在无 GPU 的机器上 1 分钟内跑完。

### 9. 量化评估脚本（必须新增、独立、机器可读）

新增独立文件 `eval_koopman.py`（不要污染训练脚本），用于「**先看图、再看数、并且数能看出发散**」。它要满足：**只看输出文件就能定量判断速度曲线拟合好坏，尤其能看出误差随时间步发散的趋势**。

CLI：

```
python eval_koopman.py \
    --ckpt checkpoints/koopman_best.pth \
    --data koopman_test.npz \
    --pred_len 20 --dt 0.1 \
    --tag v2_run01 \
    --out_dir test_analysis/v2_run01 \
    [--compare checkpoints/koopman_v1_best.pth:v1 checkpoints/koopman_v2_best.pth:v2]
```

#### 9.1 对每个 checkpoint，必须落盘下列文件

写入 `--out_dir` 目录，命名都带 `--tag` 前缀；结构稳定，便于 Cursor 自己 `pandas.read_csv` / `json.load` 二次解析迭代调参：

1. **`<tag>_per_step_metrics.csv`** — 横轴是 step k = 1..pred_len，每一行包含**全数据集聚合**指标：

   | 列 | 含义 |
   |---|---|
   | `step` | 1..K |
   | `n_samples` | 该 step 参与统计的样本数 |
   | `vel_rmse` | sqrt(E[(u−û)² + (v−v̂)²])，水平速度合成 RMSE |
   | `vel_mae` | 平均绝对误差（同上） |
   | `vel_p50`, `vel_p90`, `vel_p99` | 速度误差分位数（看长尾） |
   | `u_rmse`, `v_rmse`, `r_rmse` | 三个通道分开的 RMSE |
   | `u_bias`, `v_bias`, `r_bias` | E[û − u]，**带正负号**——能看出系统性偏置 |
   | `u_r2`, `v_r2`, `r_r2` | 对该 step 上所有样本的 R²（决定系数） |
   | `u_corr`, `v_corr`, `r_corr` | Pearson 相关系数 |
   | `u_nrmse`, `v_nrmse`, `r_nrmse` | RMSE / std(GT)，无量纲化 |
   | `acc_rmse` | 加速度 RMSE（k≥2 才有） |
   | `traj_xy_err` | 与 `test_and_plot.py` 同款外部欧拉积分得到的 step-k 水平距离误差 |

2. **`<tag>_per_sample_step20.csv`** — 每一行是一个测试样本在 step 20 的指标（用于看长尾、找 worst case）：

   ```
   sample_idx, seg_idx, t_start,
   vel_err_step20, u_err_step20, v_err_step20, r_err_step20,
   traj_xy_err_step20, divergence_ratio, divergence_slope
   ```

3. **`<tag>_summary.json`** — **重点**，扁平结构方便 LLM 直接读：

   ```json
   {
     "tag": "v2_run01",
     "ckpt": "checkpoints/koopman_best.pth",
     "n_samples": 12345,
     "pred_len": 20,
     "dt": 0.1,
     "aggregate": {
       "vel_rmse_mean": 0.082,
       "vel_rmse_step_1": 0.012, "vel_rmse_step_5": 0.034,
       "vel_rmse_step_10": 0.061, "vel_rmse_step_20": 0.118,
       "u_rmse_step_20": 0.090, "v_rmse_step_20": 0.071, "r_rmse_step_20": 0.045,
       "acc_rmse_mean": 0.21,
       "traj_xy_rmse_step_20": 0.46
     },
     "divergence": {
       "ratio_step20_over_step1": 9.83,
       "slope_loglog": 0.71,
       "slope_linear": 0.0058,
       "lyapunov_like": 0.097,
       "auc_error_curve": 1.34,
       "monotonic_increasing": true,
       "divergent_sample_pct": 18.2,
       "instability_score": 0.41
     },
     "channel_bias": { "u_bias_mean": 0.013, "v_bias_mean": -0.004, "r_bias_mean": 0.001 },
     "tail": {
       "vel_err_step20_p50": 0.085, "vel_err_step20_p90": 0.213,
       "vel_err_step20_p99": 0.488, "vel_err_step20_max": 0.812
     },
     "fit_quality": {
       "u_r2_mean": 0.84, "v_r2_mean": 0.71, "r_r2_mean": 0.62,
       "u_corr_mean": 0.93, "v_corr_mean": 0.85, "r_corr_mean": 0.81
     }
   }
   ```

   **发散指标定义（必须实现，写在文档字符串里）**：
   - `ratio_step20_over_step1 = vel_rmse_step_20 / max(vel_rmse_step_1, 1e-6)`。理想 ≈ 1～3；> 5 强烈发散。
   - `slope_loglog`：对 `log(step)` vs `log(vel_rmse_step)` 做最小二乘线性拟合的斜率。≈ 0 表示几乎不增长，≈ 1 是线性误差累积，> 1 是超线性发散，应作为强警报。
   - `slope_linear`：对 `step` vs `vel_rmse_step` 直接最小二乘的斜率（m/s per step）。
   - `lyapunov_like = mean_k log(vel_rmse_step_{k+1} / vel_rmse_step_k)`（去掉 k=0），> 0 即指数级发散。
   - `auc_error_curve = trapz(vel_rmse_step_k, k=1..K) / K`。
   - `monotonic_increasing`：是否对 k 严格单调不降（容忍 ε=1e-4）。
   - `divergent_sample_pct`：**每条样本**自己的 `divergence_ratio = vel_err_step_K / vel_err_step_1`，统计有多少比例 > 5。
   - `instability_score = sigmoid(slope_loglog) * (1 + ratio_step20_over_step1 / 10)`，归一化到 [0, 1+]，越高越坏。

4. **`<tag>_per_step_metrics.md`** — 把上面 CSV 渲染成 markdown 表，标题里直接写 `vel_rmse@20 = X，divergence_slope_loglog = Y，instability_score = Z`，便于 PR 直接贴。

5. **绘图（沿用 `test_and_plot.py` 风格但更密）**——保存为 PNG，每张 200 dpi：
   - `<tag>_velocity_curve_grid.png`：6×3 网格，挑 18 条样本（按 step20 误差均匀分位选取，覆盖 best / median / worst），每行 3 列分别是 u、v、r 时间曲线，GT 实线、Pred 虚线，副标题写该样本的 `vel_err@20` 与 `divergence_ratio`。
   - `<tag>_error_vs_step.png`：折线图 + 半透明阴影是 [P10, P90] 区间，**线性 + 对数双 y 轴**两个子图（左线性、右 log）——log 子图能立刻看出指数发散。
   - `<tag>_error_band_per_channel.png`：u、v、r 三个子图，纵轴 RMSE，横轴 step，带 P50/P90 区间。
   - `<tag>_step20_error_hist.png`：step 20 处误差的直方图 + KDE，标出 P50/P90/P99 三条竖线。
   - `<tag>_bias_vs_step.png`：每通道 bias 随 step 的折线（带零线），用来识别系统性偏置（典型「越来越发散」常常是 bias 单调累积造成）。
   - `<tag>_trajectory_grid.png`：xy 轨迹对比，与现 `test_and_plot.py` 一致风格，但样本同样按分位选取。

#### 9.2 对比模式 `--compare`

当传入 `--compare A.pth:tagA B.pth:tagB ...`（≥ 2 个）时，**额外**生成：

1. **`compare_summary.csv`** — 每行一个 ckpt，列是 `summary.json["aggregate"]` + `summary.json["divergence"]` 拍平。Cursor 一眼能横向比。
2. **`compare_summary.md`** — 同样内容的 markdown 表，并在末尾自动给一段「**结论摘要**」（脚本里写一段简单规则即可）：
   - 哪个 ckpt `vel_rmse_step_20` 最低？
   - 哪个 ckpt `slope_loglog` 最低（最不发散）？
   - `instability_score` 是否随版本下降？
   - 若新 ckpt 的 `vel_rmse_step_1` 反而变差但 `step_20` 变好——给出提示「短程退化、长程改善，可能是过强的 latent 一致性损失」。
3. **`compare_error_vs_step.png`** — 把多个 ckpt 的 `vel_rmse_step_k` 叠在同一张图（线性 + log 双子图），每条线一个 tag。这是判断「谁更不发散」的核心图。
4. **`compare_step20_box.png`** — 多 ckpt 在 step 20 误差的 violin / box plot，看分布而不只是均值。
5. **`compare_trajectory_grid.png`** — 在同一批样本（按 v1 ckpt 的 step20 误差分位选取，固定样本 ID）上把所有 ckpt 的预测画在一起对比。

#### 9.3 与训练循环的联动（让 Cursor 自己读得懂）

- 训练脚本 `train_koopman_v2.py` 在每个 epoch 末**调用** `eval_koopman` 里的核心函数（不是 subprocess，是 import），对 `val` 集计算上述 `summary.json`，并把 `divergence.slope_loglog`、`divergence.ratio_step20_over_step1`、`divergence.instability_score` 三个标量写进 TensorBoard（`Val/Divergence/...`），同时追加进 `logs/metrics_<timestamp>.jsonl`。
- 训练结束后**自动**调用 `eval_koopman` 在 `koopman_test.npz` 上跑一次 best ckpt，落盘到 `test_analysis/<run_tag>/`。
- 训练日志末尾打印一段固定格式的「QUANTITATIVE VERDICT」块（Cursor 可以正则解析）：
  ```
  === QUANTITATIVE VERDICT (test set) ===
  vel_rmse@1=0.012  vel_rmse@5=0.034  vel_rmse@10=0.061  vel_rmse@20=0.118
  divergence: ratio20/1=9.83  slope_loglog=0.71  lyapunov_like=0.097
  instability_score=0.41  monotonic_increasing=True  divergent_sample_pct=18.2%
  bias: u=+0.013  v=-0.004  r=+0.001
  fit_quality: R²(u,v,r) = 0.84 / 0.71 / 0.62
  ========================================
  ```

#### 9.4 验收强约束（Cursor 自己跑完要给出的对比）

完成后必须执行下列**三步对比闭环**并把结果贴在 PR 里：

1. 用旧脚本 `train_multistep_voyage.py` 已有的 best ckpt 跑：
   ```
   python eval_koopman.py --ckpt checkpoints/koopman_v1_best.pth --tag v1 --out_dir test_analysis/v1
   ```
   （如果没有 v1 best ckpt，先用 `train_multistep_voyage.py` 训一个；不准跳过基线。）
2. 用新脚本训完后跑：
   ```
   python eval_koopman.py --ckpt checkpoints/koopman_best.pth --tag v2 --out_dir test_analysis/v2
   ```
3. 对比：
   ```
   python eval_koopman.py --compare checkpoints/koopman_v1_best.pth:v1 checkpoints/koopman_best.pth:v2 \
       --out_dir test_analysis/compare_v1_v2
   ```

   把生成的 `compare_summary.md` 全文 + `compare_error_vs_step.png` 贴到 PR 描述里。**新版本必须同时满足下面任一组合**才算达标，否则继续迭代：
   - `vel_rmse@20` 下降 ≥ 30% **且** `slope_loglog` 下降 ≥ 20%；或
   - `instability_score` 下降 ≥ 25% **且** `vel_rmse@20` 不上升。

#### 9.5 实现要点（避免 Cursor 走偏）

- 误差全部在**物理空间**计算（反归一化后），单位标在 CSV header 里（`u_rmse [m/s]`, `r_rmse [rad/s]`, `traj_xy_err [m]`）。
- `R²` 用 `1 - SS_res/SS_tot`，按「该 step 上所有样本的同一时刻值」计算，不要逐样本算再平均（前者更稳）。
- Pearson 用 `scipy.stats.pearsonr` 或自己实现，注意常数序列时 std=0 要返回 NaN 不要抛错。
- 所有图必须 `plt.close(fig)`，避免内存泄漏（参考 `check_dataset.py` 已有写法）。
- `monotonic_increasing` 用 `np.all(np.diff(curve) >= -1e-4)`。
- 数值稳定：`max(x, 1e-6)` 防 log(0)；`np.errstate(divide='ignore', invalid='ignore')` 包裹 R² 与 ratio 计算。
- 速度合成误差用 `sqrt((u-û)² + (v-v̂)²)` 而**非** `|u-û| + |v-v̂|`；r 单独成列，不要混进水平合成。
- 如果某 step 在某些样本上数据缺失（短段），**忽略而不是补零**，并在 `n_samples` 列体现。

## 不要做的事

- 不要修改 `koopman.py`（如需新模型，新建 `koopman_v2.py`）。
- 不要改动 `koopman_*.npz` 数据格式或字段命名。
- 不要把 yaw / 位置塞进 Koopman 隐空间——位置和航向通过外部欧拉积分恢复，与 `test_and_plot.py` 一致。
- 不要在 `__getitem__` 里写 Python 循环式的特征拼装；预先向量化好。
- 不要默认开启 `weights_only=True` 加载 checkpoint（stats 是 numpy dict，会失败）。

## 自验收

完成后请：

1. 用 `python train_koopman_v2.py --smoketest` 自测通过；用 `python eval_koopman.py --smoketest` 自测通过（评估脚本同样要支持 smoketest，仅取 1 段、`pred_len=4`）。
2. 在 `koopman_train_merged.npz` / `koopman_val.npz` 上以默认参数训练 30 epoch，附上一段日志摘要（每 5 epoch 一行：`epoch | lr | L_total | L_vel | L_acc | L_lin | val_vel_rmse_mean | val_vel_rmse@20 | val_traj_xy_rmse@20 | val_slope_loglog | val_instability_score | spec_radius`）。
3. 跑「v1 vs v2」对比闭环（见 9.4），把 `test_analysis/compare_v1_v2/compare_summary.md` 全文 + `compare_error_vs_step.png` 贴到 PR。**新版必须满足 9.4 的硬性指标门槛**。
4. 在 PR 描述里列出新增/修改文件、关键超参数、消融建议（哪些权重最关键、curriculum 是否生效），并就**发散指标**给出归因：是 `slope_loglog` 改善（曲线趋势变平），还是 `bias` 改善（系统性偏置消失），还是长尾 `vel_err_step20_p99` 改善（worst case 收敛）。

## 交付物

- `train_koopman_v2.py`（新文件，主体）
- `eval_koopman.py`（新文件，量化评估 + 对比，必须）
- 可选：`configs/koopman_v2_default.yaml`
- 可选：`koopman_v2.py`（仅在你确实换模型时）
- 不修改：`koopman.py`、`test_and_plot.py`、所有 `.npz`
- README 顶部追加「How to train v2 / How to evaluate」两节，每节 3 行命令足矣

请直接开始实现，遇到二选一的设计权衡时优先选「让速度多步 RMSE 更低、`slope_loglog` 更接近 0」的那个。所有数值结论都必须基于 `eval_koopman.py` 的输出文件，不要凭感觉描述「效果改善」。
