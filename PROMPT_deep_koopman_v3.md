# Cursor 提示词：Deep-Koopman v3 —— 扩展物理字典到 3 阶项，攻克 u-bias 漂移与高速强耦合段

> **本文件是 Cursor cloud agent 的自主任务说明。**
> 在 cursor.com/agents 起新 agent 时，**Base branch** 选当前所在分支
> （或合入 master 后的 master），任务描述只需输入一行：
>
> > 请严格按 `PROMPT_deep_koopman_v3.md` 执行，全程自主运行，§7 的 12 条
> > 阈值全部 ✅ 之前不要停下；按 §8 调参规则迭代，最多 6 轮。
>
> agent 会自动读本文件并按 §0–§11 执行。

## 0. 开工 5 分钟：先读懂现状

你接手的是已完成 v2、且 9.4 验收 PASS 的分支。**第一件事是按下面顺序读完这 5 个文件**，再开始改代码：

1. `README.md`（v2 用法 + v1 vs v2 实测表）
2. `train_koopman_v2.py`（v2 训练管线全貌：dataset、损失、curriculum、EMA、composite best）
3. `eval_koopman.py`（量化评估管线全貌：per-step CSV、summary.json、divergence 指标、`--compare` verdict）
4. `test_analysis/v2/v2_summary.json`（v2 在 test 上的所有 aggregate / divergence / channel_bias / tail / fit_quality）
5. `test_analysis/v2/v2_per_step_metrics.csv`（v2 每步的逐通道 RMSE / bias，能看清 u 通道线性漂移）

跑一条命令确认环境可用：

```
python3 eval_koopman.py --ckpt checkpoints/koopman_v2_best.pth --data koopman_test.npz --pred_len 20 --tag v2_check --out_dir /tmp/v2_check
```

期待输出末尾 `vel_rmse@20=0.01606`、`slope_loglog=0.6695`、`instability_score=1.2363`，与 `v2_summary.json` 一致。若不一致就先排查环境/数据，再开始 v3。

## 1. 任务定义（一句话）

在不改 `koopman.py`、`test_and_plot.py`、任何 `koopman_*.npz` 的前提下，**新增 `koopman_v3.py`**、**增量改 `train_koopman_v2.py` 与 `eval_koopman.py`**，把 Koopman 物理字典从 5 阶（`u|u|, v|v|, r|r|, vr, ur`）扩到 **5 + 11 = 16** 阶（加 11 个 3 阶 atom），并以 `eval_koopman.py` 落盘数字证明 §7 那 12 条硬阈值同时满足。

## 2. v2 现状（必须背熟，因为 v3 阈值全部以此为基线）

`test_analysis/v2/v2_summary.json` 在 `koopman_test.npz, pred_len=20` 上：

```
aggregate:
  vel_rmse_mean      = 0.01037
  vel_rmse_step_1    = 0.001847
  vel_rmse_step_5    = 0.007284
  vel_rmse_step_10   = 0.010982
  vel_rmse_step_20   = 0.016059
  u_rmse_step_20     = 0.015805     ← u 通道独吞 ~98% 合成速度误差
  v_rmse_step_20     = 0.002847
  r_rmse_step_20     = 0.000653
  acc_rmse_mean      = 0.010454
  traj_xy_rmse_step_20 = 0.020081
divergence:
  ratio_step20_over_step1 = 8.69
  slope_loglog            = 0.6695
  instability_score       = 1.2363
  divergent_sample_pct    = 81.78%
channel_bias:
  u_bias_mean = -0.002553   ← 单步漂移 ~ -3.8e-4 m/s，线性累积
  v_bias_mean = -0.000352
  r_bias_mean ≈ 0
```

per-segment 诊断（v2 已在前一轮归因里跑过）：

- worst-5 段：test seg 7/9/11/13/16，**共同点：u_mean ≥ 3.0 m/s + (v_std + r_std + dthr) 之一偏高**。
- best 段 seg 12 vel_err@20 mean = 0.00566；worst 段 seg 7 = 0.01961 → **段间差 3.5×**。
- per-seg vel_err@20 与 `u_mean` 相关 +0.53、与 `r_std` +0.29、与 `v_std` +0.23、与 `dthr` +0.22、与 `u_std` ≈ 0。
- 26.1% 的样本 v2 反而比 v1 差（主要在 seg 13 与 seg 8）。

结论：**残留误差的本质是高速 + 偏航 + 漂角同时存在时的 3 阶非线性耦合没被字典覆盖**，导致 u 通道残留一个方向稳定的系统性漂移。这正是 v3 要修复的对象。

## 3. 接口约束（违反任意一条都视为不通过）

- **不修改** `koopman.py`、`test_and_plot.py`、`koopman_train_merged.npz`、`koopman_val.npz`、`koopman_test.npz`、`checkpoints/koopman_v1_best.pth`、`checkpoints/koopman_v2_best.pth`、`test_analysis/v2/*`。
- 不动 stats 字段命名：`state_mean(6,), state_std(6,), ctrl_mean(4,), ctrl_std(4,)`。
- checkpoint 字典沿用 `{'epoch', 'model_state_dict', 'stats', 'ema_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict', 'best_metric', 'args'}`，**新增**两键：`'model_class': 'HorizontalKoopmanModelV3'`、`'feature_dict_atoms': [16 个 atom 字符串名]`、`'latent_dim': 43`。
- YAML 导出沿用 `normalization.{dyn_mean, dyn_std, ctrl_mean, ctrl_std}` + `system_matrices.{A_weight(含 +I), A_bias, B}`，**新增** `dictionary` 字段：

  ```yaml
  dictionary:
    state_atoms: [u, v, r]
    quadratic_atoms: [u_abs_u, v_abs_v, r_abs_r, v_times_r, u_times_r]
    cubic_atoms: [uvr, u2r, v2r, ur2, vr2, u_vabs_v, v_uabs_u, r_uabs_u, r_vabs_v, uuu, vvv]
    hidden_dim: 24
    latent_dim: 43
    note: "encoder = concat(state(3), quadratic(5), cubic(11), hidden_mlp(24)); all on NORMALIZED inputs"
  ```

- `model.encode(x_dyn_norm) → z`、`model.latent_step(z, u_norm) = z + A·z + B·u`、`model.reconstruct_state(z) = z[..., :3]`、`model.spectral_radius()` 这四个方法签名与语义不变。
- `__getitem__` 用整数索引切片，禁止 Python 循环式拼装。
- `torch.load(..., weights_only=True)` 默认禁用（stats 是 numpy dict，会失败）。

## 4. `koopman_v3.py` 详细规格

### 4.1 类骨架

```python
class HorizontalKoopmanModelV3(BaseKoopmanModel):
    def __init__(self, state_dim=3, control_dim=4, hidden_dim=24, n_cubic=11, clamp_pif=5.0):
        # latent_dim = state_dim + 5 + n_cubic + hidden_dim
        # 默认 = 3 + 5 + 11 + 24 = 43
        ...
```

- `encoder_mlp = res_mlp([state_dim, 64, 64, 64, hidden_dim], dropout=0.0)`（从 `koopman` import `res_mlp` 与 `BaseKoopmanModel`，**不复制粘贴**这两个）。
- `A = nn.Linear(latent_dim, latent_dim, bias=True)`，`A.weight ~ N(0, 0.01²)`，`A.bias = 0`。
- `B = nn.Linear(control_dim, latent_dim, bias=False)`，`xavier_uniform_(gain=0.1)`。

### 4.2 16 项物理字典（顺序固定，不要改，否则 YAML 反序列化无法对齐）

输入 `x = [u_n, v_n, r_n]` 是归一化后的张量，所有 atom 在归一化空间计算：

```
quadratic[0] = u * |u|                   # u_abs_u
quadratic[1] = v * |v|                   # v_abs_v
quadratic[2] = r * |r|                   # r_abs_r
quadratic[3] = v * r                     # v_times_r
quadratic[4] = u * r                     # u_times_r

cubic[0]  = u * v * r                    # uvr
cubic[1]  = u * u * r                    # u2r
cubic[2]  = v * v * r                    # v2r
cubic[3]  = u * r * r                    # ur2
cubic[4]  = v * r * r                    # vr2
cubic[5]  = u * |v| * v                  # u_vabs_v
cubic[6]  = v * |u| * u                  # v_uabs_u
cubic[7]  = r * |u| * u                  # r_uabs_u
cubic[8]  = r * |v| * v                  # r_vabs_v
cubic[9]  = u * |u| * u   (== u²·|u|)    # uuu
cubic[10] = v * |v| * v   (== v²·|v|)    # vvv
```

`compute_pif_atoms(x) -> Tensor[..., 16]`：内部 `cat([quadratic, cubic], dim=-1)`；**返回前做 `torch.clamp(pif, -clamp_pif, clamp_pif)`** 防止训练初期罕见样本上 3 阶值爆炸（默认 5.0，CLI 可调）。clamp 不影响绝大多数样本梯度。

`encode(x)`：

```python
def encode(self, x):
    pif = self.compute_pif_atoms(x)        # (..., 16)
    h   = self.encoder_mlp(x)              # (..., hidden_dim)
    return torch.cat([x, pif, h], dim=-1)  # (..., latent_dim)
```

`reconstruct_state(z) = z[..., :3]`、`latent_step(z, u) = z + self.A(z) + self.B(u)`、`spectral_radius()` 直接复用 `BaseKoopmanModel.spectral_radius`。

### 4.3 自检（必须有）

实现 `model._self_check_dict()`，用一组手算输入断言 16 维输出值（容差 1e-6）。**在 `--smoketest` 模式下必须被触发**，断言失败直接 `raise AssertionError`。手算参考：

```python
x = torch.tensor([[0.5, -0.3, 0.2]])  # 一组人工输入 [u, v, r]
expected = [
    +0.25,    # u|u|
    -0.09,    # v|v|
    +0.04,    # r|r|
    -0.06,    # vr
    +0.10,    # ur
    -0.03,    # uvr  = 0.5 * -0.3 * 0.2
    +0.05,    # u2r  = 0.5 * 0.5 * 0.2
    +0.018,   # v2r  = -0.3 * -0.3 * 0.2
    +0.02,    # ur2  = 0.5 * 0.2 * 0.2
    -0.012,   # vr2  = -0.3 * 0.2 * 0.2
    -0.045,   # u_vabs_v = 0.5 * 0.3 * -0.3
    -0.075,   # v_uabs_u = -0.3 * 0.5 * 0.5
    +0.05,    # r_uabs_u = 0.2 * 0.5 * 0.5
    -0.018,   # r_vabs_v = 0.2 * 0.3 * -0.3
    +0.125,   # uuu = 0.5 * 0.5 * 0.5
    -0.027,   # vvv = -0.3 * 0.3 * -0.3  → 注意符号：v*|v|*v = -0.3*0.3*-0.3 = +0.027；实际等价于 v^3 = -0.027
]
# 注意 vvv: v·|v|·v = -0.3 * 0.3 * -0.3 = +0.027，因此期望值 +0.027（不是 -0.027）。
# 请在实现时按 v·|v|·v 直接计算，并把 expected[-1] 改为 +0.027；自检公式以代码定义为准。
```

> 自检脚本应当用「按定义重算」的方式生成 expected，而非硬编码上表——这样字典扩展 / 顺序调整时只需改一次表即可。

### 4.4 forward 期间数值卫生

每次 `encode` 后，如果 `z.norm(dim=-1).max() > 50`，打 `logger.warning`（但不中断训练）。这能尽早发现 atom 溢出。

## 5. `train_koopman_v2.py` 增量改动

1. 新增 CLI：
   - `--model {v1, v2, v3}`，默认 `v3`。
   - `--n_cubic`（默认 11）、`--clamp_pif`（默认 5.0）。
   - `--w_bias_u` / `--w_bias_v` / `--w_bias_r`（默认各 50.0）—— **替代** 现有标量 `--w_bias`：实现成 dict `{u: ..., v: ..., r: ...}`，per-channel 加权 bias 平方惩罚。原 `--w_bias` 保留为兼容别名（若同时给定 `--w_bias_u/v/r` 则后者覆盖）。
2. 工厂 `build_model(args, device)`：根据 `args.model` 实例化 v1 / v2 / v3。
3. ckpt 字典每次写时**额外**塞入：

   ```python
   ckpt['model_class'] = model.__class__.__name__
   ckpt['feature_dict_atoms'] = ['u_abs_u', 'v_abs_v', 'r_abs_r', 'v_times_r', 'u_times_r',
                                  'uvr', 'u2r', 'v2r', 'ur2', 'vr2',
                                  'u_vabs_v', 'v_uabs_u', 'r_uabs_u', 'r_vabs_v', 'uuu', 'vvv']
   ckpt['latent_dim'] = model.latent_dim
   ```

4. `export_params_to_yaml(...)` 增量加 `dictionary` 字段（详见 §3）。
5. 启动 banner 必须打印 `args.model`, `n_cubic`, `latent_dim`, `hidden_dim`, `clamp_pif`, `w_bias_{u,v,r}`, `spectral_radius` 初值。
6. **默认推荐运行命令**（agent 首次完整训练用这条，然后按 §8 自检结果迭代）：

   ```
   python3 train_koopman_v2.py --model v3 \
       --epochs 60 --pred_len_start 4 --pred_len_max 20 --pred_len_grow_every 3 \
       --batch_size 1024 --num_workers 4 --no-amp --val_max_samples 8192 \
       --w_vel 1.0 --w_acc 0.5 --w_lin 0.0 --w_stab 5.0 \
       --w_bias_u 80.0 --w_bias_v 30.0 --w_bias_r 30.0 \
       --w_l2 5e-4 \
       --gamma_step 1.20 --gamma_bias 1.10 --huber_beta 0.1 \
       --rho_max 0.999 --noise_std 0.03 --ctrl_noise_std 0.005 \
       --ema_decay 0.999 --best_metric composite \
       --run_tag v3 --out_dir test_analysis/v3
   ```

7. **v2 回归保护**：跑 `python3 train_koopman_v2.py --model v2 --smoketest` 必须通过；且按 v2 历史命令训练得到的 ckpt 在 `eval_koopman.py` 上读出来的结果必须**与 `test_analysis/v2/v2_summary.json` 数字一致**（容差 1e-9）。

## 6. `eval_koopman.py` 增量改动

1. `load_model_from_ckpt(path)`：根据 `ckpt.get('model_class', 'HorizontalKoopmanModel')` 自动 dispatch：
   - `HorizontalKoopmanModel` → 从 `koopman` 导入；
   - `HorizontalKoopmanModelV3` → 从 `koopman_v3` 导入，并根据 `ckpt['args']` 还原 `n_cubic`、`clamp_pif`、`hidden_dim`。
2. 新增 per-segment 聚合：`<tag>_per_segment_metrics.csv`，每行一段 18 列：

   ```
   seg_idx, n_windows,
   vel_rmse_1, vel_rmse_5, vel_rmse_10, vel_rmse_K,
   u_rmse_K, v_rmse_K, r_rmse_K, u_bias_K, traj_xy_rmse_K,
   vel_err_p50_K, vel_err_p90_K, vel_err_p99_K, vel_err_max_K,
   ratio_K_over_1, slope_loglog, instability_score,
   u_mean, u_std, v_std, r_std, dthr_mean
   ```

   `u_mean / u_std / v_std / r_std / dthr_mean` 直接从原始段数据算（不归一化）。

3. 新增 `<tag>_per_segment_metrics.md`，标题写：

   ```
   # per-segment metrics (<tag>) — worst_seg = X (vel@K=Y); best_seg = X (vel@K=Y); 段间 mean 差距 = Z×; high_speed_mean = a; low_speed_mean = b
   ```

4. `<tag>_summary.json` 顶层**新增** `per_segment` 嵌套：

   ```json
   "per_segment": {
     "worst_seg_idx": 7, "worst_vel_rmse_K": 0.0196,
     "best_seg_idx": 12, "best_vel_rmse_K": 0.00566,
     "ratio_worst_over_best": 3.47,
     "per_seg_vel_rmse_K_std": 0.0042,
     "high_speed_seg_mean": 0.01525,
     "low_speed_seg_mean":  0.00910,
     "high_speed_seg_count": 9,
     "low_speed_seg_count":  9
   }
   ```

   `high_speed` 定义：原始段 `u_mean > 3.0 m/s`。

5. `--compare` 模式**额外**生成：
   - `compare_per_segment_bar.png`：x 轴 = 段 idx（按 v2 段误差降序固定顺序），y 轴 = vel_rmse@K，多 ckpt 并排柱状。这张图必须能一眼看出 v3 是否**专挑 v2 的差段在改善**。
   - `compare_u_bias_per_step.png`：把每个 ckpt 的 u_bias_per_step（20 个值）叠图，**这是证明 v3 把 u 漂移斜率压平的核心图**。
   - `compare_summary.md` 末尾新增「3 阶字典效果归因」段落，规则化判定（脚本里 if/else 实现即可）：
     - `u_bias_drift_rate_v3 / v2 < 0.6` → ✅ `u_bias` 漂移斜率显著改善；
     - `worst_seg_vel_rmse_K_v3 / v2 < 0.80` → ✅ worst-seg 改善；
     - `(high_speed_mean_v3 / v2) / (low_speed_mean_v3 / v2) < 0.95` → ✅ 高速段改善幅度 > 低速段。

6. `--compare` 的 verdict 部分**必须**把 §7 的 12 条阈值逐条用 `✅` / `❌` 打印，最后给一个 `**PASS** / **FAIL**` 字符串。

7. 回归保护：`python3 eval_koopman.py --ckpt checkpoints/koopman_v2_best.pth --tag v2 --out_dir /tmp/v2_regress` 的输出（除新增 `per_segment` 字段外）必须**与现有 `test_analysis/v2/v2_summary.json` 逐位一致**。

## 7. 验收硬阈值（12 条，全部满足才算 v3 PASS）

在 `koopman_test.npz, pred_len=20` 上：

### 7.1 全局（5 条，必须全部满足）

| # | 指标 | v2 当前 | v3 要求 |
|---|---|---|---|
| G1 | `aggregate.vel_rmse_step_20` | 0.01606 | **≤ 0.01285**（↓ ≥ 20%）|
| G2 | `aggregate.u_rmse_step_20` | 0.01580 | **≤ 0.01185**（↓ ≥ 25%）|
| G3 | `|channel_bias.u_bias_mean|` | 0.00255 | **≤ 0.00153**（↓ ≥ 40%）|
| G4 | `divergence.slope_loglog` | 0.6695 | **≤ 0.6695**（不退化）|
| G5 | `divergence.instability_score` | 1.2363 | **≤ 1.2363**（不退化）|

### 7.2 段间（4 条，必须全部满足）

| # | 指标 | v2 当前 | v3 要求 |
|---|---|---|---|
| S1 | `per_segment.worst_vel_rmse_K`（test seg 7） | 0.01961 | **≤ 0.01569**（↓ ≥ 20%）|
| S2 | `per_segment.ratio_worst_over_best` | 3.47 | **≤ 2.78**（↓ ≥ 20%）|
| S3 | `per_segment.high_speed_seg_mean` | 0.01525 | **≤ 0.01144**（↓ ≥ 25%）|
| S4 | v3 vs v1 退化样本占比 | 26.1%（v2）| **≤ 20%** |

### 7.3 不准退化（3 条，必须全部满足）

| # | 指标 | v2 当前 | v3 要求 |
|---|---|---|---|
| N1 | `aggregate.vel_rmse_step_1` | 0.001847 | **≤ 0.00240**（退化 ≤ 30%）|
| N2 | `aggregate.traj_xy_rmse_step_20` | 0.02008 | **≤ 0.02108**（退化 ≤ 5%）|
| N3 | `spectral_radius` (I+A) | ~0.999 | **≤ 1.005** |

**12 条任一 ❌ 即视为 FAIL，必须按 §8 继续调参重训，直到全部 ✅**。

## 8. 调参迭代规则（必读，防止「看着差不多」就停）

- 第一次按 §5.6 默认命令训练 60 epoch。
- 训完自动跑 `--compare v1:v1 v2:v2 v3:v3`，看 §7 那 12 条结果。
- 如果某条 ❌，按下表对症下药，**最多迭代 6 次**；每轮把 ckpt 备份为 `checkpoints/koopman_v3_run0K_best.pth` 留档：

  | 失败的阈值 | 优先尝试的调参 |
  |---|---|
  | G2（u_rmse 没降） | 提高 `--w_bias_u` 到 150 或 200；提高 `--n_cubic` 到 14（额外加 `u·v·|r|`, `v·u·|r|`, `r·u·|v|`） |
  | G3（u_bias 没降） | `--w_bias_u` 加倍；或加新 atom `sign(u)`、`|u|` 等线性漂移补偿项 |
  | G4 / G5（slope/inst 退化） | 提高 `--noise_std` 到 0.04；提高 `--rho_max` 收紧到 0.998；`--w_stab` 加到 10 |
  | S1 / S3（worst/high-speed 没降） | 把 `--w_vel` 在高速段加权（按段 `u_mean` 重采样训练样本，权重 ∝ `u_mean²`） |
  | S4（退化样本占比高） | 降 `--noise_std` 到 0.02；同时 `--w_bias_u` 加倍补偿 |
  | N1（vel@1 退化太多） | 降 `--gamma_step` 到 1.10；降 `--noise_std` 到 0.02 |

- 每次调参后**必须**重新跑完整 60 epoch 训练 + 自动 eval（不要从 ckpt resume 一两个 epoch 就交差）。
- 6 次迭代后仍 FAIL，把所有 run0K 的 `summary.json` 关键字段汇总成 `test_analysis/v3_iterations.md`，给出诊断与下一步建议，然后停止并报告。

## 9. 工程化 / 自验收清单

跑这两条命令必须分别在 CPU 1 分钟内通过：

```
python3 train_koopman_v2.py --model v3 --smoketest          # 触发 _self_check_dict
python3 eval_koopman.py --smoketest                          # eval 端点正常
```

完整训练日志（`logs/train_v2_*.log`）每 5 epoch 一行，列：

```
ep | lr | pl | L_total | L_vel | L_acc | L_lin | L_bias | val_vel_rmse_mean | val_vel_rmse@K | val_u_rmse@K | val_slope_loglog | val_instability_score | spec_radius
```

训练结束自动调用：

```python
eval_koopman.evaluate_one(
    ckpt='checkpoints/koopman_best.pth',
    data='koopman_test.npz', pred_len=20,
    tag='v3', out_dir='test_analysis/v3',
)
```

并打印 `=== QUANTITATIVE VERDICT (test set) ===` 块（沿用 `quantitative_verdict_block` 函数，但在内部补上 12 条阈值逐条 ✅/❌）。

最后自动跑：

```
python3 eval_koopman.py --compare \
    checkpoints/koopman_v1_best.pth:v1 \
    checkpoints/koopman_v2_best.pth:v2 \
    checkpoints/koopman_v3_best.pth:v3 \
    --data koopman_test.npz --pred_len 20 \
    --out_dir test_analysis/compare_v1_v2_v3
```

落盘 `compare_summary.md` + `compare_per_segment_bar.png` + `compare_u_bias_per_step.png`。

## 10. 不要做的事

- **不修改** `koopman.py`、`test_and_plot.py`、`koopman_*.npz`、`checkpoints/koopman_v1_best.pth`、`checkpoints/koopman_v2_best.pth`、`test_analysis/v2/*`。
- 不要把 atom 实现成可学权重的小网络——必须是闭式公式，否则 YAML 部署无法机械复刻。
- 不要为了凑阈值偷偷增大 `hidden_dim`——`--n_cubic ≥ 11` 这个真实物理字典扩展是 v3 的灵魂；可以微调 `hidden_dim` 但不能用它替代 cubic atom。
- 不要默认 `weights_only=True`。
- 不要在 `__getitem__` 里写 Python 循环。
- 不要把位置 / 航向塞进 Koopman 隐空间。
- 不要在 v3 PASS 之前就修改任何 v2 产物文件。
- **不要因为「看着差不多」就停**——12 条全部 ✅ 才算交差，否则按 §8 继续迭代。

## 11. 交付物清单

- `koopman_v3.py`（新文件，含 `HorizontalKoopmanModelV3` + `compute_pif_atoms` + `_self_check_dict`）。
- `train_koopman_v2.py` 增量改（`--model {v1,v2,v3}` 工厂、per-channel `--w_bias_{u,v,r}`、ckpt 写入 `model_class/feature_dict_atoms/latent_dim`、yaml 写入 `dictionary` 字段、banner 打印新维度、v2 行为回归不变）。
- `eval_koopman.py` 增量改（dispatch、per-segment CSV/MD、`per_segment` summary 字段、`--compare` 两张新图、12 条阈值 verdict、v2 回归不变）。
- `checkpoints/koopman_v3_best.pth`、`checkpoints/koopman_v3_best.yaml`。
- `test_analysis/v3/` 完整产物（per_step / per_sample / per_segment / summary.json / md / 6 张原图 + 1 张 bar）。
- `test_analysis/compare_v1_v2_v3/` 三角对比产物。
- 迭代过程留档：`checkpoints/koopman_v3_run0{1..K}_best.pth`，最终选 PASS 的那个复制成 `koopman_v3_best.pth`。
- `README.md` 顶部新增 `How to train v3` / `How to evaluate v3` 两小节，每节 3 行命令足矣。
- PR 描述包含：
  1. `test_analysis/compare_v1_v2_v3/compare_summary.md` 全文；
  2. `compare_per_segment_bar.png` 与 `compare_u_bias_per_step.png` 内联（HTML img 标签 + 绝对路径）；
  3. §7 的 12 条阈值逐条 ✅ / ❌ 表；
  4. **3 阶字典效果归因段落**：把 `per_segment` 与 `bias_per_step` 数据对照 v2，量化回答「是 u_bias 漂移率改善了？是 worst-seg 改善了？是高速段相对低速段改善幅度更大？」——三问必须用数字回答，不准凭图描述「改善」。

直接开始实现。遇到二选一时优先选「让 worst-seg `vel_rmse@K` 更低 + `u_bias` 漂移斜率更平」的那个。所有数值结论必须来自 `eval_koopman.py` 落盘文件，不准凭图描述「改善」。
