# Cursor 提示词：Deep-Koopman v3 收尾 —— A 方案（不动模型架构，只补 v/best/采样三件事）

> **本文件是 Cursor cloud agent 的自主任务说明。**
> 在 cursor.com/agents 起新 agent 时，**Base branch** 选当前所在分支
> （`cursor/deep-koopman-v3-3rd-order-9442` 或其后续 merge 进 master 的分支），
> 任务描述只需输入一行：
>
> > 请严格按 `PROMPT_deep_koopman_v3_planA.md` 执行，全程自主运行，§5 的
> > 14 条阈值全部 ✅ 之前不要停下；按 §6 调参规则迭代，最多 3 轮。
>
> agent 会自动读本文件并按 §0–§9 执行。

---

## 0. 开工 5 分钟：先读懂现状

你接手的是已完成 v3、且 §7 12 条阈值中 **10 ✅ / 2 ❌** 的分支（`run02` 是当前
best）。**第一件事是按下面顺序读完这 5 个文件**，再开始改代码：

1. `PROMPT_deep_koopman_v3.md`（v3 原始规格，§3 接口约束、§7 阈值表必须背熟）
2. `test_analysis/v3_iterations.md`（10 轮迭代结论 + 为什么 G4/S4 失败的诊断）
3. `test_analysis/v3/v3_summary.json`（v3 best 的全部 aggregate / per_segment / divergence）
4. `test_analysis/v3/v3_per_step_metrics.csv`（v3 逐步 u_bias 翻号、v 通道 RMSE 抬高的微观证据）
5. `test_analysis/compare_v1_v2_v3/compare_summary.md`（v1↔v2↔v3 的官方 verdict）

跑一条命令确认环境可用：

```
python3 eval_koopman.py --ckpt checkpoints/koopman_v3_best.pth --data koopman_test.npz --pred_len 20 --tag v3_check --out_dir /tmp/v3_check
```

期待输出末尾 `vel_rmse@20=0.01107`、`u_rmse@20=0.01010`、`|u_bias|=3.79e-4`、
`slope_loglog=0.7116`、`instability_score=1.2812`，与 `v3_summary.json` 一致。
若不一致就先排查环境/数据，再开始 A 方案。

---

## 1. 任务定义（一句话）

在**不改 `koopman.py`、`koopman_v3.py`、`test_and_plot.py`、任何 `koopman_*.npz`、
任何 v2 产物**的前提下，**仅增量改 `train_koopman_v2.py` 与 `eval_koopman.py`**，
通过下面三件协同的工程改造把 v3 的 §7 12 条阈值从 10/12 拉到 **12/12 PASS**，
并额外把 §5.4 的 2 条 v 通道保护阈值也带上：

* **A1**：把 `--w_bias_v` 从 30 调到 80~120，**专门压平被 11 个 cubic atom
  放大的 v 通道误差**（v3 v_rmse@20 从 v2 的 0.00285 抬到 0.00452，+59%
  是结构性问题，不修就永远拿不到 G4）。
* **A2**：把 `best_metric=composite` 的复合权重**显式加入 `slope_loglog`
  与 `degraded_pct_vs_v1`**，让 ckpt 选择阶段就直接对 G4 / S4 施压；
  并新增一个**离线 ckpt 重选脚本**，可以**不重训**就从已有 run0{1..10} 的
  per-epoch ckpt 池里挑出新口径下的最优。
* **A3**：训练 sampler 改成 **per-segment 加权 `WeightedRandomSampler`**，
  权重 `∝ (u_std² + 0.5 r_std² + ε)`，**专挑 v3 worst-seg（seg 3、seg 7）
  这类高 u_std/r_std 段做加强**——这是 S4 退化样本的主因。

三件事**协同**：A1 解决 v 通道回退、A3 解决 worst-seg 与 S4、A2 用合成指标
保证 ckpt 选择不再被 vel@K 一个口径绑架。

---

## 2. v3 现状（必须背熟，本 A 方案阈值全部以此为基线）

`test_analysis/v3/v3_summary.json` 在 `koopman_test.npz, pred_len=20` 上：

```
aggregate:
  vel_rmse_mean      = 0.00702
  vel_rmse_step_1    = 0.00122
  vel_rmse_step_5    = 0.00455
  vel_rmse_step_10   = 0.00724
  vel_rmse_step_20   = 0.01107  ✅ G1 (≤0.01285)
  u_rmse_step_20     = 0.01010  ✅ G2 (≤0.01185)
  v_rmse_step_20     = 0.00452  ⚠ 比 v2(0.00285) 退化 +59% —— 隐性回退
  r_rmse_step_20     = 0.00076
  traj_xy_rmse_step_20 = 0.01420 ✅ N2 (≤0.02108)

divergence:
  ratio_step20_over_step1 = 9.10
  slope_loglog            = 0.7116  ❌ G4 (≤0.6695，超 6.3%)
  instability_score       = 1.2812  ❌ G5 (≤1.2363，超 3.6%)
  divergent_sample_pct    = 90.5%   ⚠ 比 v2(81.8%) 抬高 8.7pp

channel_bias:
  u_bias_mean = +3.79e-4   ✅ G3 (≤0.00153) 但翻号、step1→step20 振荡
  v_bias_mean = +1.54e-4   ⚠ v2 是 -3.52e-4，方向翻号
  r_bias_mean = +4.75e-5

per_segment:
  worst_seg_idx = 3 (u_mean=2.91, u_std=0.29 全段最大, r_std=0.0226)
  worst_vel_rmse_K = 0.01419  ✅ S1 (≤0.01569)
  ratio_worst_over_best = 2.22 ✅ S2 (≤2.78)
  high_speed_seg_mean = 0.00867 ✅ S3 (≤0.01144)
  low_speed_seg_mean  = 0.01077

degraded_pct_vs_v1 = 22.94%  ❌ S4 (≤20%，超 2.94pp)
spectral_radius (I+A) = 0.9979 ✅ N3 (≤1.005)
```

**两条 ❌ 的本质归因**（来自 `v3_iterations.md §3` + 本次 per-step 重读）：

1. **G4 / G5 退化**：v3 用 16 atom 把 step1~5 huber loss 压到几乎 0，
   step10~20 全靠 `(I+A)^k z_0` 线性外推（`spec_radius≈0.998`），
   ratio 自然 ≈ exp(0.7·log K)。**但 v 通道 RMSE 抬高 59%**
   实际上把 step10~20 的总 RMSE 拉高，是 G4 直接的可拆解原因；只压 v
   通道就能把 slope 拉回。
2. **S4 退化样本占比高**：worst_seg 从 v2 的 seg 7 漂到 seg 3，
   后者特征是 `u_std=0.29`（全段最大）+ `r_std=0.0226`。10 轮迭代里
   一直按 `u_mean²` 加权（或干脆均匀采样），从未对 `u_std/r_std` 加权，
   seg 3 这类「速度抖动大」的段一直没被强训。

---

## 3. 接口约束（违反任意一条都视为不通过）

> 在 PROMPT_deep_koopman_v3.md §3 **基础上**追加，原有约束不放宽。

* **不修改** `koopman.py`、`koopman_v3.py`、`test_and_plot.py`、`koopman_*.npz`、
  `checkpoints/koopman_v1_best.pth`、`checkpoints/koopman_v2_best.pth`、
  `checkpoints/koopman_v3_best.pth`、`checkpoints/koopman_v3_run0{1..10}_best.pth`、
  `test_analysis/v2/*`、`test_analysis/v3/*`、`test_analysis/v3_run0{1..10}/*`、
  `test_analysis/compare_v1_v2_v3/*`、`test_analysis/v3_iterations.md`。
* 必须**新建** `checkpoints/koopman_v3a_best.pth` + `koopman_v3a_best.yaml`
  与 `test_analysis/v3a/`、`test_analysis/compare_v3_v3a/`，不要覆盖 v3 任何产物。
* ckpt 字典沿用 PROMPT_deep_koopman_v3.md §3 定义；`'args'` 里多出的
  `w_bias_v_eff` 必须能被 `eval_koopman.py` 正确还原。
* `--model v3` 默认行为**完全不变**（v3 默认参数下复跑应当与 run02 数值一致）。
* A2 的新指标 `degraded_pct_vs_v1` 必须**只用 val 集**算（避免训练直接对齐
  test 的 S4 指标，导致泄漏）；test 上的 S4 仅作为最终验收。
* A3 的 `WeightedRandomSampler` 必须用 `KoopmanVoyageDataset.seg_idx`（已存在），
  不允许在 `__getitem__` 内写 Python 循环。

---

## 4. 三件事的详细规格

### 4.1 A1 — per-channel bias 权重重平衡

`train_koopman_v2.py` CLI 改动（**只动默认值与文档，不增不减 flag**）：

* `--w_bias_v` 默认从 30 改为 **100**；保留 CLI 可覆盖。
* `--w_bias_u` 默认保持 80；`--w_bias_r` 默认保持 30。
* banner 必须显式打印 `effective w_bias_{u,v,r}` 与「相对 PROMPT_v3 默认值的
  比例」。

理论支撑（必须在新 CLI 注释里写一句）：v3 的 16-atom 字典里**含 v 的
cubic 项**多达 7 项（`uvr, v2r, vr2, u_vabs_v, v_uabs_u, r_vabs_v, vvv`），
对 v 的梯度灵敏度比 v2 的 5-atom 字典翻倍以上；若不同步抬高 `w_bias_v`，
v 通道一定回退——这是 v3_summary.json 已经实证的事实，不是猜测。

### 4.2 A2 — 复合 best_metric + 离线 ckpt 重选

**第 1 步：训练内的 best 选择口径**（`train_koopman_v2.py`）。

新增 best_metric 选项 `composite_v3a`，定义为：

```python
# val 集上算（保护 G4 与 S4）
cur_metric = (
    val_metrics["val/vel_rmse_mean"]
    * max(1.0, val_metrics["val/instability_score"])
    * (1.0 + 2.0 * max(0.0, val_metrics["val/slope_loglog"] - 0.65))
    * (1.0 + 5.0 * max(0.0, val_metrics["val/degraded_pct_vs_v1_val"] / 100.0 - 0.18))
)
```

含义（缺一不可）：

* 基础项：v3 现有 `vel_rmse_mean · max(1, instability)`，保 G1+G5。
* slope 项：超过 0.65 后**线性加罚**（系数 2.0），把 G4 显式纳入选择。
* degraded 项：超过 18% 后**5 倍线性加罚**，留 2pp 安全垫给 test。

`val_metrics["val/slope_loglog"]` 与 `val_metrics["val/degraded_pct_vs_v1_val"]`
必须每个 epoch 在 val 集上算出，复用 `eval_koopman.py` 的现有实现
（用 `from eval_koopman import compute_divergence_metrics, compute_degraded_pct`
**复用，不复制**）。注意 `degraded_pct_vs_v1_val` 需要在每次 val 时把
`checkpoints/koopman_v1_best.pth` 加载一次并 cache，避免重复 IO。

`--best_metric` 新增枚举值 `composite_v3a`，默认仍为 `composite`（即 v3 行为
不变），只有 A 方案训练时 CLI 显式传 `--best_metric composite_v3a`。

**第 2 步：离线 ckpt 重选脚本**（新建 `scripts/reselect_best_v3a.py`）。

* 输入：`--ckpt_glob 'checkpoints/koopman_v3_run*_best.pth'`，
  `--data koopman_val.npz`，`--out checkpoints/koopman_v3a_reselect_best.pth`。
* 遍历每个 ckpt → 在 val 集上跑 `evaluate_one(...)` → 用上面的
  `composite_v3a` 公式算分 → 取最低 → 复制为 `koopman_v3a_reselect_best.pth`。
* 落盘 `test_analysis/v3a/reselect_table.md`，列：`ckpt | vel_rmse_mean |
  slope | inst | degraded_pct_val | composite_v3a`。
* 这一步**不重训**，是 A2 的"低成本兜底"，与 A1+A2+A3 完整重训
  得到的 `koopman_v3a_best.pth` 并存（两个 ckpt 都跑一遍 eval、二选一即可）。

### 4.3 A3 — per-segment 加权 sampler

`train_koopman_v2.py` 新增 CLI：

* `--seg_resample {none, u_mean2, u_var_r_var}`，默认 `none`（v3 行为不变）；
  A 方案训练用 `u_var_r_var`。
* `--seg_resample_alpha`，默认 1.0（指数，控制权重 sharpness）。

实现规格（必须遵守）：

1. 训练 dataset 构造后，按 `seg_idx` 聚合得到每段的 `u_std`、`r_std`
   （在**未归一化**空间算），公式：
   ```
   w_seg = (u_std**2 + 0.5 * r_std**2 + 1e-3) ** alpha
   ```
2. 每个样本的采样权重 = 它所属段的 `w_seg`；归一化后传给
   `torch.utils.data.WeightedRandomSampler(replacement=True,
   num_samples=len(dataset))`。
3. 启动 banner 必须打印每段的 `w_seg` 及其 min/max/mean/std，避免某段
   权重失控（>10× mean 时打 `logger.warning`）。
4. **val / test loader 不动**——sampler 只在 train 阶段启用，否则评估口径就乱了。

### 4.4 eval_koopman.py 增量

* 新增 `evaluate_one(...)` 的返回字段 `degraded_pct_vs_v1`：以
  `--baseline_ckpt`（默认 `checkpoints/koopman_v1_best.pth`）的逐样本
  `vel_err@K` 为基线，统计 `v3a_vel_err@K > v1_vel_err@K` 的样本占比，
  落到 `summary.json.s4`。
* `--compare` 新增 `compare_v3_v3a` 模式：自动比较 v3 与 v3a，在
  `compare_summary.md` 里把 §5 的 14 条阈值逐条 ✅/❌（含 v3 的现状对比）。
* 回归保护：`python3 eval_koopman.py --ckpt checkpoints/koopman_v3_best.pth
  --tag v3 --out_dir /tmp/v3_regress` 的输出（除新增 `degraded_pct_vs_v1`
  字段外）必须**与现有 `test_analysis/v3/v3_summary.json` 逐位一致**。

---

## 5. 验收硬阈值（14 条，全部满足才算 v3a PASS）

在 `koopman_test.npz, pred_len=20` 上，沿用 PROMPT_deep_koopman_v3.md §7 的
G1~G5 / S1~S4 / N1~N3 共 12 条（数值不变），**额外**追加 2 条：

### 5.1 12 条原 v3 阈值

| # | 指标 | v3 实测 | v3a 要求 |
|---|---|---|---|
| G1 | `vel_rmse_step_20` | 0.01107 | **≤ 0.01285** |
| G2 | `u_rmse_step_20` | 0.01010 | **≤ 0.01185** |
| G3 | `|u_bias_mean|` | 3.79e-4 | **≤ 0.00153** |
| **G4** | `slope_loglog` | 0.7116 ❌ | **≤ 0.6695** |
| **G5** | `instability_score` | 1.2812 ❌ | **≤ 1.2363** |
| S1 | `per_segment.worst_vel_rmse_K` | 0.01419 | **≤ 0.01569** |
| S2 | `per_segment.ratio_worst_over_best` | 2.22 | **≤ 2.78** |
| S3 | `per_segment.high_speed_seg_mean` | 0.00867 | **≤ 0.01144** |
| **S4** | `degraded_pct vs v1` | 22.94% ❌ | **≤ 20%** |
| N1 | `vel_rmse_step_1` | 0.00122 | **≤ 0.00240** |
| N2 | `traj_xy_rmse_step_20` | 0.01420 | **≤ 0.02108** |
| N3 | `spectral_radius (I+A)` | 0.9979 | **≤ 1.005** |

### 5.2 新增 2 条「v 通道保护」（A1 的直接目标）

| # | 指标 | v3 实测 | v3a 要求 |
|---|---|---|---|
| V1 | `v_rmse_step_20` | 0.00452 | **≤ 0.00342**（v2 的 1.2×，即把 v 通道至少压回接近 v2） |
| V2 | `|v_bias_mean|` | 1.54e-4 | **≤ 0.00040**（≈ v2 的 1.1×） |

**14 条任一 ❌ 即视为 FAIL，必须按 §6 继续调参重训，最多 3 轮**。

---

## 6. 调参迭代规则

* **第 1 轮**：A1 + A2 + A3 默认配置，命令见 §7。同时跑 A2 的离线重选脚本
  得到 `koopman_v3a_reselect_best.pth`，两者都 eval，取阈值通过更多的那个
  存为 `checkpoints/koopman_v3a_best.pth`。
* **第 2 轮**（若 1 轮 FAIL）：按下表对症下药：

  | 失败阈值 | 优先尝试 |
  |---|---|
  | V1 / V2（v 通道没压住） | `--w_bias_v` 再加到 150；同时 `--w_bias_u` 降回 60 给 v 通道腾出梯度预算 |
  | G4（slope 没过） | A2 公式里 slope 系数 2.0 → 4.0；或 `--seg_resample_alpha` 1.0 → 0.7（弱化采样强度，恢复 ratio_step20_over_step1 的分布） |
  | G5（inst 没过） | 提高 `--noise_std` 到 0.025，提高 `--rho_max` 到 0.998 |
  | S4（退化样本没降） | `--seg_resample_alpha` 1.0 → 1.5（加强对 v3 worst-seg 的强训）；或 A2 公式里 degraded 系数 5.0 → 10.0 |
  | S1 / S3（worst/high-speed 退化） | `--seg_resample` 改回 `u_mean2`，配合 alpha=1.5 |

* **第 3 轮**仍 FAIL：把 3 轮的 `summary.json` 关键字段汇总成
  `test_analysis/v3a_iterations.md`，给出诊断（重点回答：v 通道是否压住了？
  slope 与 v_rmse 是否仍正相关？S4 是否在 worst-seg 改善但其它段抬升？），
  停止并报告。**不准超过 3 轮**——A 方案本来就是低风险微调，超过 3 轮还过
  不了，说明应该转 B 方案（非线性 latent step），不是这个 PROMPT 的职责。
* 每轮把 ckpt 备份为 `checkpoints/koopman_v3a_run0K_best.pth` 留档。

---

## 7. 默认命令（第 1 轮自动跑这两条）

### 7.1 完整重训（A1 + A2 + A3）

```
python3 train_koopman_v2.py --model v3 \
    --epochs 60 --pred_len_start 4 --pred_len_max 20 --pred_len_grow_every 3 \
    --batch_size 1024 --num_workers 4 --no-amp --val_max_samples 8192 \
    --w_vel 1.0 --w_acc 0.5 --w_lin 0.0 --w_stab 5.0 \
    --w_bias_u 80.0 --w_bias_v 100.0 --w_bias_r 30.0 \
    --w_l2 1e-4 \
    --gamma_step 1.10 --gamma_bias 1.10 --huber_beta 0.1 \
    --rho_max 0.999 --noise_std 0.02 --ctrl_noise_std 0.005 \
    --clamp_pif 30.0 \
    --seg_resample u_var_r_var --seg_resample_alpha 1.0 \
    --ema_decay 0.999 --best_metric composite_v3a \
    --run_tag v3a --out_dir test_analysis/v3a
```

> 与 run02（v3 best）的差别只有 4 行：`--w_bias_v 30→100`、新增 `--seg_resample
> u_var_r_var --seg_resample_alpha 1.0`、`--best_metric composite→composite_v3a`。
> 其它一律沿用 run02 的最佳配置，保证可比性。

### 7.2 A2 离线重选（**不重训，1 分钟内出结果**）

```
python3 scripts/reselect_best_v3a.py \
    --ckpt_glob 'checkpoints/koopman_v3_run*_best.pth' \
    --data koopman_val.npz \
    --baseline_ckpt checkpoints/koopman_v1_best.pth \
    --out checkpoints/koopman_v3a_reselect_best.pth
```

### 7.3 自动评估 + 三角对比

```
python3 eval_koopman.py \
    --ckpt checkpoints/koopman_v3a_best.pth \
    --data koopman_test.npz --pred_len 20 \
    --baseline_ckpt checkpoints/koopman_v1_best.pth \
    --tag v3a --out_dir test_analysis/v3a

python3 eval_koopman.py --compare \
    checkpoints/koopman_v2_best.pth:v2 \
    checkpoints/koopman_v3_best.pth:v3 \
    checkpoints/koopman_v3a_best.pth:v3a \
    --data koopman_test.npz --pred_len 20 \
    --out_dir test_analysis/compare_v3_v3a
```

---

## 8. 工程化 / 自验收清单

跑这三条命令必须分别在 CPU 1 分钟内通过：

```
python3 train_koopman_v2.py --model v3 --smoketest          # v3 行为不变
python3 train_koopman_v2.py --model v3 --seg_resample u_var_r_var --smoketest  # sampler 启用 + 触发段权重 banner
python3 eval_koopman.py --smoketest                          # eval 端点正常
python3 scripts/reselect_best_v3a.py --smoketest             # 离线重选脚本 smoke
```

完整训练日志每 5 epoch 一行，列扩展为：

```
ep | lr | pl | L_total | L_vel | L_acc | L_bias_u | L_bias_v | L_bias_r |
val_vel_rmse_mean | val_vel_rmse@K | val_u_rmse@K | val_v_rmse@K |
val_slope_loglog | val_instability_score | val_degraded_pct_v1 |
composite_v3a | spec_radius
```

训练结束自动调用：

```python
eval_koopman.evaluate_one(
    ckpt='checkpoints/koopman_best.pth',
    data='koopman_test.npz', pred_len=20,
    tag='v3a', out_dir='test_analysis/v3a',
    baseline_ckpt='checkpoints/koopman_v1_best.pth',
)
```

并打印 `=== QUANTITATIVE VERDICT (test set) ===` 块，含 §5 全部 14 条
逐条 ✅/❌ 与最终 `**PASS** / **FAIL**`。

---

## 9. 交付物清单

* `train_koopman_v2.py` 增量改：`--w_bias_v` 默认 100、新增 `--seg_resample
  {none,u_mean2,u_var_r_var}` 与 `--seg_resample_alpha`、新增 `--best_metric
  composite_v3a` 枚举值、val 阶段算 `slope_loglog` 与 `degraded_pct_vs_v1_val`、
  banner 多打印这几项；**v3 默认行为不变**（必须有 v3 回归保护测试）。
* `eval_koopman.py` 增量改：`evaluate_one(...)` 新增 `--baseline_ckpt` 与
  `degraded_pct_vs_v1` 字段，`--compare` 支持 v3↔v3a 14 条 verdict；
  **v3 现有产物逐位一致回归保护**。
* `scripts/reselect_best_v3a.py`（**新增**）：纯离线、不重训，
  从已有 ckpt 池里按 `composite_v3a` 重选 best。
* `checkpoints/koopman_v3a_best.pth` + `koopman_v3a_best.yaml`、
  `checkpoints/koopman_v3a_reselect_best.pth`、
  `checkpoints/koopman_v3a_run0{1..K}_best.pth`（每轮迭代留档）。
* `test_analysis/v3a/`（per_step / per_sample / per_segment / summary.json /
  md / 全图）+ `test_analysis/v3a/reselect_table.md` +
  `test_analysis/compare_v3_v3a/` 三件对比产物 +（若 3 轮 FAIL）
  `test_analysis/v3a_iterations.md` 诊断。
* `README.md` 顶部新增 `How to train v3a` / `How to evaluate v3a` 两小节，
  每节 3 行命令足矣。
* PR 描述包含：
  1. §5 全部 14 条阈值逐条 ✅/❌ 表（v2 / v3 / v3a 三列）；
  2. `compare_v3_v3a/compare_summary.md` 全文；
  3. **A1+A2+A3 三件事的「数字归因」段落**：分别量化回答
     「v 通道压住了多少？」「composite_v3a 选 best 与 vel_rmse 选 best 差几
     个 epoch、最终阈值差几条？」「per-segment 加权后 seg 3/7 改善多少、
     有没有牺牲其它段？」；
  4. 离线重选 `koopman_v3a_reselect_best.pth` 与完整重训
     `koopman_v3a_best.pth` 的对比表（说明最终选了哪一个、为什么）。

## 10. 不要做的事

* **不修改** PROMPT_deep_koopman_v3.md §10 列出的所有受保护文件、`koopman_v3.py`、
  以及 v3 产物（包括 `koopman_v3_best.pth` 与 `test_analysis/v3/*`）。
* 不要新增 cubic atom（A 方案的灵魂是「不动字典只动权重/采样/选择」）。
* 不要把 `--w_bias_v` 调到 > 200（梯度会失衡，v 反而 overshoot）。
* 不要在 sampler 里把权重做归一化外的非线性变换（如 softmax/log），保持
  公式 `(u_std² + 0.5 r_std² + ε)^alpha` 的可解释性。
* 不要在 `degraded_pct_vs_v1` 计算里用 test 数据（必须用 val，否则 S4 验收
  会泄漏）。
* **不要因为「看着差不多」就停**——14 条全部 ✅ 才算交差，否则按 §6 继续，
  最多 3 轮。

直接开始实现。遇到二选一时优先选「能同时压 G4 与 V1 的那个」——这两个
指标在本方案里是耦合的，任何让 v 通道继续抬高的改动都会立刻让 G4 失败。
所有数值结论必须来自 `eval_koopman.py` 落盘文件，不准凭图描述「改善」。
