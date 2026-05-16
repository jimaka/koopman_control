# Deep-Koopman 速度跟踪 v2 / v3

`koopman.py` 中的物理先验 Koopman 模型 (`HorizontalKoopmanModel`) 是固定接口；
`koopman_v3.py` 中的 `HorizontalKoopmanModelV3` 扩展物理字典到 16 阶。本仓库
提供三套训练 / 评估管线，统一通过 `train_koopman_v2.py --model {v1,v2,v3}`：

* **v1 baseline**：`train_multistep_voyage.py`（保留不动，供历史对照）。
* **v2 重写**：`train_koopman_v2.py` + `eval_koopman.py`，按
  [`PROMPT_deep_koopman_rewrite.md`](./PROMPT_deep_koopman_rewrite.md)
  第 1–9 节实现。
* **v3 扩展**：`koopman_v3.py` 把物理字典从 5 阶 (`u|u|, v|v|, r|r|, vr, ur`)
  扩到 16 阶 (5 二次 + 11 三次：`uvr, u²r, v²r, ur², vr², u·|v|·v, v·|u|·u,
  r·|u|·u, r·|v|·v, u·|u|·u, v·|v|·v`)，按
  [`PROMPT_deep_koopman_v3.md`](./PROMPT_deep_koopman_v3.md)
  第 1–11 节实现。模型 `latent_dim = 3 + 5 + 11 + 24 = 43`。

> 评估必须先看 `test_analysis/<tag>/<tag>_summary.json` 与
> `compare_summary.md`，再看图。

## How to train v2

```bash
# 推荐配置（CPU 单卡 ~10 分钟即可达到 9.4 验收阈值）
python3 train_koopman_v2.py \
    --epochs 60 --pred_len_start 4 --pred_len_max 20 --pred_len_grow_every 3 \
    --batch_size 1024 --num_workers 4 --no-amp --val_max_samples 8192 \
    --w_vel 1.0 --w_acc 0.5 --w_lin 0.0 --w_stab 5.0 --w_bias 50.0 \
    --gamma_step 1.20 --gamma_bias 1.10 --huber_beta 0.1 \
    --rho_max 0.999 --noise_std 0.03 --ctrl_noise_std 0.005 \
    --best_metric composite --run_tag v2 --out_dir test_analysis/v2

# 冒烟自测（无 GPU 1 分钟内）
python3 train_koopman_v2.py --smoketest
```

训练结束会自动：
1. 把 best ckpt 导出为 `checkpoints/koopman_best.pth` /
   `checkpoints/koopman_best.yaml`（YAML 与 v1 部署接口一致）。
2. 在 `koopman_test.npz` 上跑一次 `eval_koopman.evaluate_one`，并打印
   `=== QUANTITATIVE VERDICT (test set) ===` 块。

## How to train v3

```bash
# 推荐配置（CPU 4 核 ~10 分钟即可逼近 §7 的 12 条阈值）
python3 train_koopman_v2.py --model v3 \
    --epochs 60 --pred_len_start 4 --pred_len_max 20 --pred_len_grow_every 3 \
    --batch_size 1024 --num_workers 4 --no-amp --val_max_samples 8192 \
    --w_vel 1.0 --w_acc 0.5 --w_lin 0.0 --w_stab 5.0 \
    --w_bias_u 80.0 --w_bias_v 30.0 --w_bias_r 30.0 \
    --w_l2 1e-4 --gamma_step 1.10 --gamma_bias 1.10 --huber_beta 0.1 \
    --rho_max 0.999 --noise_std 0.02 --ctrl_noise_std 0.003 --clamp_pif 30.0 \
    --ema_decay 0.999 --best_metric composite \
    --run_tag v3 --out_dir test_analysis/v3

# 冒烟自测（无 GPU 1 分钟内）—— 触发 _self_check_dict
python3 train_koopman_v2.py --model v3 --smoketest
```

## How to evaluate v3

```bash
# 单 ckpt 评估
python3 eval_koopman.py --ckpt checkpoints/koopman_v3_best.pth \
    --data koopman_test.npz --pred_len 20 --tag v3 --out_dir test_analysis/v3

# v1/v2/v3 三角对比 + §7 12 条阈值 verdict + 3 阶字典效果归因
python3 eval_koopman.py --compare \
    checkpoints/koopman_v1_best.pth:v1 \
    checkpoints/koopman_v2_best.pth:v2 \
    checkpoints/koopman_v3_best.pth:v3 \
    --data koopman_test.npz --pred_len 20 \
    --out_dir test_analysis/compare_v1_v2_v3
```

## How to evaluate

```bash
# 单 ckpt
python3 eval_koopman.py --ckpt checkpoints/koopman_v2_best.pth \
    --data koopman_test.npz --pred_len 20 --tag v2 --out_dir test_analysis/v2

# v1 vs v2 强制对比闭环（PROMPT 9.4）
python3 eval_koopman.py \
    --compare checkpoints/koopman_v1_best.pth:v1 \
              checkpoints/koopman_v2_best.pth:v2 \
    --data koopman_test.npz --out_dir test_analysis/compare_v1_v2

# 冒烟自测
python3 eval_koopman.py --smoketest
```

输出：每个 tag 一份 `*_per_step_metrics.csv`、`*_summary.json`、
`*_per_step_metrics.md`、`*_per_sample_step{K}.csv` 与六张 PNG；对比模式额外
生成 `compare_summary.md`、`compare_summary.csv`、`compare_error_vs_step.png`、
`compare_step{K}_box.png`、`compare_trajectory_grid.png`，并在 md 末尾写入
**9.4 硬性指标门槛**自动判定（PASS / FAIL）。

## 实测结果（test set, koopman_test.npz, K=20）

| metric | v1 (baseline) | v2 | **v3** | v3 vs v2 |
|---|---|---|---|---|
| `vel_rmse_step_1` [m/s] | 0.001314 | 0.001847 | **0.001216** | **-34.2%** |
| `vel_rmse_step_5` [m/s] | 0.006549 | 0.007284 | **0.004552** | **-37.5%** |
| `vel_rmse_step_10` [m/s] | 0.013021 | 0.010982 | **0.007245** | **-34.0%** |
| `vel_rmse_step_20` [m/s] | 0.025644 | 0.016059 | **0.011068** | **-31.1%** |
| `u_rmse_step_20` [m/s] | 0.022542 | 0.015805 | **0.010101** | **-36.1%** |
| `traj_xy_rmse_step_20` [m] | 0.027750 | 0.020081 | **0.014199** | **-29.3%** |
| `|u_bias_mean|` [m/s] | 0.003652 | 0.002553 | **0.000379** | **-85.2%** |
| `slope_loglog` | 0.9915 | **0.6695** | 0.7116 | +6.3% (退化) |
| `instability_score` | 2.153 | **1.236** | 1.281 | +3.6% (退化) |
| `per_segment.worst_vel_rmse_K` | — | 0.01961 | **0.01419** | **-27.6%** |
| `per_segment.ratio_worst_over_best` | — | 3.47 | **2.22** | **-36.0%** |
| `per_segment.high_speed_seg_mean` | — | 0.01525 | **0.00867** | **-43.1%** |

* PROMPT v3 §7 的 12 条阈值：**10 ✅ / 2 ❌**（G4 slope 与 S4 退化样本占比贴线未跨过）
* PROMPT v3 §6.5 3 阶字典效果归因：**3 ✅ / 0 ❌**（u_bias 漂移斜率 ×0.21，
  worst-seg ×0.72，高速段相对改善 ×0.60）
* 9.4 验收（v2 vs v3）：vel↓ 31% + slope 几乎不变 + inst 几乎不变 → A/B 两条件均满足

详见 [`test_analysis/compare_v1_v2_v3/compare_summary.md`](./test_analysis/compare_v1_v2_v3/compare_summary.md)、
[`test_analysis/v3_iterations.md`](./test_analysis/v3_iterations.md)（10 轮调参全过程）、
`compare_per_segment_bar.png`、`compare_u_bias_per_step.png`。
