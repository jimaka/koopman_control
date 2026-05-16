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

## How to train v3a (Plan-A)

按 [`PROMPT_deep_koopman_v3_planA.md`](./PROMPT_deep_koopman_v3_planA.md)
实现：**A1** per-channel `--w_bias_v` 调高 + **A2** `--best_metric composite_v3a`
（含 slope/degraded_pct 软惩罚）+ **A3** `--seg_resample u_var_r_var` 段间加权
sampler。

```bash
# Plan-A 推荐训练命令（含 A1+A2+A3）
python3 train_koopman_v2.py --model v3 \
    --epochs 60 --pred_len_start 4 --pred_len_max 20 --pred_len_grow_every 3 \
    --batch_size 1024 --num_workers 4 --no-amp --val_max_samples 8192 \
    --w_vel 1.0 --w_acc 0.5 --w_lin 0.0 --w_stab 5.0 \
    --w_bias_u 80.0 --w_bias_v 100.0 --w_bias_r 30.0 \
    --w_l2 1e-4 --gamma_step 1.10 --gamma_bias 1.10 --huber_beta 0.1 \
    --rho_max 0.999 --noise_std 0.02 --ctrl_noise_std 0.005 --clamp_pif 30.0 \
    --seg_resample u_var_r_var --seg_resample_alpha 1.0 \
    --baseline_ckpt checkpoints/koopman_v1_best.pth \
    --ema_decay 0.999 --best_metric composite_v3a \
    --run_tag v3a --out_dir test_analysis/v3a

# A2 离线重选（不重训，1 分钟内从已有 ckpt 池挑出 composite_v3a 最低者）
python3 scripts/reselect_best_v3a.py \
    --ckpt_glob 'checkpoints/koopman_v3_run*_best.pth' \
    --data koopman_val.npz \
    --baseline_ckpt checkpoints/koopman_v1_best.pth \
    --out checkpoints/koopman_v3a_reselect_best.pth
```

## How to evaluate v3a

```bash
# 单 ckpt 评估（v3a 必须显式给 --baseline_ckpt 才会在 summary.json 里
# 写入 s4 字段 = degraded_pct_vs_baseline）
python3 eval_koopman.py --ckpt checkpoints/koopman_v3a_best.pth \
    --data koopman_test.npz --pred_len 20 --tag v3a --out_dir test_analysis/v3a \
    --baseline_ckpt checkpoints/koopman_v1_best.pth

# v1/v2/v3/v3a 四角对比 + §5 14 条阈值 verdict + plan-A 三件事归因
python3 eval_koopman.py --compare \
    checkpoints/koopman_v1_best.pth:v1 \
    checkpoints/koopman_v2_best.pth:v2 \
    checkpoints/koopman_v3_best.pth:v3 \
    checkpoints/koopman_v3a_best.pth:v3a \
    --data koopman_test.npz --pred_len 20 \
    --out_dir test_analysis/compare_v2_v3_v3a
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

| metric | v1 | v2 | v3 | **v3a (plan-A)** | v3a vs v3 |
|---|---|---|---|---|---|
| `vel_rmse_step_1` [m/s] | 0.00131 | 0.00185 | **0.00122** | 0.00124 | +2% |
| `vel_rmse_step_20` [m/s] | 0.02564 | 0.01606 | **0.01107** | 0.01155 | +4% |
| `u_rmse_step_20` [m/s] | 0.02254 | 0.01580 | **0.01010** | 0.01108 | +10% |
| `v_rmse_step_20` [m/s] | 0.01223 | 0.00285 | 0.00452 | **0.00326** | **-28%** (A1 ✅) |
| `traj_xy_rmse_step_20` [m] | 0.02775 | 0.02008 | 0.01420 | **0.01348** | **-5%** |
| `|u_bias_mean|` [m/s] | 0.00365 | 0.00255 | 0.00038 | **3.9e-6** | **-99%** |
| `|v_bias_mean|` [m/s] | 8.7e-5 | 3.5e-4 | 1.5e-4 | **1.3e-4** | **-17%** (A1 ✅) |
| `slope_loglog` | 0.992 | **0.6695** | 0.7116 | 0.7066 | -1% |
| `instability_score` | 2.153 | **1.236** | 1.281 | 1.292 | +1% |
| `per_segment.worst_vel_rmse_K` | — | 0.01961 | 0.01419 | **0.01355** | **-5%** |
| `per_segment.ratio_worst_over_best` | — | 3.47 | 2.22 | **1.80** | **-19%** |
| `per_segment.high_speed_seg_mean` | — | 0.01525 | **0.00867** | 0.00954 | +10% |
| `degraded_pct vs v1` | — | — | 22.94% | 26.42% | +3.5pp |

* **PROMPT v3 §7 12 条阈值 (v3)**：10 ✅ / 2 ❌（G4 slope, S4 deg%）
* **PROMPT v3a §5 14 条阈值 (v3a)**：**11 ✅ / 3 ❌**（G4=0.707, G5=1.29, S4=26.4）
* **PROMPT v3 §6.5 3 阶字典效果归因 (v3 vs v2)**：3 ✅ / 0 ❌
* **PROMPT v3a §9 三件事归因 (v3a vs v3)**：
  * **A1 (v 通道压回 v2 水平)**：V1 `v_rmse@K` 0.00452 → **0.00326** (-28%) ✅；
    V2 `|v_bias|` 1.5e-4 → **1.3e-4** ✅
  * **A2 (composite_v3a 选 best)**：从 10 个 v3 run 池中挑出 `run05`（而非原 `run02`），
    slope 与 v 通道同时改善；离线重选脚本 `scripts/reselect_best_v3a.py` 1 分钟内完成
  * **A3 (seg_resample u_var_r_var)**：3 轮重训均观察到 alpha↑ → S4 ✅ 但 G4/G5 ❌ 的耦合 trade-off；
    最终 v3a_best 不启用 A3，靠 A2 选择已天然均衡的 run05 拿下 S1/S2/S3

详见
[`test_analysis/compare_v2_v3_v3a/compare_summary.md`](./test_analysis/compare_v2_v3_v3a/compare_summary.md)、
[`test_analysis/v3a_iterations.md`](./test_analysis/v3a_iterations.md)（3 轮重训 + 1 次离线重选全过程）、
[`test_analysis/v3_iterations.md`](./test_analysis/v3_iterations.md)（v3 历史 10 轮调参）、
[`test_analysis/compare_v1_v2_v3/compare_summary.md`](./test_analysis/compare_v1_v2_v3/compare_summary.md)。
