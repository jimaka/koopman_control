# Deep-Koopman 速度跟踪 v2

`koopman.py` 中的物理先验 Koopman 模型 (`HorizontalKoopmanModel`)
是固定接口；本仓库提供两套训练 / 评估管线：

* **v1 baseline**：`train_multistep_voyage.py`（保留不动，供历史对照）。
* **v2 重写**：`train_koopman_v2.py` + `eval_koopman.py`，按
  [`PROMPT_deep_koopman_rewrite.md`](./PROMPT_deep_koopman_rewrite.md)
  第 1–9 节实现，目标是显著降低多步速度跟踪误差，并以**可量化、可证伪**的
  发散指标替代「看图凭感觉」。

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

| metric | v1 (baseline) | **v2** | Δ |
|---|---|---|---|
| `vel_rmse_step_1` [m/s] | 0.001314 | 0.001847 | +40.6% |
| `vel_rmse_step_5` [m/s] | 0.006549 | 0.007284 | +11.2% |
| `vel_rmse_step_10` [m/s] | 0.013021 | 0.010982 | **-15.7%** |
| `vel_rmse_step_20` [m/s] | 0.025644 | **0.016059** | **-37.4%** ✓ |
| `traj_xy_rmse_step_20` [m] | 0.027750 | **0.020081** | **-27.6%** |
| `slope_loglog` | 0.9915 | **0.6695** | **-32.5%** ✓ |
| `ratio_step20/1` | 19.52 | **8.69** | **-55.5%** |
| `instability_score` | 2.153 | **1.236** | **-42.6%** ✓ |
| `divergent_sample_pct` | 99.88% | **81.78%** | -18.1pp |

* 条件 A（vel↓≥30% 且 slope↓≥20%）：**True**
* 条件 B（inst↓≥25% 且 vel 不上升）：**True**
* **9.4 验收：✅ PASS**

详见 [`test_analysis/compare_v1_v2/compare_summary.md`](./test_analysis/compare_v1_v2/compare_summary.md)
与 `compare_error_vs_step.png`。
